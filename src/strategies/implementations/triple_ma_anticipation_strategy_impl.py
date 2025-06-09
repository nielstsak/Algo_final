# src/strategies/implementations/triple_ma_anticipation_strategy_impl.py
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, model_validator

from src.strategies.base_strategy import BaseStrategy
from src.core.constants import Trading
from src.core.exceptions import InvalidStrategyParamsError
from src.strategies.params import BaseFixedParams, BaseOptimizableParams
from src.strategies.technical_indicators import IndicatorManager
from src.utils.exchange_utils import get_precision_from_filter, adjust_precision
from src.data.enriched_dataframe import EnrichedDataFrame

# --- Pydantic Parameter Models ---

class TripleMAAnticipationFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class TripleMAAnticipationOptimizableParams(BaseOptimizableParams):
    fast_period: int = Field(default=10, gt=0, description="Période de la SMA rapide.")
    medium_period: int = Field(default=20, gt=0, description="Période de la SMA moyenne.")
    slow_period: int = Field(default=50, gt=0, description="Période de la SMA lente.")
    stop_loss_pct: float = Field(default=0.03, gt=0, lt=1, description="Pourcentage de Stop Loss.")
    take_profit_pct: float = Field(default=0.06, gt=0, lt=1, description="Pourcentage de Take Profit.")
    position_sizing_pct_capital: float = Field(default=0.01, gt=0, lt=1, description="Pourcentage du capital à risquer.")
    anticipation_threshold_pct: float = Field(default=0.001, ge=0, description="Seuil d'anticipation du croisement (en % des prix).")

    @model_validator(mode='after')
    def check_periods_logic(self) -> 'TripleMAAnticipationOptimizableParams':
        if not (self.fast_period < self.medium_period < self.slow_period):
            raise ValueError("Les périodes des SMA doivent être dans l'ordre croissant : fast < medium < slow.")
        return self

# --- Strategy Implementation ---

class TripleMAAnticipationStrategy(BaseStrategy):
    name: str = "TripleMAAnticipationStrategy"
    version: str = "1.1.0"
    description: str = "Stratégie de triple SMA qui anticipe les croisements."

    fixed_params_model = TripleMAAnticipationFixedParams
    optimizable_params_model = TripleMAAnticipationOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)

        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = self.get_param('slow_period') + 2
        
        freq = self.get_param('indicator_frequency')
        self.fast_sma_col = f"{freq}_SMA_{self.get_param('fast_period')}"
        self.medium_sma_col = f"{freq}_SMA_{self.get_param('medium_period')}"
        self.slow_sma_col = f"{freq}_SMA_{self.get_param('slow_period')}"

        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        df = self._prepare_indicator_data(data)
        indicator_mgr = IndicatorManager()
        indicator_configs = [
            {"name": "sma", "length": self.get_param('fast_period'), "output_col_prefix": self.get_param('indicator_frequency')},
            {"name": "sma", "length": self.get_param('medium_period'), "output_col_prefix": self.get_param('indicator_frequency')},
            {"name": "sma", "length": self.get_param('slow_period'), "output_col_prefix": self.get_param('indicator_frequency')},
        ]
        df_with_indicators = indicator_mgr.calculate_multiple_indicators(df, indicator_configs)
        self._indicators_cache = df_with_indicators
        logger.info(f"{self.strategy_name_log_prefix} Indicateurs calculés.")
        return df_with_indicators

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        if indicators_df.empty: return pd.DataFrame()
        df = indicators_df
        threshold = self.get_param('anticipation_threshold_pct')

        # Conditions d'entrée
        golden_cross_confirmed = (df[self.fast_sma_col] > df[self.medium_sma_col]) & (df[self.medium_sma_col] > df[self.slow_sma_col])
        anticipation_long = df[self.fast_sma_col] > (df[self.medium_sma_col] * (1 - threshold))
        entry_long = golden_cross_confirmed & anticipation_long

        death_cross_confirmed = (df[self.fast_sma_col] < df[self.medium_sma_col]) & (df[self.medium_sma_col] < df[self.slow_sma_col])
        anticipation_short = df[self.fast_sma_col] < (df[self.medium_sma_col] * (1 + threshold))
        entry_short = death_cross_confirmed & anticipation_short
        
        # Conditions de sortie (croisement simple fast/medium)
        exit_long = (df[self.fast_sma_col] < df[self.medium_sma_col]) & (df[self.fast_sma_col].shift(1) >= df[self.medium_sma_col].shift(1))
        exit_short = (df[self.fast_sma_col] > df[self.medium_sma_col]) & (df[self.fast_sma_col].shift(1) <= df[self.medium_sma_col].shift(1))

        signals = pd.DataFrame(index=df.index)
        signals['entry_long'], signals['entry_short'] = entry_long, entry_short
        signals['exit_long'], signals['exit_short'] = exit_long, exit_short

        self._signals = signals
        return self._signals.copy()

    def generate_order_request(
        self, data_dict: Dict[str, pd.DataFrame], symbol: str, current_position: int,
        available_capital: float, symbol_info: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        log_pref = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        primary_freq = self.get_param('indicator_frequency')
        if primary_freq not in data_dict or len(data_dict[primary_freq]) < 2: return None
            
        df = data_dict[primary_freq]
        latest = df.iloc[-1]
        
        required_cols = [self.fast_sma_col, self.medium_sma_col, self.slow_sma_col, 'close']
        if latest[required_cols].isnull().any(): return None

        side: Optional[str] = None
        threshold = self.get_param('anticipation_threshold_pct')

        golden_cross = latest[self.fast_sma_col] > latest[self.medium_sma_col] > latest[self.slow_sma_col]
        death_cross = latest[self.fast_sma_col] < latest[self.medium_sma_col] < latest[self.slow_sma_col]
        
        anticipate_long = latest[self.fast_sma_col] > (latest[self.medium_sma_col] * (1 - threshold))
        anticipate_short = latest[self.fast_sma_col] < (latest[self.medium_sma_col] * (1 + threshold))
        
        exit_long_cond = latest[self.fast_sma_col] < latest[self.medium_sma_col]
        exit_short_cond = latest[self.fast_sma_col] > latest[self.medium_sma_col]

        if current_position == 0:
            if golden_cross and anticipate_long: side = Trading.SIDE_BUY
            elif death_cross and anticipate_short: side = Trading.SIDE_SELL
        elif current_position > 0 and exit_long_cond: side = Trading.SIDE_SELL
        elif current_position < 0 and exit_short_cond: side = Trading.SIDE_BUY
        
        if not side: return None

        if (current_position > 0 and side == Trading.SIDE_SELL) or \
           (current_position < 0 and side == Trading.SIDE_BUY):
            return {"symbol": symbol, "side": side, "type": "MARKET", "quantity": "CLOSE_POSITION"}, {}

        entry_price = latest['close']
        sl_pct, tp_pct = self.get_param('stop_loss_pct'), self.get_param('take_profit_pct')
        
        if side == Trading.SIDE_BUY:
            sl_price, tp_price = entry_price * (1 - sl_pct), entry_price * (1 + tp_pct)
        else: # SIDE_SELL
            sl_price, tp_price = entry_price * (1 + sl_pct), entry_price * (1 - tp_pct)
            
        capital_to_risk = available_capital * self.get_param('position_sizing_pct_capital')
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit < 1e-9: return None
        
        quantity = capital_to_risk / risk_per_unit
        price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize') or 8
        qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8

        final_quantity = adjust_precision(quantity, qty_precision, np.floor)
        if final_quantity <= 0: return None

        order_params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": f"{final_quantity:.{qty_precision}f}"}
        sl_tp_params = {'sl_price': float(sl_price), 'tp_price': float(tp_price)}
        
        logger.info(f"{log_pref} Requête d'ordre générée: {order_params}, SL/TP: {sl_tp_params}")
        return order_params, sl_tp_params
