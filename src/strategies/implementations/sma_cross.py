# src/strategies/implementations/sma_cross.py
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, model_validator

from src.strategies.base_strategy import BaseStrategy
from src.core.constants import Trading
from src.core.exceptions import InvalidStrategyParamsError, SignalGenerationError
from src.strategies.params import BaseFixedParams, BaseOptimizableParams
from src.strategies.technical_indicators import IndicatorManager
from src.utils.exchange_utils import get_precision_from_filter, adjust_precision
from src.data.enriched_dataframe import EnrichedDataFrame

# --- Pydantic Parameter Models ---

class SMACrossFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class SMACrossOptimizableParams(BaseOptimizableParams):
    fast_period: int = Field(default=10, gt=0, description="Période de la SMA rapide.")
    slow_period: int = Field(default=30, gt=0, description="Période de la SMA lente.")
    stop_loss_pct: float = Field(default=0.02, gt=0, lt=1, description="Pourcentage de Stop Loss.")
    take_profit_pct: float = Field(default=0.04, gt=0, lt=1, description="Pourcentage de Take Profit.")
    position_sizing_pct_capital: float = Field(default=0.02, gt=0, lt=1, description="Pourcentage du capital à risquer par trade.")

    @model_validator(mode='after')
    def check_periods_logic(self) -> 'SMACrossOptimizableParams':
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period doit être inférieur à slow_period.")
        return self

# --- Strategy Implementation ---

class SMACrossStrategy(BaseStrategy):
    name: str = "SMACrossStrategy"
    version: str = "3.0.0"
    description: str = "Stratégie de croisement de SMA avec configuration Pydantic."

    # Link to Pydantic models
    fixed_params_model = SMACrossFixedParams
    optimizable_params_model = SMACrossOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)
        
        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = self.get_param('slow_period') + 2
        
        freq = self.get_param('indicator_frequency')
        fast_p = self.get_param('fast_period')
        slow_p = self.get_param('slow_period')
        
        self.fast_sma_col = f"{freq}_SMA_{fast_p}"
        self.slow_sma_col = f"{freq}_SMA_{slow_p}"
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    # validate_params is now handled by BaseStrategy and Pydantic models

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        df = self._prepare_indicator_data(data)
        
        indicator_mgr = IndicatorManager()
        indicator_configs = [
            {"name": "sma", "length": self.get_param('fast_period'), "output_col_prefix": self.get_param('indicator_frequency')},
            {"name": "sma", "length": self.get_param('slow_period'), "output_col_prefix": self.get_param('indicator_frequency')},
        ]
        
        df_with_indicators = indicator_mgr.calculate_multiple_indicators(df, indicator_configs)
        self._indicators_cache = df_with_indicators
        logger.info(f"{log_pref} Indicateurs calculés avec succès.")
        return df_with_indicators

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        if indicators_df.empty: return pd.DataFrame()

        df = indicators_df
        required_cols = [self.fast_sma_col, self.slow_sma_col, 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise SignalGenerationError(f"Colonnes requises manquantes: {missing}", strategy_name=self.name)

        fast_sma = df[self.fast_sma_col]
        slow_sma = df[self.slow_sma_col]

        entry_long = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        entry_short = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        
        signals = pd.DataFrame(index=df.index)
        signals['entry_long'], signals['entry_short'] = entry_long, entry_short
        signals['exit_long'], signals['exit_short'] = entry_short, entry_long
        
        exit_and_entry_long = signals['exit_long'] & signals['entry_long']
        signals.loc[exit_and_entry_long, 'entry_long'] = False
        exit_and_entry_short = signals['exit_short'] & signals['entry_short']
        signals.loc[exit_and_entry_short, 'entry_short'] = False

        sl_pct, tp_pct = self.get_param('stop_loss_pct'), self.get_param('take_profit_pct')
        entry_price = df['close']

        signals['sl'] = np.where(entry_long, entry_price * (1 - sl_pct), np.where(entry_short, entry_price * (1 + sl_pct), np.nan))
        signals['tp'] = np.where(entry_long, entry_price * (1 + tp_pct), np.where(entry_short, entry_price * (1 - tp_pct), np.nan))
        
        logger.info(f"{log_pref} Signaux générés. Entrées Long: {signals['entry_long'].sum()}, Entrées Short: {signals['entry_short'].sum()}.")
        self._signals = signals
        return self._signals.copy()

    def generate_order_request(
        self, data_dict: Dict[str, pd.DataFrame], symbol: str, current_position: int,
        available_capital: float, symbol_info: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        log_pref = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        primary_freq = self.get_param('indicator_frequency')

        if primary_freq not in data_dict or len(data_dict[primary_freq]) < 2:
            logger.warning(f"{log_pref} Données pour '{primary_freq}' manquantes ou insuffisantes.")
            return None
            
        df = data_dict[primary_freq]
        latest, previous = df.iloc[-1], df.iloc[-2]
        
        required_cols = [self.fast_sma_col, self.slow_sma_col, 'close']
        if latest[required_cols].isnull().any() or previous[required_cols].isnull().any():
            logger.warning(f"{log_pref} Indicateurs NaN sur les dernières données.")
            return None

        fast_sma_curr, slow_sma_curr = latest[self.fast_sma_col], latest[self.slow_sma_col]
        fast_sma_prev, slow_sma_prev = previous[self.fast_sma_col], previous[self.slow_sma_col]
        
        side: Optional[str] = None
        if current_position == 0:
            if fast_sma_curr > slow_sma_curr and fast_sma_prev <= slow_sma_prev: side = Trading.SIDE_BUY
            elif fast_sma_curr < slow_sma_curr and fast_sma_prev >= slow_sma_prev: side = Trading.SIDE_SELL
        elif current_position > 0 and fast_sma_curr < slow_sma_curr: side = Trading.SIDE_SELL
        elif current_position < 0 and fast_sma_curr > slow_sma_curr: side = Trading.SIDE_BUY

        if not side: return None
        
        if (current_position > 0 and side == Trading.SIDE_SELL) or \
           (current_position < 0 and side == Trading.SIDE_BUY):
            order_params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": "CLOSE_POSITION"}
            return order_params, {}

        sl_pct, tp_pct = self.get_param('stop_loss_pct'), self.get_param('take_profit_pct')
        entry_price = latest['close']
        
        if side == Trading.SIDE_BUY:
            sl_price, tp_price = entry_price * (1 - sl_pct), entry_price * (1 + tp_pct)
        else:
            sl_price, tp_price = entry_price * (1 + sl_pct), entry_price * (1 - tp_pct)
        
        capital_to_risk = available_capital * self.get_param('position_sizing_pct_capital')
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit < 1e-9: return None
        
        quantity = capital_to_risk / risk_per_unit
        price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize') or 8
        qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8

        final_quantity = adjust_precision(quantity, qty_precision, np.floor)
        final_price = adjust_precision(entry_price, price_precision, round)
        final_sl = adjust_precision(sl_price, price_precision, round)
        final_tp = adjust_precision(tp_price, price_precision, round)
        
        if not all([final_quantity, final_price, final_sl, final_tp]) or final_quantity <= 0: return None
        
        order_params = {"symbol": symbol, "side": side, "type": "LIMIT", "quantity": f"{final_quantity:.{qty_precision}f}", "price": f"{final_price:.{price_precision}f}"}
        sl_tp_params = {'sl_price': float(final_sl), 'tp_price': float(final_tp)}
        
        logger.info(f"{log_pref} Requête d'ordre générée: {order_params}, SL/TP: {sl_tp_params}")
        return order_params, sl_tp_params
