# src/strategies/implementations/bbands_volume_rsi_strategy_impl.py

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

class BbandsVolumeRsiFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class BbandsVolumeRsiOptimizableParams(BaseOptimizableParams):
    bb_period: int = Field(default=20, gt=0, description="Période pour les Bandes de Bollinger.")
    bb_std_dev: float = Field(default=2.0, gt=0, description="Écart-type pour les Bandes de Bollinger.")
    rsi_period: int = Field(default=14, gt=0, description="Période pour le RSI.")
    rsi_overbought: int = Field(default=70, gt=0, lt=100, description="Seuil de surachat du RSI.")
    rsi_oversold: int = Field(default=30, gt=0, lt=100, description="Seuil de survente du RSI.")
    volume_factor: float = Field(default=1.5, ge=0, description="Multiplicateur pour le volume moyen de confirmation.")
    volume_period: int = Field(default=20, gt=0, description="Période pour la SMA du volume.")
    position_sizing_pct_capital: float = Field(default=0.01, gt=0, lt=1, description="Pourcentage du capital à utiliser par trade.")

    @model_validator(mode='after')
    def check_rsi_logic(self) -> 'BbandsVolumeRsiOptimizableParams':
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold doit être inférieur à rsi_overbought.")
        return self

# --- Strategy Implementation ---

class BbandsVolumeRsiStrategy(BaseStrategy):
    name: str = "BbandsVolumeRsiStrategy"
    version: str = "2.0.0"
    description: str = "Stratégie combinant Bandes de Bollinger, RSI et confirmation par volume."

    fixed_params_model = BbandsVolumeRsiFixedParams
    optimizable_params_model = BbandsVolumeRsiOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)

        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = max(self.get_param('bb_period'), self.get_param('rsi_period')) + 1
        
        freq = self.get_param('indicator_frequency')
        bb_p = self.get_param('bb_period')
        bb_std = self.get_param('bb_std_dev')
        rsi_p = self.get_param('rsi_period')

        self.bb_lower_col = f"{freq}_BBL_{bb_p}_{bb_std}"
        self.bb_middle_col = f"{freq}_BBM_{bb_p}_{bb_std}"
        self.bb_upper_col = f"{freq}_BBU_{bb_p}_{bb_std}"
        self.rsi_col = f"{freq}_RSI_{rsi_p}"
        
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        df = self._prepare_indicator_data(data)

        indicator_mgr = IndicatorManager()
        indicator_configs = [
            {"name": "bbands", "length": self.get_param('bb_period'), "std": self.get_param('bb_std_dev'), "output_col_prefix": self.get_param('indicator_frequency')},
            {"name": "rsi", "length": self.get_param('rsi_period'), "output_col_prefix": self.get_param('indicator_frequency')},
            {"name": "sma", "close": "volume", "length": self.get_param('volume_period'), "output_col_name": "volume_sma"}
        ]
        
        df_with_indicators = indicator_mgr.calculate_multiple_indicators(df, indicator_configs)
        self._indicators_cache = df_with_indicators
        logger.info(f"{log_pref} Indicateurs calculés avec succès.")
        return df_with_indicators

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        if indicators_df.empty: return pd.DataFrame()
        df = indicators_df

        rsi_overbought = self.get_param('rsi_overbought')
        rsi_oversold = self.get_param('rsi_oversold')
        volume_factor = self.get_param('volume_factor')

        price_crosses_upper = (df['close'] > df[self.bb_upper_col]) & (df['close'].shift(1) <= df[self.bb_upper_col].shift(1))
        price_crosses_lower = (df['close'] < df[self.bb_lower_col]) & (df['close'].shift(1) >= df[self.bb_lower_col].shift(1))
        
        rsi_is_overbought = df[self.rsi_col] > rsi_overbought
        rsi_is_oversold = df[self.rsi_col] < rsi_oversold

        volume_confirms = df['volume'] > (df['volume_sma'] * volume_factor)

        signals = pd.DataFrame(index=df.index)
        signals['entry_long'] = price_crosses_lower & rsi_is_oversold & volume_confirms
        signals['entry_short'] = price_crosses_upper & rsi_is_overbought & volume_confirms
        
        price_crosses_middle_from_above = (df['close'] < df[self.bb_middle_col]) & (df['close'].shift(1) >= df[self.bb_middle_col].shift(1))
        price_crosses_middle_from_below = (df['close'] > df[self.bb_middle_col]) & (df['close'].shift(1) <= df[self.bb_middle_col].shift(1))
        
        signals['exit_long'] = price_crosses_middle_from_above
        signals['exit_short'] = price_crosses_middle_from_below
        
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
        latest, previous = df.iloc[-1], df.iloc[-2]
        
        required_cols = ['close', 'volume', self.bb_lower_col, self.bb_middle_col, self.bb_upper_col, self.rsi_col, 'volume_sma']
        if latest[required_cols].isnull().any() or previous[required_cols].isnull().any(): return None

        side: Optional[str] = None
        rsi_overbought, rsi_oversold = self.get_param('rsi_overbought'), self.get_param('rsi_oversold')
        volume_factor = self.get_param('volume_factor')

        price_crossed_upper = latest['close'] > latest[self.bb_upper_col] and previous['close'] <= previous[self.bb_upper_col]
        price_crossed_lower = latest['close'] < latest[self.bb_lower_col] and previous['close'] >= previous[self.bb_lower_col]
        rsi_is_overbought = latest[self.rsi_col] > rsi_overbought
        rsi_is_oversold = latest[self.rsi_col] < rsi_oversold
        volume_confirms = latest['volume'] > (latest['volume_sma'] * volume_factor)
        
        price_crossed_middle_from_above = latest['close'] < latest[self.bb_middle_col] and previous['close'] >= previous[self.bb_middle_col]
        price_crossed_middle_from_below = latest['close'] > latest[self.bb_middle_col] and previous['close'] <= previous[self.bb_middle_col]

        if current_position == 0:
            if price_crossed_lower and rsi_is_oversold and volume_confirms: side = Trading.SIDE_BUY
            elif price_crossed_upper and rsi_is_overbought and volume_confirms: side = Trading.SIDE_SELL
        elif current_position > 0 and price_crossed_middle_from_above: side = Trading.SIDE_SELL
        elif current_position < 0 and price_crossed_middle_from_below: side = Trading.SIDE_BUY
        
        if not side: return None
        
        capital_to_use = available_capital * self.get_param('position_sizing_pct_capital')
        quantity = capital_to_use / latest['close']

        qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8
        final_quantity = adjust_precision(quantity, qty_precision, np.floor)

        if final_quantity <= 0: return None

        order_params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": f"{final_quantity:.{qty_precision}f}"}
        sl_tp_params = {}
        
        logger.info(f"{log_pref} Requête d'ordre générée: {order_params}")
        return order_params, sl_tp_params
