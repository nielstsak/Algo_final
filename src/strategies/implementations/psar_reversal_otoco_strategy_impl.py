# src/strategies/implementations/psar_reversal_otoco_strategy_impl.py
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field

from src.strategies.base_strategy import BaseStrategy
from src.core.constants import Trading, DataFrameCols
from src.core.exceptions import SignalGenerationError
from src.strategies.params import BaseFixedParams, BaseOptimizableParams
from src.strategies.technical_indicators import TechnicalIndicators
from src.utils.exchange_utils import get_precision_from_filter, adjust_precision
from src.data.enriched_dataframe import EnrichedDataFrame

# --- Pydantic Parameter Models ---

class PsarReversalOtocoFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class PsarReversalOtocoOptimizableParams(BaseOptimizableParams):
    psar_inc: float = Field(default=0.02, gt=0, description="Incrément pour le Parabolic SAR.")
    psar_max: float = Field(default=0.2, gt=0, description="Valeur maximale pour le Parabolic SAR.")
    atr_period: int = Field(default=14, gt=0, description="Période de l'ATR pour le calcul du SL/TP.")
    atr_multiplier_sl: float = Field(default=2.0, gt=0, description="Multiplicateur de l'ATR pour le Stop Loss.")
    atr_multiplier_tp: float = Field(default=3.0, gt=0, description="Multiplicateur de l'ATR pour le Take Profit.")
    otoco_candle_body_ratio_thld: float = Field(default=0.7, ge=0, le=1, description="Seuil du ratio corps/range pour le pattern OTOCO.")
    otoco_wick_ratio_thld: float = Field(default=0.1, ge=0, le=1, description="Seuil du ratio mèche/range pour le pattern OTOCO.")
    position_sizing_pct_capital: float = Field(default=0.01, gt=0, lt=1, description="Pourcentage du capital à risquer par trade.")

# --- Strategy Implementation ---

class PsarReversalOtocoStrategy(BaseStrategy):
    name: str = "PsarReversalOtocoStrategy"
    version: str = "3.0.0"
    description: str = "Combine les retournements de PSAR avec le pattern OTOCO pour confirmation."

    fixed_params_model = PsarReversalOtocoFixedParams
    optimizable_params_model = PsarReversalOtocoOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)
        
        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = self.get_param('atr_period') + 2
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        df = self._prepare_indicator_data(data).copy()
        
        psar_df = TechnicalIndicators.psar(df['high'], df['low'], df['close'], inc=self.get_param('psar_inc'), max_af=self.get_param('psar_max'))
        df = pd.concat([df, psar_df], axis=1)

        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=self.get_param('atr_period'))

        df['otoco_long'], df['otoco_short'] = TechnicalIndicators.otoco_pattern(
            df['open'], df['high'], df['low'], df['close'],
            body_ratio_thld=self.get_param('otoco_candle_body_ratio_thld'),
            wick_ratio_thld=self.get_param('otoco_wick_ratio_thld')
        )
        self._indicators_cache = df
        logger.info(f"{self.strategy_name_log_prefix} Indicateurs calculés.")
        return df

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        df = indicators_df.copy()
        
        df['psar_reversal_up'] = (df['close'] > df['psar']) & (df['close'].shift(1) < df['psar'].shift(1))
        df['psar_reversal_down'] = (df['close'] < df['psar']) & (df['close'].shift(1) > df['psar'].shift(1))

        df[DataFrameCols.ENTRY_LONG.value] = df['psar_reversal_up'] & df['otoco_long']
        df[DataFrameCols.ENTRY_SHORT.value] = df['psar_reversal_down'] & df['otoco_short']
        
        df[DataFrameCols.EXIT_LONG.value] = df['psar_reversal_down']
        df[DataFrameCols.EXIT_SHORT.value] = df['psar_reversal_up']

        sl_mult = self.get_param('atr_multiplier_sl')
        tp_mult = self.get_param('atr_multiplier_tp')

        df['sl'] = np.nan
        df['tp'] = np.nan

        long_entries = df[DataFrameCols.ENTRY_LONG.value]
        df.loc[long_entries, 'sl'] = df['close'] - (df['atr'] * sl_mult)
        df.loc[long_entries, 'tp'] = df['close'] + (df['atr'] * tp_mult)

        short_entries = df[DataFrameCols.ENTRY_SHORT.value]
        df.loc[short_entries, 'sl'] = df['close'] + (df['atr'] * sl_mult)
        df.loc[short_entries, 'tp'] = df['close'] - (df['atr'] * tp_mult)
        
        self._signals = df
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
        
        required_cols = ['close', 'psar', 'otoco_long', 'otoco_short', 'atr']
        if latest[required_cols].isnull().any() or previous[required_cols].isnull().any():
            logger.warning(f"{log_pref} Indicateurs NaN sur les dernières données.")
            return None

        side: Optional[str] = None
        
        psar_reversal_up = (latest['close'] > latest['psar']) and (previous['close'] < previous['psar'])
        psar_reversal_down = (latest['close'] < latest['psar']) and (previous['close'] > previous['psar'])

        if current_position == 0:
            if psar_reversal_up and latest['otoco_long']: side = Trading.SIDE_BUY
            elif psar_reversal_down and latest['otoco_short']: side = Trading.SIDE_SELL
        elif current_position > 0 and psar_reversal_down: side = Trading.SIDE_SELL
        elif current_position < 0 and psar_reversal_up: side = Trading.SIDE_BUY
        
        if not side: return None

        if (current_position > 0 and side == Trading.SIDE_SELL) or \
           (current_position < 0 and side == Trading.SIDE_BUY):
            return {"symbol": symbol, "side": side, "type": "MARKET", "quantity": "CLOSE_POSITION"}, {}

        entry_price = latest['close']
        atr_val = latest['atr']
        sl_mult = self.get_param('atr_multiplier_sl')
        tp_mult = self.get_param('atr_multiplier_tp')

        if side == Trading.SIDE_BUY:
            sl_price = entry_price - atr_val * sl_mult
            tp_price = entry_price + atr_val * tp_mult
        else:
            sl_price = entry_price + atr_val * sl_mult
            tp_price = entry_price - atr_val * tp_mult
            
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
