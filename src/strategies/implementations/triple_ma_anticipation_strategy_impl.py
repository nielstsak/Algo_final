# src/strategies/implementations/triple_ma_anticipation_strategy_impl.py
import logging # Standard logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# Attempt to import the correct BaseStrategy and other necessary modules
try:
    from ..base_strategy import BaseStrategy # Relative import for strategy loader
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError # Correct path
    from src.core.constants import Trading, Kline # Correct path
    from src.utils.exchange_utils import ( 
        adjust_precision,
        get_filter_value, # Make sure this is defined in exchange_utils if used
        get_precision_from_filter # Make sure this is defined in exchange_utils if used
    )
except ImportError as e:
    logging.getLogger(__name__).critical(
        "TripleMAAnticipationStrategy: CRITICAL - Failed to import BaseStrategy or other core modules. "
        "Ensure 'src' is in PYTHONPATH and all dependencies are installed. Error: %s", e
    )
    if 'BaseStrategy' not in globals():
        from abc import ABC, abstractmethod
        class BaseStrategy(ABC): # type: ignore
            name: str = "DummyBaseForTripleMA"
            version: str = "0.0.0"
            description: str = "Dummy BaseStrategy (Import Failed)"
            default_params: Dict[str, Any] = {}
            required_timeframes: List[str] = []
            min_required_periods: int = 1

            def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs): 
                self.params = self.default_params.copy()
                if params: self.params.update(params)
                self.params.update(kwargs)

            def validate_params(self) -> None: pass
            def get_param(self, key: str, default: Any = None) -> Any: return self.params.get(key, default)
            @abstractmethod
            def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]: raise NotImplementedError
            @abstractmethod
            def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame: raise NotImplementedError
            def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None # type: ignore
            def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None # type: ignore
    
        if 'InvalidStrategyParamsError' not in globals(): class InvalidStrategyParamsError(Exception): pass # type: ignore
        if 'StrategyError' not in globals(): class StrategyError(Exception): pass # type: ignore
        if 'get_precision_from_filter' not in globals(): def get_precision_from_filter(s, f, k) -> Optional[int]: return 8 # type: ignore
        if 'adjust_precision' not in globals(): def adjust_precision(v, p, r=round) -> Optional[float]: return round(v,p) if v is not None else None # type: ignore
        if 'get_filter_value' not in globals(): def get_filter_value(s,f,k) -> Optional[Any]: return None # type: ignore


logger = logging.getLogger(__name__)

class TripleMAAnticipationStrategy(BaseStrategy):
    name: str = "TripleMAAnticipationStrategy"
    version: str = "1.0.3" # Updated version
    description: str = "Stratégie d'anticipation de croisement de trois moyennes mobiles avec SL/TP basé sur ATR."

    default_params: Dict[str, Any] = {
        'short_ma_period': 5,
        'medium_ma_period': 10,
        'long_ma_period': 20,
        'indicator_frequency': '1h', 
        'atr_period_sl_tp': 14,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
        'anticipation_candles': 1, 
        'position_sizing_pct_capital': 0.02,
    }

    required_timeframes: List[str] = ['1m'] 
    min_required_periods: int = 25 

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        super().__init__(params, **kwargs)
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"

        freq = self.get_param('indicator_frequency')
        self.short_ma_col = f"SMA_{freq}_p{self.get_param('short_ma_period')}"
        self.medium_ma_col = f"SMA_{freq}_p{self.get_param('medium_ma_period')}"
        self.long_ma_col = f"SMA_{freq}_p{self.get_param('long_ma_period')}"
        self.atr_col = f"ATR_{freq}_p{self.get_param('atr_period_sl_tp')}"
        
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")
        logger.debug(f"{self.strategy_name_log_prefix} Expected indicator columns: "
                     f"ShortMA='{self.short_ma_col}', MediumMA='{self.medium_ma_col}', LongMA='{self.long_ma_col}', "
                     f"ATR='{self.atr_col}'")

    def validate_params(self) -> None:
        """Valide les paramètres de la stratégie."""
        short_p = self.get_param('short_ma_period')
        medium_p = self.get_param('medium_ma_period')
        long_p = self.get_param('long_ma_period')
        atr_p = self.get_param('atr_period_sl_tp')
        sl_mult = self.get_param('sl_atr_mult')
        tp_mult = self.get_param('tp_atr_mult')
        antic_candles = self.get_param('anticipation_candles')
        indicator_freq = self.get_param('indicator_frequency')

        if not (isinstance(short_p, int) and short_p > 0 and \
                isinstance(medium_p, int) and medium_p > 0 and \
                isinstance(long_p, int) and long_p > 0):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="ma_periods", details="Must be positive integers.")
        if not (short_p < medium_p < long_p):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="ma_periods", details="Must be in increasing order: short < medium < long.")
        if not (isinstance(atr_p, int) and atr_p > 0):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="atr_period_sl_tp", details="Must be a positive integer.")
        if not (isinstance(sl_mult, (float, int)) and sl_mult > 0) or \
           not (isinstance(tp_mult, (float, int)) and tp_mult > 0):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="sl_atr_mult/tp_atr_mult", details="Must be positive numbers.")
        if not (isinstance(antic_candles, int) and antic_candles >= 0): # Correction: >= 0 au lieu de > 0 si 0 est permis
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="anticipation_candles", details="Must be a non-negative integer.")

        valid_freqs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'] 
        if indicator_freq not in valid_freqs:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='indicator_frequency', details=f"Invalid frequency '{indicator_freq}'. Supported: {valid_freqs}")

        self.min_required_periods = max(long_p, atr_p) + antic_candles + 2 
        logger.debug(f"{self.strategy_name_log_prefix} Parameters validated. min_required_periods set to {self.min_required_periods}.")

    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        log_pref = self.strategy_name_log_prefix
        primary_freq = self.get_param('indicator_frequency')

        if primary_freq not in klines or klines[primary_freq] is None:
            raise StrategyError(f"{log_pref} DataFrame for '{primary_freq}' not in klines.", strategy_name=self.name)
        
        df_primary = klines[primary_freq].copy()
        logger.debug(f"{log_pref} Using data for '{primary_freq}'. Columns: {list(df_primary.columns)}")

        ohlc_map = {'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close', 'base_asset_volume': 'volume'}
        for src, target in ohlc_map.items():
            if src in df_primary.columns and target not in df_primary.columns:
                df_primary.rename(columns={src: target}, inplace=True)
            if target in df_primary.columns:
                 df_primary[target] = pd.to_numeric(df_primary[target], errors='coerce')
        
        expected_cols = [self.short_ma_col, self.medium_ma_col, self.long_ma_col, self.atr_col, 'volume'] # Ajout de 'volume' si utilisé
        missing = [col for col in expected_cols if col not in df_primary.columns]
        if missing:
            logger.warning(f"{log_pref} Missing expected indicator columns in '{primary_freq}' DF: {missing}.")

        self._indicators_cache = {primary_freq: df_primary}
        for tf, df_tf in klines.items():
            if tf != primary_freq and tf not in self._indicators_cache:
                self._indicators_cache[tf] = df_tf.copy()
        
        logger.debug(f"{log_pref} Indicators prepared. Main DF columns: {list(df_primary.columns)}")
        return self._indicators_cache

    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        primary_freq = self.get_param('indicator_frequency')

        if primary_freq not in indicators or indicators[primary_freq] is None:
            logger.error(f"{log_pref} DataFrame for '{primary_freq}' not in indicators dict.")
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).astype(bool)

        df = indicators[primary_freq].copy()
        logger.debug(f"{log_pref} Generating signals on data from '{primary_freq}'. Shape: {df.shape}")

        required_cols = ['close', self.short_ma_col, self.medium_ma_col, self.long_ma_col, self.atr_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"{log_pref} Missing columns for signal generation: {missing}. Returning empty signals.")
            return pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).fillna(False)

        if df.empty or any(df[col].isnull().all() for col in required_cols) or len(df) < self.min_required_periods:
            logger.warning(f"{log_pref} Insufficient data or all-NaN essential columns. Shape: {df.shape}")
            return pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).fillna(False)

        short_ma = df[self.short_ma_col]
        medium_ma = df[self.medium_ma_col]
        long_ma = df[self.long_ma_col]
        atr = df[self.atr_col]
        close_price = df['close']
        antic_candles = self.get_param('anticipation_candles')

        # Conditions d'alignement haussier des MMs
        bullish_alignment_now = (short_ma > medium_ma) & (medium_ma > long_ma)
        # Conditions d'alignement baissier des MMs
        bearish_alignment_now = (short_ma < medium_ma) & (medium_ma < long_ma)

        # Pour l'anticipation, on regarde si les conditions étaient différentes `antic_candles` auparavant
        # et si elles sont maintenant alignées ou sur le point de s'aligner.
        
        # Anticipation de croisement haussier:
        # Short MA était en dessous de Medium MA il y a `antic_candles` bougies
        # ET Short MA est maintenant au-dessus de Medium MA
        # ET Medium MA est (ou était récemment) au-dessus de Long MA (tendance de fond haussière)
        
        # Condition1: Short MA a croisé Medium MA à la hausse récemment
        short_crossed_medium_up = (short_ma > medium_ma) & (short_ma.shift(antic_candles) <= medium_ma.shift(antic_candles))
        # Condition2: Tendance de fond haussière (Medium MA > Long MA)
        medium_above_long = medium_ma > long_ma
        entry_long_cond = short_crossed_medium_up & medium_above_long & ~bullish_alignment_now.shift(1).fillna(False)


        # Anticipation de croisement baissier:
        short_crossed_medium_down = (short_ma < medium_ma) & (short_ma.shift(antic_candles) >= medium_ma.shift(antic_candles))
        medium_below_long = medium_ma < long_ma
        entry_short_cond = short_crossed_medium_down & medium_below_long & ~bearish_alignment_now.shift(1).fillna(False)
        
        signals_df = pd.DataFrame(index=df.index)
        signals_df['entry_long'] = entry_long_cond.fillna(False)
        signals_df['entry_short'] = entry_short_cond.fillna(False)

        # Sorties: peuvent être un croisement inverse ou gérées par SL/TP
        # Sortie de LONG si alignement devient baissier
        signals_df['exit_long'] = bearish_alignment_now & ~bearish_alignment_now.shift(1).fillna(False)
        # Sortie de SHORT si alignement devient haussier
        signals_df['exit_short'] = bullish_alignment_now & ~bullish_alignment_now.shift(1).fillna(False)
        
        sl_mult = self.get_param('sl_atr_mult')
        tp_mult = self.get_param('tp_atr_mult')

        signals_df['sl'] = np.where(
            signals_df['entry_long'], close_price - sl_mult * atr,
            np.where(signals_df['entry_short'], close_price + sl_mult * atr, np.nan)
        )
        signals_df['tp'] = np.where(
            signals_df['entry_long'], close_price + tp_mult * atr,
            np.where(signals_df['entry_short'], close_price - tp_mult * atr, np.nan)
        )

        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            signals_df[col_b] = signals_df[col_b].fillna(False).astype(bool)

        self._signals = signals_df
        logger.debug(f"{log_pref} Signaux générés. Longs: {signals_df['entry_long'].sum()}, Shorts: {signals_df['entry_short'].sum()}")
        return self._signals.copy()

    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame],
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        primary_freq = self.get_param('indicator_frequency')

        if primary_freq not in data_dict or data_dict[primary_freq] is None or data_dict[primary_freq].empty:
            logger.warning(f"{log_prefix_live} Données pour '{primary_freq}' manquantes ou vides.")
            return None
            
        df_indicators = data_dict[primary_freq]
        logger.info(f"{log_prefix_live} Appel generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")

        antic_candles = self.get_param('anticipation_candles')
        if len(df_indicators) < max(self.min_required_periods, antic_candles + 2): # +2 for shift(1) and current
            logger.warning(f"{log_prefix_live} Données d'indicateurs insuffisantes. Lignes: {len(df_indicators)}, Requis: {max(self.min_required_periods, antic_candles + 2)}. Pas d'ordre.")
            return None

        latest_data = df_indicators.iloc[-1]
        # Données antérieures pour l'anticipation et le shift(1)
        shifted_data_antic = df_indicators.iloc[-(antic_candles + 1)] if len(df_indicators) >= antic_candles + 1 else None
        previous_data_for_alignment_check = df_indicators.iloc[-2] if len(df_indicators) >= 2 else None


        required_cols_live = ['close', self.short_ma_col, self.medium_ma_col, self.long_ma_col, self.atr_col]
        if latest_data[required_cols_live].isnull().any() or \
           (shifted_data_antic is not None and shifted_data_antic[required_cols_live[:-1]].isnull().any()) or \
           (previous_data_for_alignment_check is not None and previous_data_for_alignment_check[required_cols_live[:-1]].isnull().any()): # ATR not needed for prev alignment
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur données récentes/décalées. Pas d'ordre.")
            return None
        
        short_ma_curr = latest_data[self.short_ma_col]
        medium_ma_curr = latest_data[self.medium_ma_col]
        long_ma_curr = latest_data[self.long_ma_col]
        atr_curr = latest_data[self.atr_col]
        close_curr = latest_data['close']

        side: Optional[str] = None
        entry_price_for_order = close_curr

        # Conditions d'alignement actuelles
        current_bullish_alignment = (short_ma_curr > medium_ma_curr) and (medium_ma_curr > long_ma_curr)
        current_bearish_alignment = (short_ma_curr < medium_ma_curr) and (medium_ma_curr < long_ma_curr)
        
        # Conditions d'alignement précédentes (t-1)
        prev_bullish_alignment = False
        prev_bearish_alignment = False
        if previous_data_for_alignment_check is not None:
            prev_short_ma = previous_data_for_alignment_check[self.short_ma_col]
            prev_medium_ma = previous_data_for_alignment_check[self.medium_ma_col]
            prev_long_ma = previous_data_for_alignment_check[self.long_ma_col]
            if pd.notna([prev_short_ma, prev_medium_ma, prev_long_ma]).all():
                 prev_bullish_alignment = (prev_short_ma > prev_medium_ma) and (prev_medium_ma > prev_long_ma)
                 prev_bearish_alignment = (prev_short_ma < prev_medium_ma) and (prev_medium_ma < prev_long_ma)


        if current_position == 0: # Logique d'entrée
            if shifted_data_antic is not None:
                short_ma_shifted_antic = shifted_data_antic[self.short_ma_col]
                medium_ma_shifted_antic = shifted_data_antic[self.medium_ma_col]

                if pd.notna([short_ma_shifted_antic, medium_ma_shifted_antic]).all():
                    # Anticipation de croisement haussier
                    if (short_ma_shifted_antic <= medium_ma_shifted_antic) and \
                       (short_ma_curr > medium_ma_curr) and \
                       (medium_ma_curr > long_ma_curr) and \
                       not prev_bullish_alignment : # Assure que ce n'est pas déjà aligné
                        side = Trading.SIDE_BUY
                        logger.info(f"{log_prefix_live} Signal ACHAT (anticipation). Entrée: {entry_price_for_order:.5f}")

                    # Anticipation de croisement baissier
                    elif (short_ma_shifted_antic >= medium_ma_shifted_antic) and \
                         (short_ma_curr < medium_ma_curr) and \
                         (medium_ma_curr < long_ma_curr) and \
                         not prev_bearish_alignment: # Assure que ce n'est pas déjà aligné
                        side = Trading.SIDE_SELL
                        logger.info(f"{log_prefix_live} Signal VENTE (anticipation). Entrée: {entry_price_for_order:.5f}")
        
        else: # Logique de sortie si une position est ouverte
            if current_position > 0 and current_bearish_alignment and not prev_bearish_alignment: # Position longue et alignement DEVIENT baissier
                side = Trading.SIDE_SELL 
                logger.info(f"{log_prefix_live} Signal SORTIE LONG (alignement baissier détecté).")
            elif current_position < 0 and current_bullish_alignment and not prev_bullish_alignment: # Position courte et alignement DEVIENT haussier
                side = Trading.SIDE_BUY 
                logger.info(f"{log_prefix_live} Signal SORTIE SHORT (alignement haussier détecté).")


        if side:
            sl_mult = self.get_param('sl_atr_mult')
            tp_mult = self.get_param('tp_atr_mult')

            if current_position != 0: # Ordre de sortie
                order_request_params = {
                    "symbol": symbol, "side": side, "type": "MARKET",
                    "quantity": "CLOSE_POSITION", 
                }
                logger.info(f"{log_prefix_live} Requête d'ordre de SORTIE générée: {order_request_params}")
                return order_request_params, {} 

            # Logique pour les ordres d'ENTRÉE (current_position == 0)
            if pd.isna(atr_curr) or atr_curr <= 1e-9:
                logger.warning(f"{log_prefix_live} ATR invalide ({atr_curr}). Pas d'ordre d'entrée.")
                return None

            if side == Trading.SIDE_BUY:
                stop_loss_price_raw = entry_price_for_order - sl_mult * atr_curr
                take_profit_price_raw = entry_price_for_order + tp_mult * atr_curr
            else: # SELL
                stop_loss_price_raw = entry_price_for_order + sl_mult * atr_curr
                take_profit_price_raw = entry_price_for_order - tp_mult * atr_curr

            price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision is None: price_precision = symbol_info.get('quoteAssetPrecision', 8)
            if qty_precision is None: qty_precision = symbol_info.get('baseAssetPrecision', 8)


            capital_to_risk = available_capital * self.get_param('position_sizing_pct_capital')
            risk_per_unit = abs(entry_price_for_order - stop_loss_price_raw)
            if risk_per_unit < 1e-9:
                logger.warning(f"{log_prefix_live} Risque par unité trop faible. Pas d'ordre.")
                return None
            quantity_raw = capital_to_risk / risk_per_unit
            
            quantity_adjusted = adjust_precision(quantity_raw, qty_precision, rounding_method=np.floor) # type: ignore
            if quantity_adjusted is None or quantity_adjusted <= 0:
                logger.warning(f"{log_prefix_live} Quantité ajustée invalide ({quantity_adjusted}). Pas d'ordre.")
                return None
            quantity_str = f"{quantity_adjusted:.{qty_precision}f}"
            
            entry_price_adjusted = adjust_precision(entry_price_for_order, price_precision, round) # type: ignore
            if entry_price_adjusted is None:
                logger.error(f"{log_prefix_live} Échec ajustement prix d'entrée. Pas d'ordre.")
                return None
            entry_price_str = f"{entry_price_adjusted:.{price_precision}f}"

            sl_price_final = adjust_precision(stop_loss_price_raw, price_precision, round) # type: ignore
            tp_price_final = adjust_precision(take_profit_price_raw, price_precision, round) # type: ignore

            if sl_price_final is None or tp_price_final is None:
                logger.error(f"{log_prefix_live} Échec ajustement SL/TP. Pas d'ordre.")
                return None

            order_request_params = {
                "symbol": symbol, "side": side, "type": "LIMIT",
                "quantity": quantity_str, "price": entry_price_str,
            }
            sl_tp_for_engine = {'sl_price': float(sl_price_final), 'tp_price': float(tp_price_final)}
            logger.info(f"{log_prefix_live} Requête d'ordre d'ENTRÉE générée: {order_request_params} avec SL/TP: {sl_tp_for_engine}")
            return order_request_params, sl_tp_for_engine
        
        logger.debug(f"{log_prefix_live} Aucune condition d'entrée/sortie remplie pour un ordre live.")
        return None

