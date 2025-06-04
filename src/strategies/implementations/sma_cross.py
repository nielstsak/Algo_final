# src/strategies/implementations/sma_cross.py
import logging # Standard logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# Attempt to import the correct BaseStrategy and other necessary modules
try:
    from ..base_strategy import BaseStrategy # Relative import for strategy loader
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError # Correct path
    from src.core.constants import Trading, Kline # Correct path if constants are used
    # from src.utils.exchange_utils import ... # Import if specific exchange utils are needed
except ImportError as e:
    # Log the specific import error to help diagnose
    logging.getLogger(__name__).critical(
        "SMACrossStrategy: CRITICAL - Failed to import BaseStrategy or other core modules. "
        "Ensure 'src' is in PYTHONPATH and all dependencies are installed. Error: %s", e
    )
    # Define a minimal dummy BaseStrategy ONLY IF ABSOLUTELY NECESSARY for the file to be parsable.
    # This indicates a deeper setup problem that needs fixing.
    if 'BaseStrategy' not in globals(): # Check if BaseStrategy was successfully imported
        from abc import ABC, abstractmethod
        class BaseStrategy(ABC): # type: ignore
            name: str = "DummyBaseForSMACross"
            version: str = "0.0.0"
            description: str = "Dummy BaseStrategy (Import Failed)"
            default_params: Dict[str, Any] = {}
            required_timeframes: List[str] = []
            min_required_periods: int = 1

            def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs): 
                self.params = self.default_params.copy()
                if params: self.params.update(params)
                self.params.update(kwargs)
                # Ensure dummy has get_param if used in init or by loader
                # self.validate_params() # Call if BaseStrategy's __init__ does

            def validate_params(self) -> None: # Dummy implementation
                pass 
            
            def get_param(self, key: str, default: Any = None) -> Any: # Dummy get_param
                return self.params.get(key, default)

            @abstractmethod
            def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]: raise NotImplementedError
            @abstractmethod
            def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame: raise NotImplementedError
    
        # Define dummy exceptions if they also failed to import
        if 'InvalidStrategyParamsError' not in globals():
            class InvalidStrategyParamsError(Exception): pass # type: ignore
        if 'StrategyError' not in globals():
            class StrategyError(Exception): pass # type: ignore

# Use standard logging
logger = logging.getLogger(__name__)

class SMACrossStrategy(BaseStrategy):
    name: str = "SMACrossStrategy"
    version: str = "1.0.1" # Updated version
    description: str = "Stratégie de croisement de moyennes mobiles simples."

    default_params: Dict[str, Any] = {
        'fast_period': 10,
        'slow_period': 30,
        'indicator_frequency': '1h', # Frequency for calculating SMAs
        'stop_loss_pct': 0.02,  # 2% stop loss
        'take_profit_pct': 0.04, # 4% take profit
    }

    required_timeframes: List[str] = ['1m'] # Expects 1m klines to potentially resample from
    min_required_periods: int = 30 # Should be at least slow_period

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        super().__init__(params, **kwargs)
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"

        # Define expected indicator column names based on params
        sma_freq = self.get_param('indicator_frequency')
        self.fast_sma_col = f"SMA_{sma_freq}_p{self.get_param('fast_period')}"
        self.slow_sma_col = f"SMA_{sma_freq}_p{self.get_param('slow_period')}"
        
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")
        logger.debug(f"{self.strategy_name_log_prefix} Expected indicator columns: "
                     f"FastSMA='{self.fast_sma_col}', SlowSMA='{self.slow_sma_col}'")


    def validate_params(self) -> None:
        """Valide les paramètres de la stratégie."""
        fast_period = self.get_param('fast_period')
        slow_period = self.get_param('slow_period')
        sl_pct = self.get_param('stop_loss_pct')
        tp_pct = self.get_param('take_profit_pct')
        indicator_freq = self.get_param('indicator_frequency')

        if not all(isinstance(p, int) and p > 0 for p in [fast_period, slow_period]):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="fast_period/slow_period", details="Must be positive integers.")
        if fast_period >= slow_period:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="fast_period/slow_period", details="Fast period must be less than slow period.")
        if not (isinstance(sl_pct, float) and 0 < sl_pct < 1):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="stop_loss_pct", details="Must be a float between 0 and 1 (exclusive).")
        if not (isinstance(tp_pct, float) and 0 < tp_pct < 1):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="take_profit_pct", details="Must be a float between 0 and 1 (exclusive).")
        
        valid_freqs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'] # Example valid frequencies
        if indicator_freq not in valid_freqs:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='indicator_frequency', details=f"Invalid frequency '{indicator_freq}'. Supported: {valid_freqs}")

        self.min_required_periods = slow_period + 1 # Update min_required_periods based on actual slow_period
        logger.debug(f"{self.strategy_name_log_prefix} Parameters validated. min_required_periods set to {self.min_required_periods}.")


    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        This strategy expects indicators (SMAs) to be pre-calculated and present
        in the DataFrame corresponding to 'indicator_frequency'.
        The DataManager or a pre-processing step is responsible for this.
        """
        log_pref = self.strategy_name_log_prefix
        primary_indicator_freq = self.get_param('indicator_frequency')

        if primary_indicator_freq not in klines or klines[primary_indicator_freq] is None:
            raise StrategyError(
                f"{log_pref} DataFrame for the primary indicator frequency '{primary_indicator_freq}' "
                f"non fourni ou None dans le dictionnaire klines.",
                strategy_name=self.name
            )
        
        df_primary_indicators = klines[primary_indicator_freq].copy()
        logger.debug(f"{log_pref} Entrée calculate_indicators. Utilisation des données pour timeframe: '{primary_indicator_freq}'. "
                     f"Colonnes reçues: {list(df_primary_indicators.columns)}")

        # Ensure standard OHLC column names exist if they are used by the strategy logic directly
        ohlc_map = {'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'}
        for src_col, target_col in ohlc_map.items():
            if src_col in df_primary_indicators.columns and target_col not in df_primary_indicators.columns:
                df_primary_indicators.rename(columns={src_col: target_col}, inplace=True)
            if target_col in df_primary_indicators.columns: # Ensure numeric after potential rename
                 df_primary_indicators[target_col] = pd.to_numeric(df_primary_indicators[target_col], errors='coerce')
        
        # Check if expected SMA columns are present
        expected_sma_cols = [self.fast_sma_col, self.slow_sma_col]
        missing_smas = [col for col in expected_sma_cols if col not in df_primary_indicators.columns]
        if missing_smas:
            logger.warning(
                f"{log_pref} Colonnes SMA attendues manquantes sur le DataFrame '{primary_indicator_freq}': {', '.join(missing_smas)}. "
                f"La génération de signaux pourrait échouer."
            )
            # Do not add NaN columns here; let generate_signals handle missing data.

        self._indicators_cache = {primary_indicator_freq: df_primary_indicators}
        # Pass through other timeframes if they exist in klines and might be needed
        for tf, df_tf in klines.items():
            if tf != primary_indicator_freq and tf not in self._indicators_cache:
                self._indicators_cache[tf] = df_tf.copy()
        
        logger.debug(f"{log_pref} Sortie calculate_indicators. DataFrame principal ('{primary_indicator_freq}') "
                     f"colonnes: {list(df_primary_indicators.columns)}")
        return self._indicators_cache


    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Génère les signaux de trading."""
        log_pref = self.strategy_name_log_prefix
        primary_indicator_freq = self.get_param('indicator_frequency')

        if primary_indicator_freq not in indicators or indicators[primary_indicator_freq] is None:
            logger.error(f"{log_pref} DataFrame pour '{primary_indicator_freq}' non trouvé dans les indicateurs.")
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).astype(bool)

        df = indicators[primary_indicator_freq].copy()
        logger.debug(f"{log_pref} Génération des signaux sur les données de '{primary_indicator_freq}'. Shape: {df.shape}")

        # Ensure required columns for signal logic are present
        required_cols_for_logic = ['close', self.fast_sma_col, self.slow_sma_col]
        missing_cols = [col for col in required_cols_for_logic if col not in df.columns]
        if missing_cols:
            logger.error(f"{log_pref} Colonnes manquantes pour la génération de signaux: {missing_cols}. Signaux vides retournés.")
            return pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).fillna(False)

        if df.empty or any(df[col].isnull().all() for col in required_cols_for_logic) or len(df) < 2:
            logger.warning(f"{log_pref} Données insuffisantes ou colonnes essentielles NaN pour signaux. Shape: {df.shape}")
            return pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).fillna(False)

        # Signal logic
        fast_sma = df[self.fast_sma_col]
        slow_sma = df[self.slow_sma_col]

        # Entry conditions
        long_entry_cond = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        short_entry_cond = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        
        signals_df = pd.DataFrame(index=df.index)
        signals_df['entry_long'] = long_entry_cond.fillna(False)
        signals_df['entry_short'] = short_entry_cond.fillna(False)

        # Exit conditions (example: exit on reverse cross)
        signals_df['exit_long'] = short_entry_cond.fillna(False) # Exit long if short signal
        signals_df['exit_short'] = long_entry_cond.fillna(False) # Exit short if long signal

        # SL/TP calculation
        sl_pct = self.get_param('stop_loss_pct')
        tp_pct = self.get_param('take_profit_pct')
        
        entry_price_series = df['close'] # Use close price for SL/TP calculation base

        signals_df['sl'] = np.where(
            signals_df['entry_long'], entry_price_series * (1 - sl_pct),
            np.where(signals_df['entry_short'], entry_price_series * (1 + sl_pct), np.nan)
        )
        signals_df['tp'] = np.where(
            signals_df['entry_long'], entry_price_series * (1 + tp_pct),
            np.where(signals_df['entry_short'], entry_price_series * (1 - tp_pct), np.nan)
        )
        
        # Ensure boolean columns are bool and NaN-free
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
        """
        Génère une requête d'ordre pour le trading live.
        Cette méthode est un exemple et doit être adaptée à la logique spécifique de trading live.
        """
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        primary_indicator_freq = self.get_param('indicator_frequency')

        if primary_indicator_freq not in data_dict or data_dict[primary_indicator_freq] is None or data_dict[primary_indicator_freq].empty:
            logger.warning(f"{log_prefix_live} Données pour timeframe indicateur principal '{primary_indicator_freq}' manquantes ou vides.")
            return None
            
        df_indicators = data_dict[primary_indicator_freq]
        logger.info(f"{log_prefix_live} Appel generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")

        if current_position != 0: # Gérer seulement les entrées pour cet exemple
            logger.info(f"{log_prefix_live} Position existante ({current_position}). Pas d'ordre d'entrée.")
            return None
        
        if len(df_indicators) < 2: # Besoin d'au moins deux points pour la logique de croisement
            logger.warning(f"{log_prefix_live} Données d'indicateurs insuffisantes (lignes: {len(df_indicators)}). Pas d'ordre.")
            return None

        latest_indicators = df_indicators.iloc[-1]
        previous_indicators = df_indicators.iloc[-2]

        # Vérifier que les colonnes nécessaires ne sont pas NaN
        required_cols_live = ['close', self.fast_sma_col, self.slow_sma_col]
        if latest_indicators[required_cols_live].isnull().any() or \
           previous_indicators[required_cols_live].isnull().any():
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur dernières données. Pas d'ordre.")
            return None

        fast_sma_curr = latest_indicators[self.fast_sma_col]
        slow_sma_curr = latest_indicators[self.slow_sma_col]
        fast_sma_prev = previous_indicators[self.fast_sma_col]
        slow_sma_prev = previous_indicators[self.slow_sma_col]

        side: Optional[str] = None
        entry_price_for_order = latest_indicators['close'] # Utiliser le close actuel
        
        if fast_sma_curr > slow_sma_curr and fast_sma_prev <= slow_sma_prev:
            side = Trading.SIDE_BUY
            logger.info(f"{log_prefix_live} Signal ACHAT. Entrée: {entry_price_for_order:.5f}")
        elif fast_sma_curr < slow_sma_curr and fast_sma_prev >= slow_sma_prev:
            side = Trading.SIDE_SELL
            logger.info(f"{log_prefix_live} Signal VENTE. Entrée: {entry_price_for_order:.5f}")

        if side:
            sl_pct = self.get_param('stop_loss_pct')
            tp_pct = self.get_param('take_profit_pct')

            if side == Trading.SIDE_BUY:
                stop_loss_price_raw = entry_price_for_order * (1 - sl_pct)
                take_profit_price_raw = entry_price_for_order * (1 + tp_pct)
            else: # SELL
                stop_loss_price_raw = entry_price_for_order * (1 + sl_pct)
                take_profit_price_raw = entry_price_for_order * (1 - tp_pct)

            # Ici, vous devriez utiliser exchange_utils pour ajuster la précision du prix et de la quantité
            # et pour calculer la quantité en fonction du capital disponible et du risque.
            # Pour cet exemple, nous allons simplifier.
            
            # Dummy quantity calculation - REMPLACEZ PAR VOTRE LOGIQUE DE SIZING
            # Exemple: Risquer 1% du capital, SL définit le risque.
            capital_to_risk_trade = available_capital * 0.01 # Risquer 1%
            risk_per_unit = abs(entry_price_for_order - stop_loss_price_raw)
            if risk_per_unit == 0: # Eviter division par zéro
                logger.warning(f"{log_prefix_live} Risque par unité nul (prix d'entrée = SL). Pas d'ordre.")
                return None
            quantity_raw = capital_to_risk_trade / risk_per_unit
            
            # Supposons que get_precision_from_filter et adjust_precision sont disponibles et fonctionnent
            price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize') or 8
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8

            quantity_adjusted = adjust_precision(quantity_raw, qty_precision, rounding_method=np.floor) # Arrondir vers le bas
            if quantity_adjusted is None or quantity_adjusted <= 0:
                logger.warning(f"{log_prefix_live} Quantité ajustée invalide ({quantity_adjusted}). Pas d'ordre.")
                return None
            quantity_str = f"{quantity_adjusted:.{qty_precision}f}"
            
            entry_price_adjusted = adjust_precision(entry_price_for_order, price_precision, round)
            if entry_price_adjusted is None:
                logger.error(f"{log_prefix_live} Échec de l'ajustement du prix d'entrée. Pas d'ordre.")
                return None
            entry_price_str = f"{entry_price_adjusted:.{price_precision}f}"

            sl_price_final = adjust_precision(stop_loss_price_raw, price_precision, round)
            tp_price_final = adjust_precision(take_profit_price_raw, price_precision, round)

            if sl_price_final is None or tp_price_final is None:
                logger.error(f"{log_prefix_live} Échec de l'ajustement SL/TP. Pas d'ordre.")
                return None


            order_request_params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT", # Ou MARKET
                "quantity": quantity_str,
                "price": entry_price_str, # Pour LIMIT
            }
            
            sl_tp_for_engine = {
                'sl_price': sl_price_final,
                'tp_price': tp_price_final
            }
            logger.info(f"{log_prefix_live} Requête d'ordre générée: {order_request_params} avec SL/TP: {sl_tp_for_engine}")
            return order_request_params, sl_tp_for_engine
        
        logger.debug(f"{log_prefix_live} Aucune condition d'entrée remplie pour un ordre live.")
        return None

