# src/strategies/implementations/bbands_volume_rsi_strategy_impl.py
import logging # Changed from import logger to import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Attempt to import the correct BaseStrategy and other necessary modules
try:
    from ..base_strategy import BaseStrategy
    from src.utils.exchange_utils import ( # Assuming these are needed by the strategy logic eventually
        adjust_precision,
        get_filter_value,
        get_precision_from_filter
    )
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError
    # If Kline or Trading constants are used directly, import them too
    # from src.core.constants import Kline, Trading 
except ImportError as e:
    # Log the specific import error to help diagnose missing dependencies or path issues
    logging.getLogger(__name__).critical(
        "BbandsVolumeRsiStrategy: CRITICAL - Failed to import BaseStrategy or other core modules. "
        "Ensure 'src' is in PYTHONPATH and all dependencies are installed. Error: %s", e
    )
    # Fallback to a minimal dummy BaseStrategy ONLY IF ABSOLUTELY NECESSARY for the file to be parsable,
    # but this indicates a deeper setup problem that needs fixing.
    # The AttributeError for 'get_param' will persist if this dummy is used.
    if 'BaseStrategy' not in globals(): # Check if BaseStrategy was successfully imported
        from abc import ABC, abstractmethod
        class BaseStrategy(ABC): # type: ignore
            name: str = "DummyBaseStrategyForBbands"
            version: str = "0.0.0"
            description: str = "Dummy BaseStrategy (Import Failed)"
            default_params: Dict[str, Any] = {}
            required_timeframes: List[str] = []
            min_required_periods: int = 1

            def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs): 
                self.params = self.default_params.copy()
                if params: self.params.update(params)
                self.params.update(kwargs)
                # The dummy MUST have validate_params and get_param if they are called in __init__
                # or immediately after by the loader.
                # For now, assume validate_params is called by super().__init__ if it exists.
                # Add a dummy get_param to avoid the immediate AttributeError.
                
            def validate_params(self) -> None: # Dummy implementation
                pass 
            
            def get_param(self, key: str, default: Any = None) -> Any: # Dummy get_param
                return self.params.get(key, default)

            @abstractmethod
            def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]: raise NotImplementedError
            @abstractmethod
            def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame: raise NotImplementedError
            def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None
            def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None

        # Define dummy exceptions if they also failed to import
        if 'InvalidStrategyParamsError' not in globals():
            class InvalidStrategyParamsError(Exception): pass # type: ignore
        if 'StrategyError' not in globals():
            class StrategyError(Exception): pass # type: ignore
        # Define dummy utility functions if they failed to import and are used
        if 'get_filter_value' not in globals():
            def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None # type: ignore
        if 'get_precision_from_filter' not in globals():
            def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return 8 # type: ignore Default precision
        if 'adjust_precision' not in globals():
            def adjust_precision(value: float, precision: int, rounding_method = round) -> Optional[float]: return round(value, precision) if value is not None else None # type: ignore


logger = logging.getLogger(__name__) # Use standard logging

class BbandsVolumeRsiStrategy(BaseStrategy):
    name: str = "BbandsVolumeRsiStrategy"
    version: str = "1.0.3" # Version mise à jour
    description: str = "Stratégie Bollinger Bands, Volume et RSI avec ATR pour SL/TP."

    default_params: Dict[str, Any] = {
        'bbands_period': 20,
        'bbands_std_dev': 2.0,
        'indicateur_frequence_bbands': '1h', # This should match a key in klines dict passed to calculate_indicators
        'volume_ma_period': 20,
        'indicateur_frequence_volume': '1h', # This should match a key in klines dict
        'rsi_period': 14,
        'indicateur_frequence_rsi': '1h',    # This should match a key in klines dict
        'rsi_buy_breakout_threshold': 60.0, # Ensure float for consistency
        'rsi_sell_breakout_threshold': 40.0, # Ensure float for consistency
        'atr_period_sl_tp': 14,
        'atr_base_frequency_sl_tp': '1h', # This should match a key in klines dict
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
        'position_sizing_pct_capital': 0.02, # Example: 2% of capital per trade
        # Added 'indicator_frequency' as a general param for primary indicator calculations
        # This helps determine which DataFrame from the input `klines` dict to use primarily.
        'indicator_frequency': '1h', 
    }

    # The strategy might receive 1m klines and then resample them internally based on
    # 'indicateur_frequence_xxxx' parameters if calculate_indicators is designed that way.
    # Or, it might expect pre-resampled klines.
    # For now, required_timeframes indicates what it *must* receive at a minimum.
    required_timeframes: List[str] = ['1m'] # Raw data it needs to function.
    min_required_periods: int = 50 # Should be max of all lookback periods used.

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        # `pair_symbol` is expected to be passed in kwargs by StrategyLoader
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        super().__init__(params, **kwargs) # This calls self.validate_params()

        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"

        # Construct column names based on parameters
        # These names should match what `calculate_indicators` is expected to produce or find.
        bb_freq = self.get_param('indicateur_frequence_bbands')
        bb_p = self.get_param('bbands_period')
        bb_sd = self.get_param('bbands_std_dev')
        self.bb_upper_col_strat = f"BB_UPPER_{bb_freq}_p{bb_p}_sd{float(bb_sd)}"
        self.bb_middle_col_strat = f"BB_MIDDLE_{bb_freq}_p{bb_p}_sd{float(bb_sd)}"
        self.bb_lower_col_strat = f"BB_LOWER_{bb_freq}_p{bb_p}_sd{float(bb_sd)}"
        self.bb_bandwidth_col_strat = f"BB_BANDWIDTH_{bb_freq}_p{bb_p}_sd{float(bb_sd)}"

        vol_freq = self.get_param('indicateur_frequence_volume')
        # If volume indicators are on 1m, 'volume' column is used directly from 1m klines.
        # If on resampled klines (e.g. 1h), KlineProcessor would have created 'volume' for that 1h df.
        self.volume_kline_col_strat = "volume" # Assumes 'volume' column exists on the target indicator_frequency df
        self.volume_ma_col_strat = f"Volume_MA_{vol_freq}_p{self.get_param('volume_ma_period')}"

        rsi_freq = self.get_param('indicateur_frequence_rsi')
        self.rsi_col_strat = f"RSI_{rsi_freq}_p{self.get_param('rsi_period')}"

        atr_freq = self.get_param('atr_base_frequency_sl_tp')
        self.atr_col_strat = f"ATR_{atr_freq}_p{self.get_param('atr_period_sl_tp')}"

        self._signals: Optional[pd.DataFrame] = None # Cache for generated signals
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")
        # Log expected column names after they are constructed
        logger.debug(f"{self.strategy_name_log_prefix} Expected indicator columns: "
                     f"BB_UP='{self.bb_upper_col_strat}', BB_LOW='{self.bb_lower_col_strat}', "
                     f"VOL_SRC='{self.volume_kline_col_strat}', VOL_MA='{self.volume_ma_col_strat}', "
                     f"RSI='{self.rsi_col_strat}', ATR='{self.atr_col_strat}'")


    def validate_params(self) -> None:
        """Validates the strategy parameters."""
        required_params_list = [
            'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
            'volume_ma_period', 'indicateur_frequence_volume',
            'rsi_period', 'indicateur_frequence_rsi',
            'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
            'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult',
            'position_sizing_pct_capital', 'indicator_frequency'
        ]
        missing = [key for key in required_params_list if self.get_param(key) is None]
        if missing:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name=", ".join(missing), details="Missing required parameters")

        if self.get_param('bbands_period') <= 0:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='bbands_period', details="Must be > 0")
        if self.get_param('sl_atr_mult') <= 0 or self.get_param('tp_atr_mult') <= 0 :
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='sl_atr_mult/tp_atr_mult', details="Must be > 0")
        
        # Validate frequencies (example, can be more robust)
        valid_freqs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'] # Add more as needed
        freq_params_to_check = [
            'indicateur_frequence_bbands', 'indicateur_frequence_volume',
            'indicateur_frequence_rsi', 'atr_base_frequency_sl_tp', 'indicator_frequency'
        ]
        for fp_key in freq_params_to_check:
            freq_val = self.get_param(fp_key)
            if freq_val not in valid_freqs:
                 raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name=fp_key, details=f"Invalid frequency '{freq_val}'. Supported: {valid_freqs}")

        logger.debug(f"{self.strategy_name_log_prefix} Parameters validated.")

    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculates indicators. This strategy assumes that the klines dict contains
        pre-calculated indicators as columns, named according to the strategy's
        parameterized column name attributes (e.g., self.bb_upper_col_strat).
        The DataManager or a pre-processing step is responsible for populating these.
        """
        log_pref = self.strategy_name_log_prefix
        
        # The primary DataFrame for signals will be based on 'indicator_frequency'
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
        # (generate_signals might use 'close', 'open')
        ohlc_map = {'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close', 'base_asset_volume': 'volume'}
        for src_col, target_col in ohlc_map.items():
            if src_col in df_primary_indicators.columns and target_col not in df_primary_indicators.columns:
                df_primary_indicators.rename(columns={src_col: target_col}, inplace=True)
            if target_col in df_primary_indicators.columns: # Ensure numeric after potential rename
                 df_primary_indicators[target_col] = pd.to_numeric(df_primary_indicators[target_col], errors='coerce')


        # This strategy expects indicators to be ALREADY PRESENT in the DataFrame passed for `primary_indicator_freq`.
        # The names of these expected columns are defined in __init__ (e.g., self.bb_upper_col_strat).
        # If they are missing, it implies a misconfiguration or data preparation issue upstream.
        expected_indicator_cols_strat = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat,
            self.bb_bandwidth_col_strat, 
            self.volume_kline_col_strat, # This should be 'volume' on the df_primary_indicators
            self.volume_ma_col_strat, 
            self.rsi_col_strat, 
            self.atr_col_strat
        ]
        
        missing_expected_indicators = []
        for col_name in expected_indicator_cols_strat:
            if col_name not in df_primary_indicators.columns:
                # If volume_kline_col_strat is 'volume' and 'volume' is missing, it's an issue.
                if col_name == self.volume_kline_col_strat and 'volume' not in df_primary_indicators.columns:
                    missing_expected_indicators.append(f"{col_name} (expected as 'volume')")
                elif col_name != self.volume_kline_col_strat : # Don't double-report 'volume' if it's the source
                    missing_expected_indicators.append(col_name)
        
        if missing_expected_indicators:
            logger.warning(
                f"{log_pref} Certaines colonnes d'indicateurs attendues sont manquantes sur le DataFrame "
                f"'{primary_indicator_freq}': {', '.join(missing_expected_indicators)}. "
                f"La génération de signaux pourrait échouer ou produire des résultats incorrects."
            )
            # Do not add NaN columns here; the absence should be handled by generate_signals or fail.

        # The `indicators` dict returned should contain the DataFrame(s) that `generate_signals` will use.
        # In this case, it's primarily the one at `primary_indicator_freq`.
        self._indicators_cache = {primary_indicator_freq: df_primary_indicators}
        
        # If other frequencies from `klines` are needed by `generate_signals`, pass them through.
        for tf, df_tf in klines.items():
            if tf != primary_indicator_freq and tf not in self._indicators_cache:
                self._indicators_cache[tf] = df_tf.copy() # Pass through other timeframes if needed

        logger.debug(f"{log_pref} Sortie calculate_indicators. DataFrame principal ('{primary_indicator_freq}') "
                     f"colonnes: {list(df_primary_indicators.columns)}")
        return self._indicators_cache


    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        
        # Determine the primary DataFrame for signal generation
        primary_indicator_freq = self.get_param('indicator_frequency')
        if primary_indicator_freq not in indicators or indicators[primary_indicator_freq] is None:
            logger.error(f"{log_pref} DataFrame pour '{primary_indicator_freq}' non trouvé dans les indicateurs fournis.")
            # Return an empty DataFrame with expected columns for vectorbt
            empty_signals_df = pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            return empty_signals_df.astype({'entry_long': bool, 'exit_long': bool, 'entry_short': bool, 'exit_short': bool})

        df_with_indicators = indicators[primary_indicator_freq].copy()
        logger.debug(f"{log_pref} Génération des signaux (backtesting) sur les données de '{primary_indicator_freq}'. "
                     f"DataFrame shape: {df_with_indicators.shape}")

        # Ensure 'close' and 'open' are present, as they are used for signal logic and SL/TP calculation.
        # These should have been mapped from 'close_price'/'open_price' in calculate_indicators.
        required_base_cols = ['close', 'open'] 
        # Also check for indicator columns that will be directly used.
        required_indicator_cols_for_logic = [
            self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, # This is 'volume' on this df
            self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        
        all_required_cols_for_logic = required_base_cols + required_indicator_cols_for_logic
        missing_cols_in_df = [col for col in all_required_cols_for_logic if col not in df_with_indicators.columns]

        if missing_cols_in_df:
            logger.error(f"{log_pref} Colonnes essentielles manquantes pour la génération de signaux sur le DataFrame "
                         f"'{primary_indicator_freq}': {', '.join(missing_cols_in_df)}. Signaux vides seront retournés.")
            empty_signals_df = pd.DataFrame(index=df_with_indicators.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals = empty_signals_df.fillna({'entry_long': False, 'exit_long': False, 'entry_short': False, 'exit_short': False, 'sl': np.nan, 'tp': np.nan})
            return self._signals

        # Check if essential columns are entirely NaN or if DataFrame is too short
        if df_with_indicators.empty or \
           any(df_with_indicators[col].isnull().all() for col in all_required_cols_for_logic) or \
           len(df_with_indicators) < 2: # Need at least 2 rows for .shift(1)
            logger.warning(f"{log_pref} Données insuffisantes ou colonnes essentielles entièrement NaN pour la génération de signaux. "
                           f"DataFrame shape: {df_with_indicators.shape}")
            empty_signals_df = pd.DataFrame(index=df_with_indicators.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals = empty_signals_df.fillna({'entry_long': False, 'exit_long': False, 'entry_short': False, 'exit_short': False, 'sl': np.nan, 'tp': np.nan})
            return self._signals

        # --- Signal Logic ---
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')

        close_curr = df_with_indicators['close']
        close_prev = close_curr.shift(1) # Allow NaNs here, handle later
        
        bb_upper_curr = df_with_indicators[self.bb_upper_col_strat]
        bb_lower_curr = df_with_indicators[self.bb_lower_col_strat]
        bb_upper_prev = df_with_indicators[self.bb_upper_col_strat].shift(1)
        bb_lower_prev = df_with_indicators[self.bb_lower_col_strat].shift(1)

        volume_kline_curr = df_with_indicators[self.volume_kline_col_strat] # Should be 'volume'
        volume_ma_curr = df_with_indicators[self.volume_ma_col_strat]
        volume_kline_prev = df_with_indicators[self.volume_kline_col_strat].shift(1)
        volume_ma_prev = df_with_indicators[self.volume_ma_col_strat].shift(1)
        
        rsi_curr = df_with_indicators[self.rsi_col_strat]
        rsi_prev = df_with_indicators[self.rsi_col_strat].shift(1)

        # Conditions actuelles
        long_breakout_cond_curr = close_curr > bb_upper_curr
        long_volume_conf_curr = volume_kline_curr > volume_ma_curr
        long_rsi_conf_curr = rsi_curr > rsi_buy_thresh
        all_long_cond_curr = long_breakout_cond_curr & long_volume_conf_curr & long_rsi_conf_curr

        short_breakout_cond_curr = close_curr < bb_lower_curr
        short_volume_conf_curr = volume_kline_curr > volume_ma_curr
        short_rsi_conf_curr = rsi_curr < rsi_sell_thresh
        all_short_cond_curr = short_breakout_cond_curr & short_volume_conf_curr & short_rsi_conf_curr

        # Conditions précédentes (pour détecter le croisement/changement d'état)
        long_breakout_cond_prev = close_prev > bb_upper_prev
        long_volume_conf_prev = volume_kline_prev > volume_ma_prev
        long_rsi_conf_prev = rsi_prev > rsi_buy_thresh
        all_long_cond_prev = long_breakout_cond_prev & long_volume_conf_prev & long_rsi_conf_prev
        
        short_breakout_cond_prev = close_prev < bb_lower_prev
        short_volume_conf_prev = volume_kline_prev > volume_ma_prev
        short_rsi_conf_prev = rsi_prev < rsi_sell_thresh
        all_short_cond_prev = short_breakout_cond_prev & short_volume_conf_prev & short_rsi_conf_prev

        # Déclencheurs d'entrée (condition actuelle vraie ET condition précédente fausse)
        entry_long_trigger = all_long_cond_curr & ~all_long_cond_prev.fillna(False) # Handle NaNs from shift
        entry_short_trigger = all_short_cond_curr & ~all_short_cond_prev.fillna(False)

        signals_df_output = pd.DataFrame(index=df_with_indicators.index)
        signals_df_output['entry_long'] = entry_long_trigger
        signals_df_output['entry_short'] = entry_short_trigger
        
        # Les sorties sont gérées par SL/TP dans vectorbt, pas besoin de signaux d'exit explicites ici
        # sauf si la stratégie a une logique de sortie spécifique (ex: croisement de moyenne mobile opposé)
        signals_df_output['exit_long'] = False 
        signals_df_output['exit_short'] = False

        # Calcul SL/TP basé sur ATR
        atr_curr = df_with_indicators[self.atr_col_strat]
        entry_price_series = df_with_indicators['close'] # Utiliser le close comme prix d'entrée pour SL/TP

        # SL/TP pour les entrées LONG
        sl_long = np.where(entry_long_trigger, entry_price_series - sl_atr_mult * atr_curr, np.nan)
        tp_long = np.where(entry_long_trigger, entry_price_series + tp_atr_mult * atr_curr, np.nan)

        # SL/TP pour les entrées SHORT
        sl_short = np.where(entry_short_trigger, entry_price_series + sl_atr_mult * atr_curr, np.nan)
        tp_short = np.where(entry_short_trigger, entry_price_series - tp_atr_mult * atr_curr, np.nan)
        
        # Combiner SL/TP pour long et short
        signals_df_output['sl'] = np.where(entry_long_trigger, sl_long, sl_short)
        signals_df_output['tp'] = np.where(entry_long_trigger, tp_long, tp_short)

        # Assurer que les colonnes booléennes sont bien booléennes et sans NaN
        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            signals_df_output[col_b] = signals_df_output[col_b].fillna(False).astype(bool)

        self._signals = signals_df_output 
        logger.debug(f"{log_pref} Signaux générés. Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")
        if self._signals['entry_long'].sum() > 0 or self._signals['entry_short'].sum() > 0:
             logger.debug(f"Exemple de signaux SL/TP non-NaN:\n{self._signals[self._signals['sl'].notna() | self._signals['tp'].notna()][['sl', 'tp']].head()}")
        return self._signals.copy() # Retourner une copie pour éviter modifications externes

    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame], # Devrait contenir les indicateurs
                               symbol: str, # Le symbole pour lequel générer l'ordre
                               current_position: int, # 0 si pas de position, >0 si long, <0 si short
                               available_capital: float, # Capital disponible pour ce trade
                               symbol_info: dict # Informations de l'exchange pour le symbole (précision, etc.)
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre pour le trading live basée sur les dernières données.
        Cette méthode est un exemple et doit être adaptée à la logique spécifique de trading live.
        """
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        
        primary_indicator_freq = self.get_param('indicator_frequency')
        if primary_indicator_freq not in data_dict or data_dict[primary_indicator_freq] is None or data_dict[primary_indicator_freq].empty:
            logger.warning(f"{log_prefix_live} Données pour timeframe indicateur principal '{primary_indicator_freq}' manquantes ou vides.")
            return None
            
        df_indicators = data_dict[primary_indicator_freq].copy() 
        logger.info(f"{log_prefix_live} Appel generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")

        if current_position != 0: # Gérer seulement les entrées pour cet exemple
            logger.info(f"{log_prefix_live} Position existante ({current_position}). Pas d'ordre d'entrée généré par cette méthode.")
            return None # Ou implémenter la logique de sortie/gestion de position ici
        
        if len(df_indicators) < 2: # Besoin d'au moins deux points pour la logique de croisement
            logger.warning(f"{log_prefix_live} Données d'indicateurs insuffisantes (lignes: {len(df_indicators)}). Pas d'ordre.")
            return None

        # Utiliser les deux dernières barres pour la décision
        latest_data = df_indicators.iloc[-1]
        previous_data = df_indicators.iloc[-2]

        # Vérifier que les colonnes nécessaires ne sont pas NaN sur les dernières barres
        required_cols_for_live_logic = [
            'close', 'open', self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        
        if latest_data[required_cols_for_live_logic].isnull().any():
            nan_cols_latest = latest_data[required_cols_for_live_logic].index[latest_data[required_cols_for_live_logic].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur dernière donnée: {nan_cols_latest}. Pas d'ordre.")
            return None

        # Pour previous_data, ATR n'est pas crucial pour la condition de croisement elle-même
        required_cols_prev_check = [col for col in required_cols_for_live_logic if col != self.atr_col_strat]
        if previous_data[required_cols_prev_check].isnull().any():
            nan_cols_previous = previous_data[required_cols_prev_check].index[previous_data[required_cols_prev_check].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur avant-dernière donnée: {nan_cols_previous}. Pas d'ordre.")
            return None

        # Logique de signal (similaire à generate_signals mais pour une seule barre)
        close_curr = latest_data['close']
        bb_upper_curr = latest_data[self.bb_upper_col_strat]
        bb_lower_curr = latest_data[self.bb_lower_col_strat]
        volume_kline_curr = latest_data[self.volume_kline_col_strat] # 'volume'
        volume_ma_curr = latest_data[self.volume_ma_col_strat]
        rsi_curr = latest_data[self.rsi_col_strat]
        atr_value = latest_data[self.atr_col_strat]
        
        close_prev = previous_data['close']
        bb_upper_prev = previous_data[self.bb_upper_col_strat]
        bb_lower_prev = previous_data[self.bb_lower_col_strat]
        volume_kline_prev = previous_data[self.volume_kline_col_strat] # 'volume'
        volume_ma_prev = previous_data[self.volume_ma_col_strat]
        rsi_prev = previous_data[self.rsi_col_strat]
        
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')

        # Conditions actuelles
        long_breakout_cond_curr = close_curr > bb_upper_curr
        long_volume_conf_curr = volume_kline_curr > volume_ma_curr
        long_rsi_conf_curr = rsi_curr > rsi_buy_thresh
        all_long_cond_curr = long_breakout_cond_curr and long_volume_conf_curr and long_rsi_conf_curr

        short_breakout_cond_curr = close_curr < bb_lower_curr
        short_volume_conf_curr = volume_kline_curr > volume_ma_curr
        short_rsi_conf_curr = rsi_curr < rsi_sell_thresh
        all_short_cond_curr = short_breakout_cond_curr and short_volume_conf_curr and short_rsi_conf_curr

        # Conditions précédentes
        long_breakout_cond_prev = close_prev > bb_upper_prev
        long_volume_conf_prev = volume_kline_prev > volume_ma_prev
        long_rsi_conf_prev = rsi_prev > rsi_buy_thresh
        all_long_cond_prev = long_breakout_cond_prev and long_volume_conf_prev and long_rsi_conf_prev
        
        short_breakout_cond_prev = close_prev < bb_lower_prev
        short_volume_conf_prev = volume_kline_prev > volume_ma_prev
        short_rsi_conf_prev = rsi_prev < rsi_sell_thresh
        all_short_cond_prev = short_breakout_cond_prev and short_volume_conf_prev and short_rsi_conf_prev

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        entry_price_for_order = latest_data['close'] # Utiliser le close actuel comme prix d'entrée
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        if all_long_cond_curr and not all_long_cond_prev:
            side = 'BUY'
            if pd.notna(atr_value) and pd.notna(entry_price_for_order):
                stop_loss_price_raw = entry_price_for_order - sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order + tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal ACHAT. Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else: 
                logger.warning(f"{log_prefix_live} ATR ou prix d'entrée NaN pour signal BUY. ATR: {atr_value}, Entrée: {entry_price_for_order}")
                side = None # Invalider le signal
        elif all_short_cond_curr and not all_short_cond_prev:
            side = 'SELL'
            if pd.notna(atr_value) and pd.notna(entry_price_for_order):
                stop_loss_price_raw = entry_price_for_order + sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order - tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal VENTE. Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else: 
                logger.warning(f"{log_prefix_live} ATR ou prix d'entrée NaN pour signal SELL. ATR: {atr_value}, Entrée: {entry_price_for_order}")
                side = None # Invalider le signal

        if side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            if atr_value is None or atr_value <= 1e-9 or np.isnan(atr_value):
                logger.warning(f"{log_prefix_live} ATR invalide ({atr_value}). Pas d'ordre.")
                return None

            # Utiliser exchange_utils pour la précision (suppose que les fonctions sont importées)
            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            
            if price_precision_val is None or qty_precision_val is None:
                logger.error(f"{log_prefix_live} Précision prix/quantité non trouvée pour {symbol} via get_precision_from_filter. Vérifiez symbol_info et les filtres.")
                # Fallback à une précision par défaut si non trouvée, ou retourner None
                price_precision_val = symbol_info.get('quoteAssetPrecision', 8) # Exemple de fallback
                qty_precision_val = symbol_info.get('baseAssetPrecision', 8)  # Exemple de fallback
                # return None # Option plus stricte

            # _calculate_quantity devrait être une méthode de BaseStrategy ou une utilitaire
            # Pour l'instant, on simule un calcul simple.
            # REMPLACER par self._calculate_quantity si défini dans BaseStrategy
            risk_per_trade = self.get_param('position_sizing_pct_capital')
            if entry_price_for_order == 0 or abs(entry_price_for_order - stop_loss_price_raw) < 1e-9: # Eviter division par zéro
                logger.warning(f"{log_prefix_live} Prix d'entrée nul ou SL trop proche pour calculer la quantité. Pas d'ordre.")
                return None
            
            # Calcul de la quantité (exemple simplifié)
            capital_to_risk = available_capital * risk_per_trade
            risk_per_unit_base_asset = abs(entry_price_for_order - stop_loss_price_raw)
            quantity_raw = capital_to_risk / risk_per_unit_base_asset
            
            quantity_adjusted_val = adjust_precision(quantity_raw, qty_precision_val, rounding_method=np.floor) # Arrondir vers le bas pour la quantité
            if quantity_adjusted_val is None or quantity_adjusted_val <= 0:
                logger.warning(f"{log_prefix_live} Quantité calculée invalide ou nulle ({quantity_adjusted_val}) après ajustement. Pas d'ordre.")
                return None
            quantity_str = f"{quantity_adjusted_val:.{qty_precision_val}f}"
            
            entry_price_adjusted_val = adjust_precision(entry_price_for_order, price_precision_val, round)
            if entry_price_adjusted_val is None: 
                logger.error(f"{log_prefix_live} Echec ajustement prix d'entrée. Pas d'ordre.")
                return None
            entry_price_str = f"{entry_price_adjusted_val:.{price_precision_val}f}"


            # Assurer que SL/TP sont valides après ajustement et par rapport au prix d'entrée ajusté
            tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
            tick_size = float(tick_size_str) if tick_size_str and tick_size_str != "None" else (10**(-price_precision_val))

            if side == 'BUY':
                if stop_loss_price_raw >= entry_price_adjusted_val:
                    stop_loss_price_raw = entry_price_adjusted_val - tick_size 
                if take_profit_price_raw <= entry_price_adjusted_val:
                    take_profit_price_raw = entry_price_adjusted_val + tick_size
            elif side == 'SELL':
                if stop_loss_price_raw <= entry_price_adjusted_val:
                    stop_loss_price_raw = entry_price_adjusted_val + tick_size
                if take_profit_price_raw >= entry_price_adjusted_val:
                    take_profit_price_raw = entry_price_adjusted_val - tick_size
            
            # Ajuster SL et TP à la précision du prix
            sl_price_final_val = adjust_precision(stop_loss_price_raw, price_precision_val, round)
            tp_price_final_val = adjust_precision(take_profit_price_raw, price_precision_val, round)

            if sl_price_final_val is None or tp_price_final_val is None:
                 logger.error(f"{log_prefix_live} Echec ajustement SL/TP. Pas d'ordre.")
                 return None

            # _build_entry_params_formatted devrait être une méthode de BaseStrategy ou une utilitaire
            # Pour l'instant, on retourne un dict simple.
            # REMPLACER par self._build_entry_params_formatted si défini dans BaseStrategy
            order_request_params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT", # Ou MARKET, selon la logique de la stratégie
                "quantity": quantity_str,
                "price": entry_price_str, # Seulement pour LIMIT
                # "newClientOrderId": f"strat_{self.name}_{int(time.time()*1000)}", # Optionnel
            }
            
            sl_tp_for_engine = {
                'sl_price': sl_price_final_val,
                'tp_price': tp_price_final_val
            }
            logger.info(f"{log_prefix_live} Requête d'ordre générée: {order_request_params} avec SL/TP finaux: {sl_tp_for_engine}")
            return order_request_params, sl_tp_for_engine
        
        logger.debug(f"{log_prefix_live} Aucune condition d'entrée remplie pour un ordre live.")
        return None

