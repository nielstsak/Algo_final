# src/strategies/implementations/triple_ma_anticipation_strategy_impl.py
import logging
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    # pandas_ta est crucial pour cette stratégie, un fallback plus complet serait nécessaire
    # si on voulait qu'elle fonctionne sans. Pour l'instant, on logue l'erreur.
    class DummyTaExtension: # Simple placeholder pour éviter AttributeError
        def __init__(self, df_instance): self.df = df_instance
        def sma(self, length, close, **kwargs): return pd.Series([np.nan]*len(self.df), index=self.df.index)
        def atr(self, length, high, low, close, **kwargs): return pd.Series([np.nan]*len(self.df), index=self.df.index)
    if not hasattr(pd.DataFrame, 'ta'):
        pd.DataFrame.ta = property(lambda df_instance: DummyTaExtension(df_instance)) # type: ignore

try:
    from ..base import BaseStrategy
except ImportError:
    try:
        from src.strategies.base import BaseStrategy
    except ImportError:
        logging.getLogger(__name__).critical("TripleMAAnticipationStrategy: CRITICAL - BaseStrategy not found.")
        from abc import ABC, abstractmethod
        class BaseStrategy(ABC): # type: ignore
            def __init__(self, params: dict, **kwargs): self.params = params
            @abstractmethod
            def validate_params(self) -> None: raise NotImplementedError
            @abstractmethod
            def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]: raise NotImplementedError
            @abstractmethod
            def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame: raise NotImplementedError
            def get_param(self, key: str, default: Any = None) -> Any: return self.params.get(key, default)
            def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None
            def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None

try:
    from src.utils.exchange_utils import (adjust_precision,
                                          get_filter_value,
                                          get_precision_from_filter)
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError, IndicatorCalculationError
except ImportError:
    logging.getLogger(__name__).error("TripleMAAnticipationStrategy: Failed to import utils or exceptions.")
    def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None
    def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return None
    def adjust_precision(value: float, precision: int, rounding_method = round) -> Optional[float]: return None
    class InvalidStrategyParamsError(Exception): pass
    class StrategyError(Exception): pass
    class IndicatorCalculationError(StrategyError): pass


logger = logging.getLogger(__name__)
if not PANDAS_TA_AVAILABLE:
    logger.error("pandas_ta non disponible. TripleMAAnticipationStrategy ne fonctionnera pas comme prévu.")


class TripleMAAnticipationStrategy(BaseStrategy):
    name: str = "TripleMAAnticipationStrategy"
    version: str = "1.0.1"
    description: str = "Stratégie Triple Moyenne Mobile avec anticipation des croisements et SL/TP par ATR."

    default_params: Dict[str, Any] = {
        'ma_short_period': 7,
        'ma_medium_period': 25,
        'ma_long_period': 99,
        'atr_period_sl_tp': 14, # Renommé pour correspondre, anciennement atr_period
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
        'indicator_frequency': '1h', # Fréquence pour calculer MAs et ATR
        'order_type_preference': "MARKET",
        'allow_shorting': False,
        'anticipate_crossovers': False,
        'anticipation_slope_period': 3,
        'anticipation_convergence_threshold_pct': 0.005, # 0.5%
        'position_sizing_pct_capital': 0.02,
    }

    required_timeframes: List[str] = ['1m'] # Données de base
    min_required_periods: int = 100 # Pour la MA la plus longue

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        # Le code original de TripleMA avait des args supplémentaires (pair_symbol, etc.)
        # BaseStrategy standard ne les prend pas. On les stocke si besoin.
        self.pair_symbol = kwargs.pop('pair_symbol', 'DEFAULT_PAIR')
        self.account_id = kwargs.pop('account_id', 'DEFAULT_ACC')
        self.deployment_id = kwargs.pop('deployment_id', 'DEFAULT_DEP')
        self.is_futures = kwargs.pop('is_futures', False)
        
        super().__init__(params, **kwargs)
        self.validate_params()
        
        self.log_prefix = f"[{self.account_id}][{self.deployment_id}][{self.pair_symbol}][{self.name}]"

        # Paramètres spécifiques
        self.ma_short_period = int(self.get_param('ma_short_period'))
        self.ma_medium_period = int(self.get_param('ma_medium_period'))
        self.ma_long_period = int(self.get_param('ma_long_period'))
        self.atr_period = int(self.get_param('atr_period_sl_tp')) # Match nom du paramètre
        self.sl_atr_mult = float(self.get_param('sl_atr_mult'))
        self.tp_atr_mult = float(self.get_param('tp_atr_mult'))
        
        self.indicator_frequency_param = self.get_param('indicator_frequency') # Sera utilisé pour resampler
        self.order_type_preference = self.get_param('order_type_preference')
        self.allow_shorting = bool(self.get_param('allow_shorting'))

        self.anticipate_crossovers = bool(self.get_param('anticipate_crossovers'))
        self.anticipation_slope_period = int(self.get_param('anticipation_slope_period'))
        if self.anticipation_slope_period < 2:
            logger.warning(f"{self.log_prefix} anticipation_slope_period ({self.anticipation_slope_period}) < 2. Mise à 2.")
            self.anticipation_slope_period = 2
        self.anticipation_convergence_threshold_pct = float(self.get_param('anticipation_convergence_threshold_pct'))

        self.active_sl_price: Optional[float] = None
        self.active_tp_price: Optional[float] = None
        self._signals: Optional[pd.DataFrame] = None

        logger.info(f"{self.log_prefix} Stratégie initialisée. MAs: {self.ma_short_period}/{self.ma_medium_period}/{self.ma_long_period}")

    def validate_params(self) -> None:
        required = ['ma_short_period', 'ma_medium_period', 'ma_long_period', 'atr_period_sl_tp', 'sl_atr_mult', 'tp_atr_mult', 'indicator_frequency']
        missing = [p for p in required if self.get_param(p) is None]
        if missing:
            raise InvalidStrategyParamsError(self.name, ", ".join(missing), "Missing parameters")
        if not (self.get_param('ma_short_period') < self.get_param('ma_medium_period') < self.get_param('ma_long_period')):
            raise InvalidStrategyParamsError(self.name, "MA periods", "Must be short < medium < long")

    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        if series.isnull().all() or len(series) < window:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        # Fonction pour calculer la pente sur une fenêtre glissante
        def get_slope_value(y_values_window):
            y_valid = y_values_window.dropna()
            if len(y_valid) < 2: # Nécessite au moins 2 points pour une pente
                return np.nan
            x_valid = np.arange(len(y_valid))
            try:
                # polyfit retourne [pente, ordonnée_origine]
                slope = np.polyfit(x_valid, y_valid, 1)[0]
                return slope
            except (np.linalg.LinAlgError, ValueError): # Gérer les erreurs de polyfit
                return np.nan
                
        return series.rolling(window=window, min_periods=max(2, window // 2)).apply(get_slope_value, raw=False)


    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        log_pref = self.log_prefix
        main_tf_raw = self.required_timeframes[0] # Ex: '1m'
        indicator_tf_target = self.indicator_frequency_param # Ex: '1h'

        if main_tf_raw not in klines or klines[main_tf_raw] is None or klines[main_tf_raw].empty:
            raise StrategyError(f"{log_pref} Données pour timeframe '{main_tf_raw}' manquantes.", strategy_name=self.name)

        df_raw = klines[main_tf_raw].copy() # Kline de base (ex: 1m)
        
        # S'assurer que les colonnes OHLCV de base sont présentes sur df_raw
        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlc:
            if col not in df_raw.columns:
                df_raw[col] = np.nan # Ajouter avec NaN si manquant
                logger.warning(f"{log_pref} Colonne OHLC de base '{col}' manquante sur les données '{main_tf_raw}'. Ajoutée avec NaN.")
            else: # Assurer le type numérique
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')


        # Resample df_raw à la fréquence des indicateurs (indicator_tf_target)
        # Si indicator_tf_target est le même que main_tf_raw, pas besoin de resample explicite pour OHLC.
        if main_tf_raw == indicator_tf_target:
            df_indicator_freq = df_raw.copy()
        else:
            try:
                # Définir les règles d'agrégation
                agg_rules = {
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }
                # S'assurer que toutes les colonnes pour agg_rules existent
                for col in agg_rules.keys():
                    if col not in df_raw.columns: df_raw[col] = np.nan
                
                df_indicator_freq = df_raw.resample(indicator_tf_target).agg(agg_rules).dropna(subset=['close'])
                if df_indicator_freq.empty:
                    raise IndicatorCalculationError(f"{log_pref} Resampling vers '{indicator_tf_target}' a produit un DataFrame vide.")
            except Exception as e:
                raise IndicatorCalculationError(f"{log_pref} Erreur lors du resampling vers '{indicator_tf_target}': {e}", original_exception=e)
        
        if not PANDAS_TA_AVAILABLE:
            logger.error(f"{log_pref} pandas_ta non disponible. Indicateurs ne seront pas calculés.")
            for col_name in ['MA_short', 'MA_medium', 'MA_long', 'ATR', 'slope_MA_short', 'slope_MA_medium']:
                df_indicator_freq[col_name] = np.nan
            self._indicators_cache = {indicator_tf_target: df_indicator_freq}
            return self._indicators_cache
            
        # Calculer les indicateurs sur df_indicator_freq
        df_indicator_freq['MA_short'] = df_indicator_freq.ta.sma(length=self.ma_short_period, close=df_indicator_freq['close'], append=False)
        df_indicator_freq['MA_medium'] = df_indicator_freq.ta.sma(length=self.ma_medium_period, close=df_indicator_freq['close'], append=False)
        df_indicator_freq['MA_long'] = df_indicator_freq.ta.sma(length=self.ma_long_period, close=df_indicator_freq['close'], append=False)
        df_indicator_freq['ATR'] = df_indicator_freq.ta.atr(length=self.atr_period, high=df_indicator_freq['high'], low=df_indicator_freq['low'], close=df_indicator_freq['close'], append=False)

        if self.anticipate_crossovers:
            df_indicator_freq['slope_MA_short'] = self._calculate_slope(df_indicator_freq['MA_short'], self.anticipation_slope_period)
            df_indicator_freq['slope_MA_medium'] = self._calculate_slope(df_indicator_freq['MA_medium'], self.anticipation_slope_period)
        else:
            df_indicator_freq['slope_MA_short'] = np.nan
            df_indicator_freq['slope_MA_medium'] = np.nan
        
        logger.debug(f"{log_pref} Indicateurs calculés sur la fréquence '{indicator_tf_target}'.")
        
        # Le DataFrame principal pour les signaux et ordres sera celui à la fréquence des indicateurs.
        # On le stocke dans le cache avec la clé de cette fréquence.
        self._indicators_cache = {indicator_tf_target: df_indicator_freq}
        return self._indicators_cache


    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        log_pref = self.log_prefix
        indicator_tf = self.indicator_frequency_param # Fréquence sur laquelle les indicateurs ont été calculés

        if indicator_tf not in indicators or indicators[indicator_tf] is None:
            logger.error(f"{log_pref} DataFrame pour la fréquence des indicateurs '{indicator_tf}' non trouvé.")
            empty_signals_df = pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp', 'ATR'])
            return empty_signals_df.astype({'entry_long': bool, 'exit_long': bool, 'entry_short': bool, 'exit_short': bool, 'ATR': float})


        df = indicators[indicator_tf].copy() # Utiliser le DF à la fréquence des indicateurs
        logger.debug(f"{log_pref} Génération des signaux (backtesting) sur les données de '{indicator_tf}'.")

        required_cols_for_signal = ['MA_short', 'MA_medium', 'MA_long', 'ATR', 'close']
        if self.anticipate_crossovers:
            required_cols_for_signal.extend(['slope_MA_short', 'slope_MA_medium'])
        
        missing_cols = [col for col in required_cols_for_signal if col not in df.columns]
        if missing_cols:
            logger.warning(f"{log_pref} Colonnes manquantes pour signaux: {missing_cols}. Signaux vides.")
            self._signals = pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp', 'ATR'])
            for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_b] = False
            for col_f in ['sl', 'tp', 'ATR']: self._signals[col_f] = np.nan
            return self._signals


        if df.empty or not all(df[col].notna().any() for col in ['MA_short', 'MA_medium', 'MA_long'] if col in df.columns):
             logger.warning(f"{log_pref} Données/indicateurs MAs insuffisants. Signaux vides.")
             self._signals = pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp', 'ATR'])
             for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_b] = False
             for col_f in ['sl', 'tp', 'ATR']: self._signals[col_f] = np.nan
             return self._signals

        # Logique de signaux (adaptée du fichier fourni)
        actual_long_entry_cross = (df['MA_short'] > df['MA_medium']) & (df['MA_short'].shift(1) <= df['MA_medium'].shift(1))
        actual_long_exit_cross = (df['MA_short'] < df['MA_medium']) & (df['MA_short'].shift(1) >= df['MA_medium'].shift(1))
        
        actual_short_entry_cross = pd.Series(False, index=df.index)
        actual_short_exit_cross = pd.Series(False, index=df.index)
        if self.allow_shorting:
            actual_short_entry_cross = (df['MA_short'] < df['MA_medium']) & (df['MA_short'].shift(1) >= df['MA_medium'].shift(1))
            actual_short_exit_cross = (df['MA_short'] > df['MA_medium']) & (df['MA_short'].shift(1) <= df['MA_medium'].shift(1))

        anticipated_long_entry = pd.Series(False, index=df.index)
        anticipated_long_exit = pd.Series(False, index=df.index)
        anticipated_short_entry = pd.Series(False, index=df.index)
        anticipated_short_exit = pd.Series(False, index=df.index)

        if self.anticipate_crossovers and \
           all(col in df.columns for col in ['slope_MA_short', 'slope_MA_medium']) and \
           df['slope_MA_short'].notna().any() and df['slope_MA_medium'].notna().any():
            
            convergence_dist = df['MA_medium'] * self.anticipation_convergence_threshold_pct
            ma_diff_abs = abs(df['MA_short'] - df['MA_medium'])
            
            is_converging_up = df['slope_MA_short'] > df['slope_MA_medium']
            is_below_and_close_long = (df['MA_short'] < df['MA_medium']) & (ma_diff_abs < convergence_dist)
            main_trend_bullish = df['MA_medium'] > df['MA_long']
            anticipated_long_entry = is_converging_up & is_below_and_close_long & main_trend_bullish

            is_converging_down_for_exit = df['slope_MA_short'] < df['slope_MA_medium']
            is_above_and_close_long_exit = (df['MA_short'] > df['MA_medium']) & (ma_diff_abs < convergence_dist)
            anticipated_long_exit = is_converging_down_for_exit & is_above_and_close_long_exit

            if self.allow_shorting:
                is_converging_down = df['slope_MA_short'] < df['slope_MA_medium']
                is_above_and_close_short = (df['MA_short'] > df['MA_medium']) & (ma_diff_abs < convergence_dist)
                main_trend_bearish = df['MA_medium'] < df['MA_long']
                anticipated_short_entry = is_converging_down & is_above_and_close_short & main_trend_bearish
                
                is_converging_up_for_exit = df['slope_MA_short'] > df['slope_MA_medium']
                is_below_and_close_short_exit = (df['MA_short'] < df['MA_medium']) & (ma_diff_abs < convergence_dist)
                anticipated_short_exit = is_converging_up_for_exit & is_below_and_close_short_exit
        
        df_signals = pd.DataFrame(index=df.index)
        df_signals['entry_long'] = actual_long_entry_cross | anticipated_long_entry
        df_signals['exit_long'] = actual_long_exit_cross | anticipated_long_exit
        df_signals['entry_short'] = actual_short_entry_cross | anticipated_short_entry
        df_signals['exit_short'] = actual_short_exit_cross | anticipated_short_exit
        
        df_signals['sl'] = np.nan
        df_signals['tp'] = np.nan
        df_signals['ATR'] = df['ATR'] # Reporter l'ATR pour information

        entry_price_series = df['close'] # Pour le calcul SL/TP

        long_entry_indices = df_signals[df_signals['entry_long']].index
        if not long_entry_indices.empty:
            valid_atr_long = df.loc[long_entry_indices, 'ATR'].notna()
            df_signals.loc[long_entry_indices[valid_atr_long], 'sl'] = entry_price_series.loc[long_entry_indices[valid_atr_long]] - (df.loc[long_entry_indices[valid_atr_long], 'ATR'] * self.sl_atr_mult)
            df_signals.loc[long_entry_indices[valid_atr_long], 'tp'] = entry_price_series.loc[long_entry_indices[valid_atr_long]] + (df.loc[long_entry_indices[valid_atr_long], 'ATR'] * self.tp_atr_mult)

        if self.allow_shorting:
            short_entry_indices = df_signals[df_signals['entry_short']].index
            if not short_entry_indices.empty:
                valid_atr_short = df.loc[short_entry_indices, 'ATR'].notna()
                df_signals.loc[short_entry_indices[valid_atr_short], 'sl'] = entry_price_series.loc[short_entry_indices[valid_atr_short]] + (df.loc[short_entry_indices[valid_atr_short], 'ATR'] * self.sl_atr_mult)
                df_signals.loc[short_entry_indices[valid_atr_short], 'tp'] = entry_price_series.loc[short_entry_indices[valid_atr_short]] - (df.loc[short_entry_indices[valid_atr_short], 'ATR'] * self.tp_atr_mult)

        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            df_signals[col_b] = df_signals[col_b].fillna(False).astype(bool)
        
        self._signals = df_signals[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp', 'ATR']].copy()
        logger.debug(f"{log_pref} Signaux générés. Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")
        return self._signals


    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame], # C'est le _indicators_cache
                               symbol: str, # pair_symbol de BaseStrategy
                               current_position: int, # -1, 0, 1
                               available_capital: float, # Capital dispo pour CETTE stratégie sur CETTE paire
                               symbol_info: dict # Infos de l'exchange sur le symbole
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]: # (entry_params, sl_tp_prices_raw)
        log_pref = self.log_prefix # Utiliser le log_prefix de l'instance
        indicator_tf = self.indicator_frequency_param

        if indicator_tf not in data_dict or data_dict[indicator_tf] is None or data_dict[indicator_tf].empty:
            logger.warning(f"{log_pref} Données pour freq indicateurs '{indicator_tf}' manquantes.")
            return None
        
        # Les indicateurs sont déjà calculés et présents dans data_dict[indicator_tf]
        # suite à l'appel de self.calculate_indicators par le moteur.
        # La méthode self.generate_signals (pour backtest) a déjà peuplé self._signals
        # sur la base de ces indicateurs.
        # Pour le live, nous utilisons la dernière ligne de self._signals (ou data_dict[indicator_tf] directement).
        
        # On utilise les signaux déjà générés et stockés dans self._signals
        # par l'appel précédent à self.generate_signals (qui est appelé par BaseStrategy.execute)
        # OU on régénère les signaux si BaseStrategy.execute n'a pas été appelé récemment.
        # Pour être sûr, on peut appeler generate_signals ici avec les dernières données des indicateurs.
        # Mais cela pourrait être redondant si le flux d'exécution s'en charge.
        # On va supposer que self._signals est à jour ou que le moteur appelle generate_signals() avant generate_order_request().
        # Pour plus de robustesse, on peut prendre la dernière ligne des indicateurs et y appliquer la logique de signal.

        df_indicators = data_dict[indicator_tf]
        if df_indicators.empty or len(df_indicators) < self.anticipation_slope_period : # Besoin d'assez de points pour la pente
            logger.warning(f"{log_pref} Pas assez de données d'indicateurs ({len(df_indicators)} lignes).")
            return None

        # Utiliser la dernière ligne des indicateurs pour prendre une décision
        latest_indicators = df_indicators.iloc[-1]
        previous_indicators = df_indicators.iloc[-2] if len(df_indicators) >=2 else latest_indicators # Fallback

        # Vérifier que les indicateurs clés ne sont pas NaN sur la dernière ligne
        check_cols = ['MA_short', 'MA_medium', 'MA_long', 'ATR', 'close']
        if self.anticipate_crossovers: check_cols.extend(['slope_MA_short', 'slope_MA_medium'])
        
        if latest_indicators[check_cols].isnull().any():
            nan_cols = latest_indicators[check_cols].index[latest_indicators[check_cols].isnull()].tolist()
            logger.warning(f"{log_pref} Indicateurs NaN sur dernière donnée: {nan_cols}. Pas d'ordre.")
            return None

        # Logique de signal (répliquée de generate_signals pour la dernière ligne)
        ma_s_curr = latest_indicators['MA_short']; ma_m_curr = latest_indicators['MA_medium']
        ma_s_prev = previous_indicators['MA_short']; ma_m_prev = previous_indicators['MA_medium']
        
        long_entry_signal = False; short_entry_signal = False
        long_exit_signal = False; short_exit_signal = False

        # Croisements actuels
        actual_long_entry_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev)
        actual_long_exit_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev)
        if self.allow_shorting:
            actual_short_entry_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev)
            actual_short_exit_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev)
        
        # Anticipation
        anticipated_long_entry = False; anticipated_long_exit = False
        anticipated_short_entry = False; anticipated_short_exit = False
        if self.anticipate_crossovers:
            slope_s = latest_indicators['slope_MA_short']; slope_m = latest_indicators['slope_MA_medium']
            ma_l_curr = latest_indicators['MA_long']
            if pd.notna(slope_s) and pd.notna(slope_m) and pd.notna(ma_l_curr):
                convergence_dist = ma_m_curr * self.anticipation_convergence_threshold_pct
                ma_diff_abs = abs(ma_s_curr - ma_m_curr)
                
                is_converging_up = slope_s > slope_m
                is_below_and_close_long = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_dist)
                main_trend_bullish = ma_m_curr > ma_l_curr
                anticipated_long_entry = is_converging_up and is_below_and_close_long and main_trend_bullish

                is_converging_down_for_exit = slope_s < slope_m
                is_above_and_close_long_exit = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_dist)
                anticipated_long_exit = is_converging_down_for_exit and is_above_and_close_long_exit
                if self.allow_shorting:
                    is_converging_down = slope_s < slope_m
                    is_above_and_close_short = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_dist)
                    main_trend_bearish = ma_m_curr < ma_l_curr
                    anticipated_short_entry = is_converging_down and is_above_and_close_short and main_trend_bearish
                    
                    is_converging_up_for_exit = slope_s > slope_m
                    is_below_and_close_short_exit = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_dist)
                    anticipated_short_exit = is_converging_up_for_exit and is_below_and_close_short_exit
        
        long_entry_signal = actual_long_entry_cross or anticipated_long_entry
        long_exit_signal = actual_long_exit_cross or anticipated_long_exit
        if self.allow_shorting:
            short_entry_signal = actual_short_entry_cross or anticipated_short_entry
            short_exit_signal = actual_short_exit_cross or anticipated_short_exit

        current_price = latest_indicators['close']
        atr_value = latest_indicators['ATR']
        
        order_side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        if current_position == 0: # Pas de position, chercher une entrée
            if pd.isna(atr_value) or atr_value <= 1e-9:
                if long_entry_signal or (self.allow_shorting and short_entry_signal):
                    logger.warning(f"{log_pref} ATR invalide ({atr_value}). Pas d'entrée.")
            elif long_entry_signal:
                order_side = "BUY"
                stop_loss_price_raw = current_price - (atr_value * self.sl_atr_mult)
                take_profit_price_raw = current_price + (atr_value * self.tp_atr_mult)
                logger.info(f"{log_pref} Signal Entrée LONG. SL brut: {stop_loss_price_raw:.4f}, TP brut: {take_profit_price_raw:.4f}")
            elif self.allow_shorting and short_entry_signal:
                order_side = "SELL"
                stop_loss_price_raw = current_price + (atr_value * self.sl_atr_mult)
                take_profit_price_raw = current_price - (atr_value * self.tp_atr_mult)
                logger.info(f"{log_pref} Signal Entrée SHORT. SL brut: {stop_loss_price_raw:.4f}, TP brut: {take_profit_price_raw:.4f}")
        
        elif current_position == 1: # Position longue ouverte
            if long_exit_signal:
                logger.info(f"{log_pref} Signal Sortie LONG.")
                # Générer un ordre de vente MARKET pour fermer la position
                # La quantité sera la taille de la position actuelle (gérée par le moteur)
                # Pas besoin de SL/TP pour un ordre de sortie.
                entry_params = self._build_entry_params_formatted(symbol=symbol, side="SELL", quantity_str="POS_SIZE", order_type="MARKET")
                return entry_params, {} # Pas de SL/TP pour une sortie explicite
            # Le moteur gérera les SL/TP actifs.

        elif current_position == -1: # Position courte ouverte
            if short_exit_signal:
                logger.info(f"{log_pref} Signal Sortie SHORT.")
                entry_params = self._build_entry_params_formatted(symbol=symbol, side="BUY", quantity_str="POS_SIZE", order_type="MARKET")
                return entry_params, {}
            # Le moteur gérera les SL/TP actifs.

        if order_side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision_val is None or qty_precision_val is None:
                logger.error(f"{log_pref} Précision prix/quantité non trouvée.")
                return None

            # La méthode _calculate_quantity est supposée exister dans BaseStrategy
            quantity = self._calculate_quantity(
                entry_price=current_price, available_capital=available_capital,
                qty_precision=qty_precision_val, symbol_info=symbol_info, symbol=symbol,
                risk_per_trade_pct=self.get_param('position_sizing_pct_capital'),
                stop_loss_price=stop_loss_price_raw
            )
            if quantity is None or quantity <= 0:
                logger.warning(f"{log_pref} Quantité invalide ({quantity}).")
                return None

            entry_price_adjusted = adjust_precision(current_price, price_precision_val, round)
            if entry_price_adjusted is None: return None
            
            entry_price_str = f"{entry_price_adjusted:.{price_precision_val}f}"
            quantity_str = f"{quantity:.{qty_precision_val}f}"
            
            # La méthode _build_entry_params_formatted est supposée exister
            entry_params = self._build_entry_params_formatted(
                symbol=symbol, side=order_side, quantity_str=quantity_str,
                entry_price_str=entry_price_str if self.order_type_preference == "LIMIT" else None, # Prix si LIMIT
                order_type=self.order_type_preference
            )
            if not entry_params: return None

            sl_tp_prices = {'sl_price': stop_loss_price_raw, 'tp_price': take_profit_price_raw}
            logger.info(f"{log_pref} Requête d'ordre: {entry_params} avec SL/TP bruts: {sl_tp_prices}")
            return entry_params, sl_tp_prices
            
        logger.debug(f"{log_pref} Aucune action d'ordre générée.")
        return None