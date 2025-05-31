# src/strategies/implementations/bbands_volume_rsi_strategy_impl.py
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    # Essayer d'importer depuis le chemin relatif .base (pour le chargement par strategy_loader)
    from ..base import BaseStrategy
except ImportError:
    # Fallback pour les tests ou exécution directe (si ce fichier n'est pas dans implementations)
    try:
        from src.strategies.base import BaseStrategy
    except ImportError:
        logging.getLogger(__name__).critical("BbandsVolumeRsiStrategy: CRITICAL - BaseStrategy not found.")
        # Définition factice minimale pour éviter les erreurs d'import globales si tout échoue
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
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError
except ImportError:
    logging.getLogger(__name__).error("BbandsVolumeRsiStrategy: Failed to import utils or exceptions. Using dummy versions.")
    def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None
    def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return None
    def adjust_precision(value: float, precision: int, rounding_method = round) -> Optional[float]: return None
    class InvalidStrategyParamsError(Exception): pass
    class StrategyError(Exception): pass

logger = logging.getLogger(__name__)

class BbandsVolumeRsiStrategy(BaseStrategy):
    name: str = "BbandsVolumeRsiStrategy"
    version: str = "1.0.2" # Version mise à jour
    description: str = "Stratégie Bollinger Bands, Volume et RSI avec ATR pour SL/TP."

    default_params: Dict[str, Any] = {
        'bbands_period': 20,
        'bbands_std_dev': 2.0,
        'indicateur_frequence_bbands': '1h',
        'volume_ma_period': 20,
        'indicateur_frequence_volume': '1h',
        'rsi_period': 14,
        'indicateur_frequence_rsi': '1h',
        'rsi_buy_breakout_threshold': 60,
        'rsi_sell_breakout_threshold': 40,
        'atr_period_sl_tp': 14,
        'atr_base_frequency_sl_tp': '1h',
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
        'position_sizing_pct_capital': 0.02,
    }

    required_timeframes: List[str] = ['1m'] # Basé sur klines 1m, indicateurs sur fréquences >
    min_required_periods: int = 50 # Ajuster selon max(bbands_period, volume_ma_period, rsi_period, atr_period_sl_tp)

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(params, **kwargs)
        self.validate_params() # Valider après initialisation des paramètres

        self.strategy_name_log_prefix = f"[{self.name}]"

        # Définition dynamique des noms de colonnes d'indicateurs attendus
        bb_freq = self.get_param('indicateur_frequence_bbands')
        bb_p = self.get_param('bbands_period')
        bb_sd = self.get_param('bbands_std_dev')
        self.bb_upper_col_strat = f"BB_UPPER_{bb_freq}_p{bb_p}_sd{bb_sd}"
        self.bb_middle_col_strat = f"BB_MIDDLE_{bb_freq}_p{bb_p}_sd{bb_sd}"
        self.bb_lower_col_strat = f"BB_LOWER_{bb_freq}_p{bb_p}_sd{bb_sd}"
        self.bb_bandwidth_col_strat = f"BB_BANDWIDTH_{bb_freq}_p{bb_p}_sd{bb_sd}"

        vol_freq = self.get_param('indicateur_frequence_volume')
        # La colonne source du volume pour l'indicateur de volume MA
        # Si la fréquence de l'indicateur de volume est '1m', on utilise la colonne 'volume' brute.
        # Sinon, on attend une colonne de volume agrégée, ex: 'Kline_1h_volume'
        if vol_freq.lower() in ['1m', '1min', '1t']:
            self.volume_kline_col_strat = "volume" # Colonne OHLCV de base
        else:
            self.volume_kline_col_strat = f"Kline_{vol_freq}_volume" # Colonne agrégée
        self.volume_ma_col_strat = f"Volume_MA_{vol_freq}_p{self.get_param('volume_ma_period')}"

        rsi_freq = self.get_param('indicateur_frequence_rsi')
        self.rsi_col_strat = f"RSI_{rsi_freq}_p{self.get_param('rsi_period')}"

        atr_freq = self.get_param('atr_base_frequency_sl_tp')
        self.atr_col_strat = f"ATR_{atr_freq}_p{self.get_param('atr_period_sl_tp')}"

        self._signals: Optional[pd.DataFrame] = None
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée. Colonnes indicateurs attendues: "
                    f"BB_UP='{self.bb_upper_col_strat}', BB_LOW='{self.bb_lower_col_strat}', "
                    f"VOL_SRC='{self.volume_kline_col_strat}', VOL_MA='{self.volume_ma_col_strat}', "
                    f"RSI='{self.rsi_col_strat}', ATR='{self.atr_col_strat}'")

    def validate_params(self) -> None:
        required_params_list = [
            'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
            'volume_ma_period', 'indicateur_frequence_volume',
            'rsi_period', 'indicateur_frequence_rsi',
            'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
            'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult'
        ]
        missing = [key for key in required_params_list if self.get_param(key) is None]
        if missing:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name=", ".join(missing), details="Missing required parameters")

        # Ajoutez ici des validations plus spécifiques pour chaque paramètre si nécessaire
        if self.get_param('bbands_period') <= 0:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='bbands_period', details="Must be > 0")
        if self.get_param('sl_atr_mult') <= 0 or self.get_param('tp_atr_mult') <= 0 :
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='sl_atr_mult/tp_atr_mult', details="Must be > 0")

    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        log_pref = self.strategy_name_log_prefix
        main_tf = self.required_timeframes[0] # Ex: '1m'

        if main_tf not in klines or klines[main_tf] is None:
            raise StrategyError(f"{log_pref} DataFrame pour le timeframe principal '{main_tf}' non fourni ou None.", strategy_name=self.name)
        
        df = klines[main_tf].copy()
        logger.debug(f"{log_pref} Entrée calculate_indicators. Colonnes reçues pour '{main_tf}': {list(df.columns)}")

        # Vérifier et ajouter les colonnes OHLCV de base si manquantes
        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlc:
            if col not in df.columns:
                df[col] = np.nan
                logger.warning(f"{log_pref} Colonne OHLC de base '{col}' manquante pour '{main_tf}'. Ajoutée avec NaN.")

        # Vérifier et ajouter les colonnes d'indicateurs spécifiques à la stratégie si manquantes
        expected_indicator_cols_strat = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat,
            self.bb_bandwidth_col_strat, self.volume_kline_col_strat,
            self.volume_ma_col_strat, self.rsi_col_strat, self.atr_col_strat
        ]
        for col_name in expected_indicator_cols_strat:
            if col_name not in df.columns:
                df[col_name] = np.nan
                logger.warning(f"{log_pref} Colonne indicateur stratégie '{col_name}' manquante pour '{main_tf}'. Ajoutée avec NaN.")

        # Cette méthode doit retourner un dictionnaire de DataFrames.
        # Pour cette stratégie, tous les indicateurs sont supposés être sur le DF principal.
        self._indicators_cache = {main_tf: df} # Mise à jour du cache interne utilisé par BaseStrategy
        logger.debug(f"{log_pref} Sortie calculate_indicators. Colonnes présentes pour '{main_tf}': {list(df.columns)}")
        return self._indicators_cache

    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        main_tf = self.required_timeframes[0]

        if main_tf not in indicators or indicators[main_tf] is None:
            logger.error(f"{log_pref} DataFrame pour '{main_tf}' non trouvé dans les indicateurs.")
            empty_signals_df = pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            return empty_signals_df.astype({'entry_long': bool, 'exit_long': bool, 'entry_short': bool, 'exit_short': bool})


        df_with_indicators = indicators[main_tf].copy()
        logger.debug(f"{log_pref} Génération des signaux (backtesting) sur les données de '{main_tf}'.")

        required_cols_for_signal = [
            self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat, 'close', 'open'
        ]
        
        # Vérification si toutes les colonnes nécessaires sont présentes
        missing_cols = [col for col in required_cols_for_signal if col not in df_with_indicators.columns]
        if missing_cols:
            logger.warning(f"{log_pref} Colonnes manquantes pour la génération de signaux: {missing_cols}. Signaux vides seront retournés.")
            self._signals = pd.DataFrame(index=df_with_indicators.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=bool).columns] = False # Mieux que select_dtypes(include=bool)
            for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_b] = False
            return self._signals


        # Vérification des données insuffisantes ou NaN
        if df_with_indicators.empty or \
           any(df_with_indicators[col].isnull().all() for col in required_cols_for_signal) or \
           len(df_with_indicators) < 2:
            logger.warning(f"{log_pref} Données insuffisantes ou colonnes essentielles entièrement NaN pour la génération de signaux (backtesting).")
            self._signals = pd.DataFrame(index=df_with_indicators.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_b] = False
            return self._signals

        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')

        close_curr = df_with_indicators['close']
        close_prev = close_curr.shift(1)
        bb_upper_curr = df_with_indicators[self.bb_upper_col_strat]
        bb_lower_curr = df_with_indicators[self.bb_lower_col_strat]
        volume_kline_curr = df_with_indicators[self.volume_kline_col_strat]
        volume_ma_curr = df_with_indicators[self.volume_ma_col_strat]
        rsi_curr = df_with_indicators[self.rsi_col_strat]

        long_breakout_cond_curr = close_curr > bb_upper_curr
        long_volume_conf_curr = volume_kline_curr > volume_ma_curr
        long_rsi_conf_curr = rsi_curr > rsi_buy_thresh
        all_long_cond_curr = long_breakout_cond_curr & long_volume_conf_curr & long_rsi_conf_curr

        long_breakout_cond_prev = close_prev > bb_upper_curr.shift(1)
        long_volume_conf_prev = volume_kline_curr.shift(1) > volume_ma_curr.shift(1)
        long_rsi_conf_prev = rsi_curr.shift(1) > rsi_buy_thresh
        all_long_cond_prev = (long_breakout_cond_prev & long_volume_conf_prev & long_rsi_conf_prev).fillna(False)
        entry_long_trigger = all_long_cond_curr & ~all_long_cond_prev

        short_breakout_cond_curr = close_curr < bb_lower_curr
        short_volume_conf_curr = volume_kline_curr > volume_ma_curr
        short_rsi_conf_curr = rsi_curr < rsi_sell_thresh
        all_short_cond_curr = short_breakout_cond_curr & short_volume_conf_curr & short_rsi_conf_curr

        short_breakout_cond_prev = close_prev < bb_lower_curr.shift(1)
        short_volume_conf_prev = volume_kline_curr.shift(1) > volume_ma_curr.shift(1)
        short_rsi_conf_prev = rsi_curr.shift(1) < rsi_sell_thresh
        all_short_cond_prev = (short_breakout_cond_prev & short_volume_conf_prev & short_rsi_conf_prev).fillna(False)
        entry_short_trigger = all_short_cond_curr & ~all_short_cond_prev

        signals_df_output = pd.DataFrame(index=df_with_indicators.index)
        signals_df_output['entry_long'] = entry_long_trigger
        signals_df_output['entry_short'] = entry_short_trigger
        signals_df_output['exit_long'] = False
        signals_df_output['exit_short'] = False
        
        atr_curr = df_with_indicators[self.atr_col_strat]
        entry_price_series = df_with_indicators['close'] # Utiliser la clôture comme prix d'entrée pour SL/TP

        signals_df_output['sl'] = np.where(
            entry_long_trigger & atr_curr.notna(), entry_price_series - sl_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), entry_price_series + sl_atr_mult * atr_curr, np.nan)
        )
        signals_df_output['tp'] = np.where(
            entry_long_trigger & atr_curr.notna(), entry_price_series + tp_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), entry_price_series - tp_atr_mult * atr_curr, np.nan)
        )

        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            signals_df_output[col_b] = signals_df_output[col_b].fillna(False).astype(bool)

        self._signals = signals_df_output # Stockage interne pour get_signals()
        logger.debug(f"{log_pref} Signaux générés (backtesting). Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")
        return self._signals

    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame],
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        main_tf = self.required_timeframes[0]

        if main_tf not in data_dict or data_dict[main_tf] is None or data_dict[main_tf].empty:
            logger.warning(f"{log_prefix_live} Données pour timeframe principal '{main_tf}' manquantes ou vides.")
            return None
            
        data = data_dict[main_tf].copy() # Utiliser le DataFrame du timeframe principal
        logger.info(f"{log_prefix_live} Appel generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")

        if current_position != 0:
            logger.info(f"{log_prefix_live} Position existante ({current_position}). Pas d'ordre d'entrée.")
            return None
        
        if len(data) < 2:
            logger.warning(f"{log_prefix_live} Données d'entrée insuffisantes (lignes: {len(data)}). Pas d'ordre.")
            return None

        # Assurer la présence des colonnes d'indicateurs (même si NaN)
        df_verified_indicators = data # Supposer que les indicateurs sont déjà dans `data`
        expected_cols_for_live = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat, 
            self.bb_bandwidth_col_strat, self.volume_kline_col_strat, 
            self.volume_ma_col_strat, self.rsi_col_strat, self.atr_col_strat,
            'open', 'high', 'low', 'close'
        ]
        for col_name in expected_cols_for_live:
            if col_name not in df_verified_indicators.columns:
                df_verified_indicators[col_name] = np.nan
                logger.warning(f"{log_prefix_live} Colonne '{col_name}' manquante, ajoutée avec NaN.")
             
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols_check = [
            'close', 'open', self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        
        log_msg_indicators = f"{log_prefix_live} Indicateurs (dernière ligne): "
        for col in required_cols_check:
            val = latest_data.get(col)
            log_msg_indicators += f"{col}=" + (f"{val:.5f}" if isinstance(val, (float, np.floating)) and pd.notna(val) else str(val)) + "; "
        logger.info(log_msg_indicators)

        if latest_data[required_cols_check].isnull().any():
            nan_cols_latest = latest_data[required_cols_check].index[latest_data[required_cols_check].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur dernière donnée: {nan_cols_latest}. Pas d'ordre.")
            return None

        required_cols_previous_check = [col for col in required_cols_check if col != self.atr_col_strat]
        if previous_data[required_cols_previous_check].isnull().any():
            nan_cols_previous = previous_data[required_cols_previous_check].index[previous_data[required_cols_previous_check].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur avant-dernière donnée: {nan_cols_previous}. Pas d'ordre.")
            return None

        close_curr = latest_data['close']
        bb_upper_curr = latest_data[self.bb_upper_col_strat]
        bb_lower_curr = latest_data[self.bb_lower_col_strat]
        volume_kline_curr = latest_data[self.volume_kline_col_strat]
        volume_ma_curr = latest_data[self.volume_ma_col_strat]
        rsi_curr = latest_data[self.rsi_col_strat]
        atr_value = latest_data[self.atr_col_strat]
        
        close_prev = previous_data['close']
        bb_upper_prev = previous_data[self.bb_upper_col_strat]
        bb_lower_prev = previous_data[self.bb_lower_col_strat]
        volume_kline_prev = previous_data[self.volume_kline_col_strat]
        volume_ma_prev = previous_data[self.volume_ma_col_strat]
        rsi_prev = previous_data[self.rsi_col_strat]
        
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')

        long_breakout_cond_curr = close_curr > bb_upper_curr
        long_volume_conf_curr = volume_kline_curr > volume_ma_curr
        long_rsi_conf_curr = rsi_curr > rsi_buy_thresh
        all_long_cond_curr = long_breakout_cond_curr and long_volume_conf_curr and long_rsi_conf_curr

        long_breakout_cond_prev = close_prev > bb_upper_prev
        long_volume_conf_prev = volume_kline_prev > volume_ma_prev
        long_rsi_conf_prev = rsi_prev > rsi_buy_thresh
        all_long_cond_prev = long_breakout_cond_prev and long_volume_conf_prev and long_rsi_conf_prev
        
        short_breakout_cond_curr = close_curr < bb_lower_curr
        short_volume_conf_curr = volume_kline_curr > volume_ma_curr
        short_rsi_conf_curr = rsi_curr < rsi_sell_thresh
        all_short_cond_curr = short_breakout_cond_curr and short_volume_conf_curr and short_rsi_conf_curr

        short_breakout_cond_prev = close_prev < bb_lower_prev
        short_volume_conf_prev = volume_kline_prev > volume_ma_prev
        short_rsi_conf_prev = rsi_prev < rsi_sell_thresh
        all_short_cond_prev = short_breakout_cond_prev and short_volume_conf_prev and short_rsi_conf_prev

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        entry_price_for_order = latest_data['close'] # Utiliser la clôture actuelle pour le prix d'entrée SL/TP
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        if all_long_cond_curr and not all_long_cond_prev:
            side = 'BUY'
            if pd.notna(atr_value) and pd.notna(entry_price_for_order):
                stop_loss_price_raw = entry_price_for_order - sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order + tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal ACHAT. Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else: side = None
        elif all_short_cond_curr and not all_short_cond_prev:
            side = 'SELL'
            if pd.notna(atr_value) and pd.notna(entry_price_for_order):
                stop_loss_price_raw = entry_price_for_order + sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order - tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal VENTE. Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else: side = None

        if side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            if atr_value is None or atr_value <= 1e-9 or np.isnan(atr_value):
                logger.warning(f"{log_prefix_live} ATR invalide ({atr_value}). Pas d'ordre.")
                return None

            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision_val is None or qty_precision_val is None:
                logger.error(f"{log_prefix_live} Précision prix/quantité non trouvée pour {symbol}.")
                return None

            quantity = self._calculate_quantity(
                entry_price=entry_price_for_order, available_capital=available_capital,
                qty_precision=qty_precision_val, symbol_info=symbol_info, symbol=symbol,
                risk_per_trade_pct=self.get_param('position_sizing_pct_capital'),
                stop_loss_price=stop_loss_price_raw
            )
            if quantity is None or quantity <= 0:
                logger.warning(f"{log_prefix_live} Quantité calculée invalide ({quantity}). Pas d'ordre.")
                return None

            entry_price_adjusted = adjust_precision(entry_price_for_order, price_precision_val, round)
            if entry_price_adjusted is None:
                logger.error(f"{log_prefix_live} Echec ajustement prix d'entrée.")
                return None
            
            tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
            tick_size = float(tick_size_str) if tick_size_str and tick_size_str != "None" else 10**(-price_precision_val)
            
            if side == 'BUY':
                if stop_loss_price_raw >= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} SL ({stop_loss_price_raw:.5f}) >= Prix d'entrée ({entry_price_adjusted:.5f}) pour un BUY. Ajustement du SL.")
                    stop_loss_price_raw = entry_price_adjusted - tick_size 
                if take_profit_price_raw <= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} TP ({take_profit_price_raw:.5f}) <= Prix d'entrée ({entry_price_adjusted:.5f}) pour un BUY. Ajustement du TP.")
                    take_profit_price_raw = entry_price_adjusted + tick_size
            elif side == 'SELL':
                if stop_loss_price_raw <= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} SL ({stop_loss_price_raw:.5f}) <= Prix d'entrée ({entry_price_adjusted:.5f}) pour un SELL. Ajustement du SL.")
                    stop_loss_price_raw = entry_price_adjusted + tick_size
                if take_profit_price_raw >= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} TP ({take_profit_price_raw:.5f}) >= Prix d'entrée ({entry_price_adjusted:.5f}) pour un SELL. Ajustement du TP.")
                    take_profit_price_raw = entry_price_adjusted - tick_size

            entry_price_str = f"{entry_price_adjusted:.{price_precision_val}f}"
            quantity_str = f"{quantity:.{qty_precision_val}f}"

            entry_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str, order_type="LIMIT"
            )
            if not entry_params:
                logger.error(f"{log_prefix_live} Echec construction params d'entrée.")
                return None

            sl_tp_prices = {'sl_price': stop_loss_price_raw, 'tp_price': take_profit_price_raw}
            logger.info(f"{log_prefix_live} Requête d'ordre: {entry_params} avec SL/TP bruts: {sl_tp_prices}")
            return entry_params, sl_tp_prices
        
        logger.info(f"{log_prefix_live} Aucune condition d'entrée remplie.")
        return None