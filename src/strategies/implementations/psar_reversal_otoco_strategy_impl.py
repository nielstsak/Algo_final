# src/strategies/implementations/psar_reversal_otoco_strategy_impl.py
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
# pandas_ta n'est pas explicitement utilisé ici car PSAR et ATR sont attendus précalculés.

try:
    from ..base import BaseStrategy
except ImportError:
    try:
        from src.strategies.base import BaseStrategy
    except ImportError:
        logging.getLogger(__name__).critical("PsarReversalOtocoStrategy: CRITICAL - BaseStrategy not found.")
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
    logging.getLogger(__name__).error("PsarReversalOtocoStrategy: Failed to import utils or exceptions. Using dummy versions.")
    def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None
    def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return None
    def adjust_precision(value: float, precision: int, rounding_method = round) -> Optional[float]: return None
    class InvalidStrategyParamsError(Exception): pass
    class StrategyError(Exception): pass

logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    name: str = "PsarReversalOtocoStrategy"
    version: str = "1.0.1"
    description: str = "Stratégie de renversement basée sur le Parabolic SAR, avec SL/TP basés sur l'ATR."

    default_params: Dict[str, Any] = {
        'psar_step': 0.02,
        'psar_max_step': 0.2,
        'indicateur_frequence_psar': '1h',
        'atr_period': 14,
        'atr_base_frequency': '1h',
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
        'position_sizing_pct_capital': 0.02,
    }

    required_timeframes: List[str] = ['1m'] # Données de base
    min_required_periods: int = 20 # Min pour ATR, PSAR pourrait en nécessiter plus

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(params, **kwargs)
        self.validate_params()
        self.strategy_name_log_prefix = f"[{self.name}]"

        psar_freq = self.get_param('indicateur_frequence_psar')
        psar_s = self.get_param('psar_step')
        psar_ms = self.get_param('psar_max_step')
        self.psarl_col_strat = f"PSARl_{psar_freq}_s{psar_s}_ms{psar_ms}"
        self.psars_col_strat = f"PSARs_{psar_freq}_s{psar_s}_ms{psar_ms}"
        
        atr_freq = self.get_param('atr_base_frequency')
        atr_p = self.get_param('atr_period')
        self.atr_col_strat = f"ATR_{atr_freq}_p{atr_p}"
        
        self._signals: Optional[pd.DataFrame] = None
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée. Colonnes indicateurs attendues: "
                    f"PSARL='{self.psarl_col_strat}', PSARS='{self.psars_col_strat}', ATR='{self.atr_col_strat}'")

    def validate_params(self) -> None:
        required_params_list = ['psar_step', 'psar_max_step', 'indicateur_frequence_psar',
                                'atr_period', 'atr_base_frequency', 'sl_atr_mult', 'tp_atr_mult']
        missing = [key for key in required_params_list if self.get_param(key) is None]
        if missing:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name=", ".join(missing), details="Missing required parameters")
        if self.get_param('psar_step') <= 0 or self.get_param('psar_max_step') <= self.get_param('psar_step'):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='psar_step/psar_max_step', details="Invalid PSAR step values")

    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        log_pref = self.strategy_name_log_prefix
        main_tf = self.required_timeframes[0]

        if main_tf not in klines or klines[main_tf] is None:
            raise StrategyError(f"{log_pref} DataFrame pour '{main_tf}' non fourni.", strategy_name=self.name)
        
        df = klines[main_tf].copy()
        logger.debug(f"{log_pref} Entrée calculate_indicators. Colonnes pour '{main_tf}': {list(df.columns)}")

        required_ohlc = ['open', 'high', 'low', 'close'] # Volume non requis pour PSAR/ATR directement
        for col in required_ohlc:
            if col not in df.columns:
                df[col] = np.nan
                logger.warning(f"{log_pref} Colonne OHLC '{col}' manquante pour '{main_tf}'. Ajoutée avec NaN.")

        expected_indicator_cols_strat = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        for col_name in expected_indicator_cols_strat:
            if col_name not in df.columns:
                df[col_name] = np.nan
                logger.warning(f"{log_pref} Colonne indicateur '{col_name}' manquante pour '{main_tf}'. Ajoutée avec NaN.")
        
        self._indicators_cache = {main_tf: df}
        logger.debug(f"{log_pref} Sortie calculate_indicators. Colonnes pour '{main_tf}': {list(df.columns)}")
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
        
        required_cols_for_signal = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat, 'close']
        missing_cols = [col for col in required_cols_for_signal if col not in df_with_indicators.columns]
        if missing_cols:
            logger.warning(f"{log_pref} Colonnes manquantes pour la génération de signaux: {missing_cols}. Signaux vides.")
            self._signals = pd.DataFrame(index=df_with_indicators.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_b] = False
            return self._signals

        if df_with_indicators.empty or \
           any(df_with_indicators[col].isnull().all() for col in required_cols_for_signal) or \
           len(df_with_indicators) < 2:
            logger.warning(f"{log_pref} Données insuffisantes ou colonnes essentielles NaN pour signaux (backtesting).")
            self._signals = pd.DataFrame(index=df_with_indicators.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_b] = False
            return self._signals

        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        close_prev = df_with_indicators['close'].shift(1)
        close_curr = df_with_indicators['close']
        
        # Remplir NaN dans PSAR avec une valeur qui n'activera pas de signal
        # Cela peut arriver au début des données ou si PSAR n'est pas calculable
        psarl_prev_filled = df_with_indicators[self.psarl_col_strat].shift(1).fillna(close_prev + 1e9) # Très haut pour ne pas être < close_prev
        psars_prev_filled = df_with_indicators[self.psars_col_strat].shift(1).fillna(close_prev - 1e9) # Très bas pour ne pas être > close_prev
        
        psarl_curr = df_with_indicators[self.psarl_col_strat] 
        psars_curr = df_with_indicators[self.psars_col_strat] 
        atr_curr = df_with_indicators[self.atr_col_strat]

        long_condition = (close_prev <= psars_prev_filled) & (close_curr > psarl_curr) & psarl_curr.notna()
        short_condition = (close_prev >= psarl_prev_filled) & (close_curr < psars_curr) & psars_curr.notna()
        
        signals_df_output = pd.DataFrame(index=df_with_indicators.index)
        signals_df_output['entry_long'] = long_condition
        signals_df_output['entry_short'] = short_condition
        signals_df_output['exit_long'] = short_condition # Sortie de long sur signal de short
        signals_df_output['exit_short'] = long_condition  # Sortie de short sur signal de long

        entry_price_series = close_curr # Utiliser la clôture pour SL/TP

        signals_df_output['sl'] = np.where(
            long_condition & atr_curr.notna(), entry_price_series - sl_atr_mult * atr_curr, 
            np.where(short_condition & atr_curr.notna(), entry_price_series + sl_atr_mult * atr_curr, np.nan)
        )
        signals_df_output['tp'] = np.where(
            long_condition & atr_curr.notna(), entry_price_series + tp_atr_mult * atr_curr, 
            np.where(short_condition & atr_curr.notna(), entry_price_series - tp_atr_mult * atr_curr, np.nan)
        )
        
        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            signals_df_output[col_b] = signals_df_output[col_b].fillna(False).astype(bool)

        self._signals = signals_df_output
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
            logger.warning(f"{log_prefix_live} Données pour '{main_tf}' manquantes ou vides.")
            return None
            
        data = data_dict[main_tf].copy()
        logger.info(f"{log_prefix_live} Appel generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")

        if current_position != 0:
            logger.info(f"{log_prefix_live} Position existante ({current_position}). Pas d'ordre d'entrée.")
            return None
            
        if len(data) < 2:
            logger.warning(f"{log_prefix_live} Données insuffisantes (lignes: {len(data)}). Pas d'ordre.")
            return None

        df_verified_indicators = data # Supposer que les indicateurs _strat sont déjà dans data
        expected_cols_for_live = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat, 'open', 'high', 'low', 'close']
        for col_name in expected_cols_for_live:
            if col_name not in df_verified_indicators.columns:
                df_verified_indicators[col_name] = np.nan
                logger.warning(f"{log_prefix_live} Colonne '{col_name}' manquante, ajoutée avec NaN.")
             
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols = ['close', self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        if latest_data[required_cols].isnull().any() or \
           previous_data[['close', self.psarl_col_strat, self.psars_col_strat]].isnull().any():
            nan_cols_latest = latest_data[required_cols].index[latest_data[required_cols].isnull()].tolist()
            nan_cols_prev = previous_data[['close', self.psarl_col_strat, self.psars_col_strat]].index[previous_data[['close', self.psarl_col_strat, self.psars_col_strat]].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs NaN. Dernière: {nan_cols_latest}, Précédente: {nan_cols_prev}. Pas d'ordre.")
            return None

        close_curr = latest_data['close']
        psarl_curr = latest_data[self.psarl_col_strat]
        psars_curr = latest_data[self.psars_col_strat]
        close_prev = previous_data['close']
        # Utiliser les valeurs _strat pour psar_prev
        psarl_prev = previous_data[self.psarl_col_strat]
        psars_prev = previous_data[self.psars_col_strat]
        atr_value = latest_data[self.atr_col_strat]
        
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        entry_price_for_order = close_curr # Utiliser la clôture actuelle pour SL/TP et comme cible d'entrée

        # Logique de renversement: si PSAR passe de sous les prix (haussier) à au-dessus (baissier) ou vice-versa.
        # Condition d'achat: SAR précédent était au-dessus des prix (baissier), SAR actuel est sous les prix (haussier)
        #   ET le prix actuel a croisé au-dessus du SAR actuel.
        #   psars_prev est la valeur du SAR baissier à t-1. Si close_prev <= psars_prev, alors le SAR était au-dessus ou égal au prix.
        #   psarl_curr est la valeur du SAR haussier à t. Si close_curr > psarl_curr, alors le prix a croisé au-dessus.
        if pd.notna(psars_prev) and pd.notna(psarl_curr) and \
           close_prev <= psars_prev and close_curr > psarl_curr:
            side = 'BUY'
            if pd.notna(atr_value):
                stop_loss_price_raw = entry_price_for_order - sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order + tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal ACHAT (PSAR Reversal). Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else: side = None # ATR invalide

        # Condition de vente: SAR précédent était sous les prix (haussier), SAR actuel est au-dessus (baissier)
        #   ET le prix actuel a croisé sous le SAR actuel.
        #   psarl_prev est la valeur du SAR haussier à t-1. Si close_prev >= psarl_prev, alors le SAR était en-dessous ou égal au prix.
        #   psars_curr est la valeur du SAR baissier à t. Si close_curr < psars_curr, alors le prix a croisé en-dessous.
        elif pd.notna(psarl_prev) and pd.notna(psars_curr) and \
             close_prev >= psarl_prev and close_curr < psars_curr:
            side = 'SELL'
            if pd.notna(atr_value):
                stop_loss_price_raw = entry_price_for_order + sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order - tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal VENTE (PSAR Reversal). Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else: side = None # ATR invalide

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
            
            entry_price_str = f"{entry_price_adjusted:.{price_precision_val}f}"
            quantity_str = f"{quantity:.{qty_precision_val}f}"

            entry_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str, order_type="LIMIT" # Ou MARKET
            )
            if not entry_params:
                logger.error(f"{log_prefix_live} Echec construction params d'entrée.")
                return None

            sl_tp_prices = {'sl_price': stop_loss_price_raw, 'tp_price': take_profit_price_raw}
            logger.info(f"{log_prefix_live} Requête d'ordre: {entry_params} avec SL/TP bruts: {sl_tp_prices}")
            return entry_params, sl_tp_prices
        
        logger.info(f"{log_prefix_live} Aucune condition de renversement PSAR remplie.")
        return None