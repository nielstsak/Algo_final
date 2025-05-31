# src/strategies/implementations/sma_cross_strategy.py
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    # Essayer d'importer depuis le chemin relatif (pour le chargement par strategy_loader)
    from ..base_strategy import BaseStrategy
    from ..technical_indicators import IndicatorManager, IndicatorCalculationError, IndicatorNotFoundError
except ImportError:
    # Fallback pour les tests ou exécution directe (si ce fichier n'est pas dans implementations)
    # Cela suppose que src est dans PYTHONPATH
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.technical_indicators import IndicatorManager, IndicatorCalculationError, IndicatorNotFoundError

try:
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError
    from src.utils.exchange_utils import (
        adjust_precision,
        get_filter_value,
        get_precision_from_filter
    )
    from src.core.constants import Trading, Kline
except ImportError:
    logging.getLogger(__name__).error(
        "SMACrossStrategy: Failed to import core.exceptions, utils.exchange_utils or core.constants. "
        "Using dummy versions or expect errors."
    )
    # Définitions factices minimales pour éviter les erreurs d'import globales si tout échoue
    class InvalidStrategyParamsError(Exception):
        def __init__(self, strategy_name: str, parameter_name: str, details: str, **kwargs):
            super().__init__(f"Invalid param {parameter_name} for {strategy_name}: {details}")
    class StrategyError(Exception): pass
    def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return 8
    def adjust_precision(value: float, precision: int, rounding_method=round) -> Optional[float]: return round(value, precision)
    class Trading: SIDE_BUY = "BUY"; SIDE_SELL = "SELL"; ORDER_TYPE_MARKET = "MARKET"; ORDER_TYPE_LIMIT = "LIMIT" # type: ignore
    class Kline: INTERVAL_1MINUTE = "1m" # type: ignore


logger = logging.getLogger(__name__)

class SMACrossStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur le croisement de deux moyennes mobiles simples (SMA).
    Un signal d'achat est généré lorsque la SMA rapide croise au-dessus de la SMA lente.
    Un signal de vente est généré lorsque la SMA rapide croise en dessous de la SMA lente.
    Le Stop Loss et le Take Profit sont basés sur l'Average True Range (ATR).
    """
    name: str = "SMACrossStrategy"
    version: str = "1.0.0"
    description: str = "Stratégie de croisement de moyennes mobiles (SMA) avec SL/TP basés sur l'ATR."

    default_params: Dict[str, Any] = {
        'fast_period': 10,
        'slow_period': 30,
        'indicator_frequency': Kline.INTERVAL_1HOUR, # Fréquence pour calculer MAs et ATR
        'atr_period_sl_tp': 14,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
        'allow_shorting': True, # Permettre ou non les positions courtes
        'order_type_preference': Trading.ORDER_TYPE_MARKET, # "MARKET" ou "LIMIT"
        'position_sizing_pct_capital': 0.02, # Pourcentage du capital à risquer par trade (ex: 2%)
    }

    # Timeframes de klines brutes requis. La stratégie peut resampler en interne si nécessaire.
    required_timeframes: List[str] = [Kline.INTERVAL_1MINUTE]
    
    # Nombre minimum de périodes de klines (à la fréquence de `indicator_frequency`)
    # nécessaires pour que la stratégie puisse calculer ses indicateurs.
    # Doit être au moins égal à max(slow_period, atr_period_sl_tp).
    min_required_periods: int = 50 # Valeur par défaut, sera recalculée dans __init__

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Constructeur de la stratégie.
        Args:
            params: Dictionnaire des paramètres pour surcharger les `default_params`.
            **kwargs: Arguments supplémentaires passés à la BaseStrategy (ex: pair_symbol).
        """
        self.pair_symbol = kwargs.get('pair_symbol', 'DEFAULT_PAIR')
        super().__init__(params, **kwargs) # Appel au constructeur de BaseStrategy

        # Valider les paramètres après la fusion avec default_params
        self.validate_params()

        # Initialiser le gestionnaire d'indicateurs
        self.indicator_manager = IndicatorManager()
        
        # Mettre à jour min_required_periods en fonction des paramètres réels
        self.min_required_periods = max(
            self.get_param('slow_period'),
            self.get_param('atr_period_sl_tp')
        ) + 5 # Ajouter une petite marge

        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée avec les paramètres: {self.params}")
        logger.info(f"{self.strategy_name_log_prefix} Min required periods (at indicator frequency): {self.min_required_periods}")

    def validate_params(self) -> None:
        """
        Valide les paramètres de la stratégie.
        Doit lever `InvalidStrategyParamsError` si un paramètre est invalide.
        """
        logger.debug(f"{self.strategy_name_log_prefix} Validation des paramètres...")
        
        fast_period = self.get_param('fast_period')
        slow_period = self.get_param('slow_period')
        atr_period = self.get_param('atr_period_sl_tp')
        sl_mult = self.get_param('sl_atr_mult')
        tp_mult = self.get_param('tp_atr_mult')
        pos_sizing_pct = self.get_param('position_sizing_pct_capital')

        if not (isinstance(fast_period, int) and fast_period > 0):
            raise InvalidStrategyParamsError(self.name, "fast_period", "Doit être un entier positif.")
        if not (isinstance(slow_period, int) and slow_period > 0):
            raise InvalidStrategyParamsError(self.name, "slow_period", "Doit être un entier positif.")
        if fast_period >= slow_period:
            raise InvalidStrategyParamsError(self.name, "fast_period/slow_period", 
                                             f"fast_period ({fast_period}) doit être inférieur à slow_period ({slow_period}).")
        if not (isinstance(atr_period, int) and atr_period > 0):
            raise InvalidStrategyParamsError(self.name, "atr_period_sl_tp", "Doit être un entier positif.")
        if not (isinstance(sl_mult, (int, float)) and sl_mult > 0):
            raise InvalidStrategyParamsError(self.name, "sl_atr_mult", "Doit être un nombre positif.")
        if not (isinstance(tp_mult, (int, float)) and tp_mult > 0):
            raise InvalidStrategyParamsError(self.name, "tp_atr_mult", "Doit être un nombre positif.")
        if not (isinstance(pos_sizing_pct, (int, float)) and 0 < pos_sizing_pct <= 1):
             raise InvalidStrategyParamsError(self.name, "position_sizing_pct_capital", "Doit être entre 0 (exclus) et 1 (inclus).")

        order_type = self.get_param('order_type_preference')
        if order_type not in [Trading.ORDER_TYPE_MARKET, Trading.ORDER_TYPE_LIMIT]:
            raise InvalidStrategyParamsError(self.name, "order_type_preference", f"Type d'ordre non supporté: {order_type}.")

        logger.info(f"{self.strategy_name_log_prefix} Paramètres validés.")

    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calcule les indicateurs techniques nécessaires pour la stratégie.
        Cette méthode est appelée par le moteur de backtesting/live trading.

        Args:
            klines: Dictionnaire où les clés sont les timeframes (ex: "1m", "1h")
                    et les valeurs sont des DataFrames de klines pour ce timeframe.
                    Ces DataFrames doivent contenir au moins les colonnes OHLCV.

        Returns:
            Dictionnaire similaire à `klines`, mais où les DataFrames contiennent
            les indicateurs calculés en plus des données OHLCV.
            Le DataFrame principal (celui de `indicator_frequency`)
            doit contenir tous les indicateurs finaux utilisés par `generate_signals`.
        """
        log_pref = self.strategy_name_log_prefix
        base_tf_raw = self.required_timeframes[0] # Ex: '1m'
        indicator_tf_target = self.get_param('indicator_frequency') # Ex: '1h'

        if base_tf_raw not in klines or klines[base_tf_raw] is None or klines[base_tf_raw].empty:
            raise StrategyError(f"{log_pref} Données pour timeframe de base '{base_tf_raw}' manquantes ou vides.", strategy_name=self.name)

        df_base = klines[base_tf_raw].copy()
        
        # S'assurer que les colonnes OHLCV standard sont présentes sur df_base
        # et renommées si elles utilisent les noms de Binance (open_price -> open)
        rename_map = {
            'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 
            'close_price': 'close', 'base_asset_volume': 'volume'
        }
        df_base.rename(columns=rename_map, inplace=True)

        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlcv:
            if col not in df_base.columns:
                raise IndicatorCalculationError(f"{log_pref} Colonne OHLCV standard '{col}' manquante sur les données '{base_tf_raw}'.")
            df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
        
        if df_base[required_ohlcv].isnull().any().any():
            logger.warning(f"{log_pref} Valeurs NaN trouvées dans les colonnes OHLCV de base. Le calcul des indicateurs peut être affecté.")
            # Option: df_base.dropna(subset=required_ohlcv, inplace=True) ou fillna

        # Resample df_base à la fréquence des indicateurs (indicator_tf_target)
        if base_tf_raw == indicator_tf_target:
            df_indicator_freq = df_base.copy()
        else:
            try:
                agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                df_indicator_freq = df_base.resample(indicator_tf_target).agg(agg_rules)
                df_indicator_freq.dropna(subset=['close'], inplace=True) # Supprimer les périodes sans clôture
                if df_indicator_freq.empty:
                    raise IndicatorCalculationError(f"{log_pref} Resampling de '{base_tf_raw}' vers '{indicator_tf_target}' a produit un DataFrame vide.")
            except Exception as e:
                raise IndicatorCalculationError(f"{log_pref} Erreur lors du resampling vers '{indicator_tf_target}': {e}", original_exception=e)
        
        if len(df_indicator_freq) < self.min_required_periods:
             logger.warning(f"{log_pref} Pas assez de données ({len(df_indicator_freq)} lignes) à la fréquence '{indicator_tf_target}' "
                            f"après resampling pour calculer les indicateurs (min requis: {self.min_required_periods}).")
             # Retourner un DF vide ou avec des NaN pour les indicateurs
             df_indicator_freq[f"SMA_{self.get_param('fast_period')}"] = np.nan
             df_indicator_freq[f"SMA_{self.get_param('slow_period')}"] = np.nan
             df_indicator_freq[f"ATR_{self.get_param('atr_period_sl_tp')}"] = np.nan
             self._indicators_cache = {indicator_tf_target: df_indicator_freq}
             return self._indicators_cache


        # Calcul des indicateurs
        try:
            # SMA Rapide
            df_indicator_freq = self.indicator_manager.calculate_indicator(
                df_indicator_freq,
                indicator_name="sma",
                length=self.get_param('fast_period'),
                output_col_prefix=f"SMA_F{self.get_param('fast_period')}" # Ex: SMA_F10_length10
            )
            self.sma_fast_col = f"SMA_F{self.get_param('fast_period')}_SMA_{self.get_param('fast_period')}" # Nom par défaut de pandas-ta

            # SMA Lente
            df_indicator_freq = self.indicator_manager.calculate_indicator(
                df_indicator_freq,
                indicator_name="sma",
                length=self.get_param('slow_period'),
                output_col_prefix=f"SMA_S{self.get_param('slow_period')}"
            )
            self.sma_slow_col = f"SMA_S{self.get_param('slow_period')}_SMA_{self.get_param('slow_period')}"

            # ATR pour SL/TP
            df_indicator_freq = self.indicator_manager.calculate_indicator(
                df_indicator_freq,
                indicator_name="atr",
                length=self.get_param('atr_period_sl_tp'),
                output_col_prefix=f"ATR_SLTP{self.get_param('atr_period_sl_tp')}"
            )
            self.atr_col = f"ATR_SLTP{self.get_param('atr_period_sl_tp')}_ATR_{self.get_param('atr_period_sl_tp')}"

        except (IndicatorNotFoundError, IndicatorCalculationError) as e:
            logger.error(f"{log_pref} Erreur de calcul d'indicateur: {e}")
            raise
        except Exception as e:
            logger.error(f"{log_pref} Erreur inattendue lors du calcul des indicateurs: {e}")
            raise IndicatorCalculationError(f"Erreur inattendue: {e}", original_exception=e)

        # Validation des indicateurs (exemple)
        if self.sma_fast_col in df_indicator_freq.columns:
            self.indicator_manager.validate_indicator_output(df_indicator_freq, self.sma_fast_col, expected_lookback=self.get_param('fast_period'))
        if self.atr_col in df_indicator_freq.columns:
            self.indicator_manager.validate_indicator_output(df_indicator_freq, self.atr_col, expected_lookback=self.get_param('atr_period_sl_tp'), value_range=(0, None))
        
        logger.info(f"{log_pref} Indicateurs calculés sur la fréquence '{indicator_tf_target}'.")
        
        self._indicators_cache = {indicator_tf_target: df_indicator_freq}
        return self._indicators_cache

    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Génère les signaux de trading (pour le backtesting).
        Utilise les indicateurs calculés à la `indicator_frequency`.
        """
        log_pref = self.strategy_name_log_prefix
        indicator_tf = self.get_param('indicator_frequency')

        if indicator_tf not in indicators or indicators[indicator_tf] is None:
            logger.error(f"{log_pref} DataFrame pour la fréquence des indicateurs '{indicator_tf}' non trouvé.")
            # Retourner un DataFrame de signaux vide avec la structure attendue
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).astype(
                {'entry_long': bool, 'exit_long': bool, 'entry_short': bool, 'exit_short': bool, 'sl': float, 'tp': float}
            )

        df = indicators[indicator_tf].copy()
        logger.debug(f"{log_pref} Génération des signaux (backtesting) sur les données de '{indicator_tf}'.")

        # Vérifier si les colonnes d'indicateurs nécessaires existent
        required_indicator_cols = [self.sma_fast_col, self.sma_slow_col, self.atr_col, 'close']
        missing_cols = [col for col in required_indicator_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"{log_pref} Colonnes d'indicateurs manquantes pour générer les signaux: {missing_cols}. Signaux vides seront retournés.")
            df_signals = pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            df_signals[['entry_long', 'exit_long', 'entry_short', 'exit_short']] = False
            df_signals[['sl', 'tp']] = np.nan
            self._signals = df_signals # Stocker pour get_signals()
            return self._signals
        
        if df[required_indicator_cols].isnull().all().any(): # Si une colonne entière est NaN
            logger.warning(f"{log_pref} Au moins une colonne d'indicateur essentielle est entièrement NaN. Signaux vides.")
            df_signals = pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            df_signals[['entry_long', 'exit_long', 'entry_short', 'exit_short']] = False
            df_signals[['sl', 'tp']] = np.nan
            self._signals = df_signals
            return self._signals


        # Conditions de croisement
        sma_fast = df[self.sma_fast_col]
        sma_slow = df[self.sma_slow_col]

        # Croisement haussier: SMA rapide passe AU-DESSUS de SMA lente
        long_entry_condition = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        # Croisement baissier: SMA rapide passe EN DESSOUS de SMA lente
        short_entry_condition = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

        df_signals = pd.DataFrame(index=df.index)
        df_signals['entry_long'] = long_entry_condition
        df_signals['exit_long'] = short_entry_condition # Sortie de long sur croisement baissier

        if self.get_param('allow_shorting'):
            df_signals['entry_short'] = short_entry_condition
            df_signals['exit_short'] = long_entry_condition # Sortie de short sur croisement haussier
        else:
            df_signals['entry_short'] = False
            df_signals['exit_short'] = False
        
        # Calcul SL/TP
        atr_series = df[self.atr_col]
        sl_mult = self.get_param('sl_atr_mult')
        tp_mult = self.get_param('tp_atr_mult')
        
        # Utiliser le prix de clôture de la bougie où le signal se produit pour calculer SL/TP
        entry_price_series = df['close'] 

        df_signals['sl'] = np.nan
        df_signals['tp'] = np.nan

        # SL/TP pour les entrées LONG
        long_entry_indices = df_signals[df_signals['entry_long']].index
        if not long_entry_indices.empty:
            valid_atr_long = atr_series.loc[long_entry_indices].notna()
            df_signals.loc[long_entry_indices[valid_atr_long], 'sl'] = \
                entry_price_series.loc[long_entry_indices[valid_atr_long]] - (atr_series.loc[long_entry_indices[valid_atr_long]] * sl_mult)
            df_signals.loc[long_entry_indices[valid_atr_long], 'tp'] = \
                entry_price_series.loc[long_entry_indices[valid_atr_long]] + (atr_series.loc[long_entry_indices[valid_atr_long]] * tp_mult)

        # SL/TP pour les entrées SHORT
        if self.get_param('allow_shorting'):
            short_entry_indices = df_signals[df_signals['entry_short']].index
            if not short_entry_indices.empty:
                valid_atr_short = atr_series.loc[short_entry_indices].notna()
                df_signals.loc[short_entry_indices[valid_atr_short], 'sl'] = \
                    entry_price_series.loc[short_entry_indices[valid_atr_short]] + (atr_series.loc[short_entry_indices[valid_atr_short]] * sl_mult)
                df_signals.loc[short_entry_indices[valid_atr_short], 'tp'] = \
                    entry_price_series.loc[short_entry_indices[valid_atr_short]] - (atr_series.loc[short_entry_indices[valid_atr_short]] * tp_mult)

        # S'assurer que les colonnes de signaux sont booléennes
        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            df_signals[col_b] = df_signals[col_b].fillna(False).astype(bool)
        
        self._signals = df_signals[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].copy()
        logger.info(f"{log_pref} Signaux (backtesting) générés. Entrées Long: {self._signals['entry_long'].sum()}, Entrées Short: {self._signals['entry_short'].sum()}.")
        return self._signals

    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame], # C'est le _indicators_cache
                               symbol: str, # pair_symbol de BaseStrategy
                               current_position: int, # -1 (short), 0 (neutre), 1 (long)
                               available_capital: float, # Capital dispo pour CETTE stratégie sur CETTE paire
                               symbol_info: dict # Infos de l'exchange sur le symbole (pour précision, etc.)
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre pour le trading live.
        Utilise la dernière ligne des indicateurs calculés à `indicator_frequency`.

        Args:
            data_dict: Dictionnaire des DataFrames d'indicateurs.
            symbol: Symbole de la paire.
            current_position: Position actuelle (-1 short, 0 neutre, 1 long).
            available_capital: Capital disponible pour cette stratégie/paire.
            symbol_info: Informations sur le symbole de l'exchange.

        Returns:
            Tuple optionnel contenant:
            - dict: Paramètres de l'ordre d'entrée formatés pour l'exchange.
            - dict: Prix bruts SL/TP {'sl_price': float, 'tp_price': float}.
            Retourne None si aucune action d'ordre n'est générée.
        """
        log_pref = f"{self.strategy_name_log_prefix}[LiveOrder]"
        indicator_tf = self.get_param('indicator_frequency')

        if indicator_tf not in data_dict or data_dict[indicator_tf] is None or data_dict[indicator_tf].empty:
            logger.warning(f"{log_pref} Données pour timeframe indicateurs '{indicator_tf}' manquantes ou vides.")
            return None
        
        df_indicators = data_dict[indicator_tf]
        if len(df_indicators) < 2: # Besoin d'au moins 2 points pour .shift(1)
            logger.warning(f"{log_pref} Pas assez de données d'indicateurs ({len(df_indicators)} lignes) pour décision live.")
            return None

        # Utiliser la dernière et l'avant-dernière ligne des indicateurs
        latest_indicators = df_indicators.iloc[-1]
        previous_indicators = df_indicators.iloc[-2]

        # Vérifier que les indicateurs clés ne sont pas NaN
        required_cols_live = [self.sma_fast_col, self.sma_slow_col, self.atr_col, 'close']
        if latest_indicators[required_cols_live].isnull().any() or \
           previous_indicators[[self.sma_fast_col, self.sma_slow_col]].isnull().any():
            nan_cols_latest = latest_indicators[required_cols_live].index[latest_indicators[required_cols_live].isnull()].tolist()
            nan_cols_prev = previous_indicators[[self.sma_fast_col, self.sma_slow_col]].index[previous_indicators[[self.sma_fast_col, self.sma_slow_col]].isnull()].tolist()
            logger.warning(f"{log_pref} Indicateurs NaN sur dernières données. Latest: {nan_cols_latest}, Previous: {nan_cols_prev}. Pas d'ordre.")
            return None

        sma_fast_curr = latest_indicators[self.sma_fast_col]
        sma_slow_curr = latest_indicators[self.sma_slow_col]
        sma_fast_prev = previous_indicators[self.sma_fast_col]
        sma_slow_prev = previous_indicators[self.sma_slow_col]
        
        current_price = latest_indicators['close'] # Prix actuel pour référence
        atr_value = latest_indicators[self.atr_col]

        order_side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        # Logique de signal pour la dernière bougie
        is_long_entry_signal = (sma_fast_curr > sma_slow_curr) and (sma_fast_prev <= sma_slow_prev)
        is_short_entry_signal = (sma_fast_curr < sma_slow_curr) and (sma_fast_prev >= sma_slow_prev)
        
        is_long_exit_signal = is_short_entry_signal # Un croisement baissier est une sortie de long
        is_short_exit_signal = is_long_entry_signal  # Un croisement haussier est une sortie de short

        if current_position == 0: # Pas de position, chercher une entrée
            if pd.isna(atr_value) or atr_value <= 1e-9: # ATR doit être valide pour SL/TP
                 if is_long_entry_signal or (self.get_param('allow_shorting') and is_short_entry_signal):
                    logger.warning(f"{log_pref} ATR invalide ({atr_value}). Impossible de calculer SL/TP. Pas d'entrée.")
            elif is_long_entry_signal:
                order_side = Trading.SIDE_BUY
                stop_loss_price_raw = current_price - (atr_value * self.get_param('sl_atr_mult'))
                take_profit_price_raw = current_price + (atr_value * self.get_param('tp_atr_mult'))
                logger.info(f"{log_pref} Signal Entrée LONG. Prix: {current_price:.4f}, ATR: {atr_value:.4f} -> SL brut: {stop_loss_price_raw:.4f}, TP brut: {take_profit_price_raw:.4f}")
            elif self.get_param('allow_shorting') and is_short_entry_signal:
                order_side = Trading.SIDE_SELL
                stop_loss_price_raw = current_price + (atr_value * self.get_param('sl_atr_mult'))
                take_profit_price_raw = current_price - (atr_value * self.get_param('tp_atr_mult'))
                logger.info(f"{log_pref} Signal Entrée SHORT. Prix: {current_price:.4f}, ATR: {atr_value:.4f} -> SL brut: {stop_loss_price_raw:.4f}, TP brut: {take_profit_price_raw:.4f}")
        
        elif current_position == 1: # Position longue ouverte, chercher une sortie
            if is_long_exit_signal:
                logger.info(f"{log_pref} Signal Sortie de position LONG (croisement baissier).")
                # Générer un ordre de vente MARKET pour fermer la position
                # La quantité sera la taille de la position actuelle (gérée par le moteur de trading live)
                entry_params = self._build_entry_params_formatted(symbol=symbol, side=Trading.SIDE_SELL, quantity_str="POS_SIZE", order_type=Trading.ORDER_TYPE_MARKET)
                return entry_params, {} # Pas de SL/TP pour un ordre de sortie explicite basé sur signal
            # Le moteur de trading live gérera les SL/TP actifs.

        elif current_position == -1: # Position courte ouverte, chercher une sortie
            if self.get_param('allow_shorting') and is_short_exit_signal:
                logger.info(f"{log_pref} Signal Sortie de position SHORT (croisement haussier).")
                entry_params = self._build_entry_params_formatted(symbol=symbol, side=Trading.SIDE_BUY, quantity_str="POS_SIZE", order_type=Trading.ORDER_TYPE_MARKET)
                return entry_params, {}
            # Le moteur de trading live gérera les SL/TP actifs.

        # Si un signal d'entrée a été généré
        if order_side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            # Récupérer les précisions de prix et de quantité
            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            
            if price_precision_val is None or qty_precision_val is None:
                logger.error(f"{log_pref} Précision prix/quantité non trouvée pour {symbol} dans symbol_info.")
                return None

            # Calculer la quantité
            # La méthode _calculate_quantity est héritée de BaseStrategy
            quantity = self._calculate_quantity(
                entry_price=current_price, # Prix actuel pour le calcul de risque
                available_capital=available_capital,
                qty_precision=qty_precision_val,
                symbol_info=symbol_info,
                symbol=symbol,
                risk_per_trade_pct=self.get_param('position_sizing_pct_capital'),
                stop_loss_price=stop_loss_price_raw # SL brut pour le calcul de risque
            )

            if quantity is None or quantity <= 0:
                logger.warning(f"{log_pref} Quantité calculée invalide ({quantity}). Pas d'ordre.")
                return None
            
            # Formater le prix d'entrée et la quantité selon les précisions
            # Pour un ordre MARKET, le prix d'entrée n'est pas spécifié.
            # Pour un ordre LIMIT, on pourrait utiliser current_price ou un prix légèrement ajusté.
            entry_price_for_order_str: Optional[str] = None
            if self.get_param('order_type_preference') == Trading.ORDER_TYPE_LIMIT:
                # Pour un ordre LIMIT, on peut viser le prix actuel ou un léger mieux
                limit_entry_price = current_price # Ou ajuster (ex: current_price - tick_size pour un achat)
                limit_entry_price_adj = adjust_precision(limit_entry_price, price_precision_val)
                if limit_entry_price_adj is None: return None
                entry_price_for_order_str = f"{limit_entry_price_adj:.{price_precision_val}f}"
            
            quantity_str = f"{quantity:.{qty_precision_val}f}"
            
            # Construire les paramètres de l'ordre d'entrée
            # La méthode _build_entry_params_formatted est héritée de BaseStrategy
            entry_params = self._build_entry_params_formatted(
                symbol=symbol,
                side=order_side,
                quantity_str=quantity_str,
                entry_price_str=entry_price_for_order_str, # None pour MARKET
                order_type=self.get_param('order_type_preference')
            )

            if not entry_params:
                logger.error(f"{log_pref} Échec de la construction des paramètres d'ordre.")
                return None

            sl_tp_prices_dict = {'sl_price': stop_loss_price_raw, 'tp_price': take_profit_price_raw}
            
            logger.info(f"{log_pref} Requête d'ordre générée: {entry_params} avec SL/TP bruts: {sl_tp_prices_dict}")
            return entry_params, sl_tp_prices_dict
            
        logger.debug(f"{log_pref} Aucune action d'ordre générée sur la dernière kline.")
        return None


if __name__ == '__main__':
    # Configuration du logger pour les tests
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # Créer un DataFrame de test avec des colonnes renommées comme attendu par la stratégie
    data_test = {
        'open': np.array([10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 10, 11]),
        'high': np.array([11, 12, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 11, 12]),
        'low': np.array([9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 9, 10]),
        'close': np.array([11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 10, 11, 10]),
        'volume': np.array([100]*18) * 1.0, # Assurer que c'est float
    }
    index_test = pd.date_range(start='2023-01-01 00:00:00', periods=18, freq='1h')
    df_klines_test = pd.DataFrame(data_test, index=index_test)
    
    # Simuler des klines 1m pour le resampling
    df_klines_1m_test = df_klines_test.resample('1min').ffill().dropna() # Simple ffill pour l'exemple
    # S'assurer que les colonnes OHLCV sont bien présentes après resampling
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df_klines_1m_test.columns and col in df_klines_test.columns:
            df_klines_1m_test[col] = df_klines_test[col].resample('1min').ffill()
    df_klines_1m_test.dropna(inplace=True)


    klines_dict_test = {Kline.INTERVAL_1MINUTE: df_klines_1m_test.copy()}

    # Test avec des paramètres par défaut
    params_sma_cross = {
        'fast_period': 3,
        'slow_period': 6,
        'indicator_frequency': '1h', # Les données de test sont déjà à 1h, donc pas de resampling effectif
        'atr_period_sl_tp': 4,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 1.5,
        'allow_shorting': True
    }
    
    strategy_instance = SMACrossStrategy(params=params_sma_cross, pair_symbol="BTCUSDT_TEST")

    logger.info("\n--- Test de calculate_indicators ---")
    try:
        indicators_result = strategy_instance.calculate_indicators(klines_dict_test)
        df_indicators_test = indicators_result[params_sma_cross['indicator_frequency']]
        
        logger.info(f"Indicateurs calculés ({params_sma_cross['indicator_frequency']}):")
        logger.info(f"Colonnes: {df_indicators_test.columns.tolist()}")
        # Afficher les colonnes pertinentes pour vérification
        cols_to_show = ['close', strategy_instance.sma_fast_col, strategy_instance.sma_slow_col, strategy_instance.atr_col]
        cols_present = [col for col in cols_to_show if col in df_indicators_test.columns]
        if cols_present:
            logger.info(df_indicators_test[cols_present].tail())
        else:
            logger.warning("Colonnes d'indicateurs attendues non trouvées après calcul.")

        logger.info("\n--- Test de generate_signals ---")
        signals_df_test = strategy_instance.generate_signals(indicators_result)
        logger.info("Signaux générés:")
        logger.info(signals_df_test[
            (signals_df_test['entry_long']) | (signals_df_test['entry_short']) |
            (signals_df_test['exit_long']) | (signals_df_test['exit_short'])
        ])

        logger.info("\n--- Test de generate_order_request (exemple simple) ---")
        # Simuler les informations pour un ordre
        symbol_info_test = {
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'},
                {'filterType': 'LOT_SIZE', 'stepSize': '0.001'}
            ]
        }
        # Cas: Pas de position, signal d'achat sur la dernière bougie
        # Pour cela, il faut que les indicateurs à la fin du df_indicators_test génèrent un signal
        # On peut forcer un signal pour tester la logique d'ordre
        
        # Si la dernière ligne des indicateurs génère un signal d'achat:
        # (sma_fast_curr > sma_slow_curr) and (sma_fast_prev <= sma_slow_prev)
        # On va vérifier les dernières valeurs
        if not df_indicators_test.empty and len(df_indicators_test) >=2:
            last_ind = df_indicators_test.iloc[-1]
            prev_ind = df_indicators_test.iloc[-2]
            sma_f_curr = last_ind[strategy_instance.sma_fast_col]
            sma_s_curr = last_ind[strategy_instance.sma_slow_col]
            sma_f_prev = prev_ind[strategy_instance.sma_fast_col]
            sma_s_prev = prev_ind[strategy_instance.sma_slow_col]

            logger.info(f"Dernières valeurs pour test d'ordre: SMA_F_CUR={sma_f_curr}, SMA_S_CUR={sma_s_curr}, SMA_F_PREV={sma_f_prev}, SMA_S_PREV={sma_s_prev}")

            if pd.notna(sma_f_curr) and pd.notna(sma_s_curr) and pd.notna(sma_f_prev) and pd.notna(sma_s_prev):
                 order_req = strategy_instance.generate_order_request(
                    data_dict=indicators_result,
                    symbol="BTCUSDT_TEST",
                    current_position=0, # Pas de position
                    available_capital=1000.0,
                    symbol_info=symbol_info_test
                )
                 if order_req:
                    logger.info(f"Requête d'ordre générée: {order_req[0]}")
                    logger.info(f"SL/TP bruts: {order_req[1]}")
                 else:
                    logger.info("Aucune requête d'ordre générée pour la dernière bougie.")
            else:
                logger.warning("Indicateurs NaN sur les dernières lignes, test generate_order_request non concluant.")
        else:
            logger.warning("Pas assez de données d'indicateurs pour tester generate_order_request.")


    except Exception as e:
        logger.exception(f"Erreur lors des tests de SMACrossStrategy: {e}")

