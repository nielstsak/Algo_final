# src/strategies/implementations/strategy_template.py
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Importations nécessaires depuis votre projet
try:
    # Essayer d'importer depuis le chemin relatif .base (pour le chargement par strategy_loader)
    from ..base import BaseStrategy # Ajustez si BaseStrategy est ailleurs
except ImportError:
    # Fallback pour les tests ou exécution directe
    try:
        from src.strategies.base import BaseStrategy
    except ImportError:
        logging.getLogger(__name__).critical("StrategyTemplate: CRITICAL - BaseStrategy not found.")
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
            # Méthodes utilitaires (supposées être dans la vraie BaseStrategy)
            def _calculate_quantity(self, entry_price: float, available_capital: float, qty_precision: int, symbol_info: dict, symbol: str, risk_per_trade_pct: float, stop_loss_price: float) -> Optional[float]: return 1.0
            def _build_entry_params_formatted(self, symbol: str, side: str, quantity_str: str, entry_price_str: Optional[str] = None, order_type: str = "MARKET") -> Optional[Dict[str, Any]]: return {"symbol": symbol, "side": side, "quantity": quantity_str, "type": order_type, "price": entry_price_str}


try:
    # Utilisation d'un import relatif pour plus de robustesse dans les packages
    from ...core.exceptions import InvalidStrategyParamsError, StrategyError, IndicatorCalculationError
    from ...utils.exchange_utils import (adjust_precision, # Assumant que utils est aussi sous src
                                          get_filter_value,
                                          get_precision_from_filter)
except ImportError:
    logging.getLogger(__name__).error("StrategyTemplate: Failed to import core.exceptions or utils.exchange_utils. Using dummy versions.")
    # Fallback si les imports relatifs échouent (par exemple, si exécuté comme script standalone non dans un package)
    try:
        from src.core.exceptions import InvalidStrategyParamsError, StrategyError, IndicatorCalculationError
        from src.utils.exchange_utils import (adjust_precision,
                                              get_filter_value,
                                              get_precision_from_filter)
    except ImportError:
        def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None
        def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return None
        def adjust_precision(value: float, precision: int, rounding_method = round) -> Optional[float]: return value # type: ignore
        class InvalidStrategyParamsError(Exception): # type: ignore
            def __init__(self, message: str, *args, **kwargs): 
                super().__init__(message)
                # Stocker les arguments nommés s'ils sont fournis, pour correspondre à la vraie classe
                self.strategy_name = kwargs.get('strategy_name')
                self.parameter_name = kwargs.get('parameter_name')
                self.details = kwargs.get('details')
                self.original_exception = kwargs.get('original_exception')


        class StrategyError(Exception): # type: ignore
             def __init__(self, message: str, *args, **kwargs):
                super().__init__(message)
                self.strategy_name = kwargs.get('strategy_name')
                self.original_exception = kwargs.get('original_exception')

        class IndicatorCalculationError(StrategyError): pass # type: ignore


logger = logging.getLogger(__name__)

class StrategyTemplate(BaseStrategy):
    """
    Template pour une nouvelle stratégie de trading.
    Cette classe sert de modèle et doit être adaptée pour une logique spécifique.
    """
    # --- Attributs de Classe Obligatoires ---
    name: str = "StrategyTemplate" # Nom unique et descriptif de la stratégie
    version: str = "1.0.0"        # Version de la stratégie
    description: str = "Un template de base pour développer de nouvelles stratégies." # Description concise

    # --- Paramètres par Défaut ---
    # Définissez ici les paramètres que votre stratégie utilisera, avec leurs valeurs par défaut.
    # Ces paramètres peuvent être surchargés lors de l'instanciation ou de l'optimisation.
    default_params: Dict[str, Any] = {
        "param_exemple_1": 10,              # Exemple: période pour un indicateur
        "param_exemple_2": "valeur_defaut", # Exemple: chaîne de caractères
        "param_exemple_3": True,            # Exemple: booléen
        "indicateur_frequence": "1h",       # Fréquence pour le calcul des indicateurs
        "sl_atr_mult": 1.5,                 # Multiplicateur ATR pour Stop Loss
        "tp_atr_mult": 2.0,                 # Multiplicateur ATR pour Take Profit
        "atr_period_sl_tp": 14,             # Période ATR pour SL/TP
        "position_sizing_pct_capital": 0.02 # Pourcentage du capital à risquer par trade
    }

    # --- Configuration des Timeframes ---
    # Liste des timeframes de klines brutes requis par la stratégie.
    # Le premier timeframe est généralement celui sur lequel les signaux sont générés.
    required_timeframes: List[str] = ["1m"] # Exemple: stratégie basée sur klines 1 minute

    # --- Périodes Minimales Requises ---
    # Nombre minimum de périodes de klines nécessaires pour que la stratégie
    # puisse calculer ses indicateurs et fonctionner correctement.
    # Doit être au moins égal à la plus longue période d'indicateur utilisée.
    min_required_periods: int = 50 # Exemple, à ajuster

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Constructeur de la stratégie.
        Args:
            params: Dictionnaire des paramètres pour surcharger les `default_params`.
            **kwargs: Arguments supplémentaires passés à la BaseStrategy (ex: pair_symbol).
        """
        self.pair_symbol = kwargs.get('pair_symbol', 'DEFAULT_PAIR') 

        super().__init__(params, **kwargs) 

        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]" 

        self.validate_params()


        self._signals: Optional[pd.DataFrame] = None 

        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée avec les paramètres: {self.params}")

    def validate_params(self) -> None:
        """
        Valide les paramètres de la stratégie.
        Cette méthode est appelée automatiquement par `__init__`.
        Doit lever `InvalidStrategyParamsError` si un paramètre est invalide.
        """
        logger.debug(f"{self.strategy_name_log_prefix} Validation des paramètres...")
        param1 = self.get_param("param_exemple_1")
        if param1 is None or not isinstance(param1, int) or param1 <= 0:
            # Correction: Passer 'message' explicitement
            raise InvalidStrategyParamsError(
                message=f"Le paramètre 'param_exemple_1' doit être un entier positif. Reçu: {param1}",
                strategy_name=self.name, 
                parameter_name="param_exemple_1", 
                details="Doit être un entier positif." 
            )

        if self.get_param('sl_atr_mult') <= 0 or self.get_param('tp_atr_mult') <= 0:
            # Correction: Passer 'message' explicitement
            raise InvalidStrategyParamsError(
                message="Les multiplicateurs SL/TP ATR doivent être positifs.",
                strategy_name=self.name, 
                parameter_name='sl_atr_mult/tp_atr_mult', 
                details="Doivent être positifs." 
            )
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
            Le DataFrame principal (celui du premier `required_timeframes` ou de `indicateur_frequence`)
            doit contenir tous les indicateurs finaux utilisés par `generate_signals`.
        """
        log_pref = self.strategy_name_log_prefix
        # indicator_tf = self.get_param("indicateur_frequence") # Non utilisé directement ici si les indicateurs sont pré-calculés
        main_tf_raw = self.required_timeframes[0] 
        if main_tf_raw not in klines or klines[main_tf_raw] is None:
            # Correction: Passer 'message' explicitement pour StrategyError si sa version dummy est utilisée
            raise StrategyError(message=f"{log_pref} DataFrame pour '{main_tf_raw}' non fourni.", strategy_name=self.name)
        
        df_indicators = klines[main_tf_raw].copy()
        logger.debug(f"{log_pref} Entrée calculate_indicators. Utilisation des données de '{main_tf_raw}'.")

        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlc:
            if col not in df_indicators.columns:
                df_indicators[col] = np.nan 
                logger.warning(f"{log_pref} Colonne OHLC '{col}' manquante sur '{main_tf_raw}'. Ajoutée avec NaN.")
            else:
                df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')


        # --- Logique de Calcul des Indicateurs ---
        # Si les indicateurs sont calculés ici (plutôt que pré-calculés) :
        # Exemple avec pandas_ta (s'assurer qu'il est importé et disponible)
        # try:
        #     import pandas_ta as ta
        # except ImportError:
        #     logger.error(f"{log_pref} pandas_ta non trouvé. Impossible de calculer les indicateurs.")
        #     # Remplir avec NaN pour éviter des erreurs plus tard
        #     df_indicators['EMA_exemple'] = np.nan
        #     df_indicators[self.atr_col_strat] = np.nan # Assurez-vous que self.atr_col_strat est défini
        #     self._indicators_cache = {main_tf_raw: df_indicators}
        #     return self._indicators_cache

        # # Exemple: Calcul d'une EMA et de l'ATR
        # ema_period = self.get_param("param_exemple_1")
        # df_indicators['EMA_exemple'] = df_indicators.ta.ema(length=ema_period, close=df_indicators['close'], append=False)
        
        # atr_period = self.get_param("atr_period_sl_tp")
        # if 'high' in df_indicators and 'low' in df_indicators and 'close' in df_indicators:
        #    df_indicators[self.atr_col_strat] = df_indicators.ta.atr(length=atr_period, high=df_indicators['high'], low=df_indicators['low'], close=df_indicators['close'], append=False)
        # else:
        #    logger.warning(f"{log_pref} Colonnes high/low/close manquantes pour calculer ATR.")
        #    df_indicators[self.atr_col_strat] = np.nan
        
        # Pour ce template, on suppose que les colonnes d'indicateurs spécifiques
        # (ex: `self.rsi_col_strat`, `self.atr_col_strat`) sont déjà présentes dans `df_indicators`
        # car elles ont été calculées en amont par un processus de feature engineering
        # et nommées selon les conventions attendues (ex: "RSI_1h_p14", "ATR_1h_p14").
        # On vérifie juste leur présence et on les ajoute avec NaN si manquantes.
        
        # Exemple de colonnes d'indicateurs attendues (à définir dans __init__)
        # expected_indicator_cols_strat = [self.rsi_col_strat, self.atr_col_strat]
        # for col_name in expected_indicator_cols_strat:
        #     if col_name not in df_indicators.columns:
        #         df_indicators[col_name] = np.nan
        #         logger.warning(f"{log_pref} Colonne indicateur stratégie '{col_name}' manquante. Ajoutée avec NaN.")

        logger.info(f"{log_pref} Indicateurs vérifiés/préparés.")
        
        self._indicators_cache = {main_tf_raw: df_indicators} 
        return self._indicators_cache

    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Génère les signaux de trading (pour le backtesting).
        """
        log_pref = self.strategy_name_log_prefix
        signal_tf = self.required_timeframes[0] 

        if signal_tf not in indicators or indicators[signal_tf] is None:
            logger.error(f"{log_pref} DataFrame pour '{signal_tf}' non trouvé dans les indicateurs pour generate_signals.")
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).astype(
                {'entry_long': bool, 'exit_long': bool, 'entry_short': bool, 'exit_short': bool, 'sl': float, 'tp': float}
            )

        df_signals = indicators[signal_tf].copy()
        logger.debug(f"{log_pref} Génération des signaux (backtesting) sur les données de '{signal_tf}'.")
        
        # --- Logique de Génération de Signaux (Placeholder) ---
        df_signals['entry_long'] = False
        df_signals['exit_long'] = False
        df_signals['entry_short'] = False
        df_signals['exit_short'] = False
        df_signals['sl'] = np.nan
        df_signals['tp'] = np.nan
        
        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            if col_b in df_signals.columns:
                df_signals[col_b] = df_signals[col_b].fillna(False).astype(bool)
            else:
                df_signals[col_b] = False

        self._signals = df_signals[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].copy()

        logger.info(f"{log_pref} Signaux générés (backtesting). Entrées Long: {self._signals['entry_long'].sum()}, Entrées Short: {self._signals['entry_short'].sum()}.")
        return self._signals

    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame], 
                               symbol: str, 
                               current_position: int, 
                               available_capital: float, 
                               symbol_info: dict 
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre pour le trading live.
        """
        log_pref = self.strategy_name_log_prefix
        indicator_tf = self.get_param("indicateur_frequence")

        if indicator_tf not in data_dict or data_dict[indicator_tf] is None or data_dict[indicator_tf].empty:
            logger.warning(f"{log_pref} Données pour timeframe indicateurs '{indicator_tf}' manquantes ou vides pour generate_order_request.")
            return None
            
        df_indicators = data_dict[indicator_tf]
        if len(df_indicators) < 2: 
            logger.warning(f"{log_pref} Pas assez de données d'indicateurs ({len(df_indicators)} lignes) pour generate_order_request.")
            return None

        # latest_indicators = df_indicators.iloc[-1]
        # La logique de génération d'ordre irait ici.
        # Pour le template, nous retournons None.

        logger.debug(f"{log_pref} Aucune action d'ordre générée pour le live trading sur la dernière kline.")
        return None

