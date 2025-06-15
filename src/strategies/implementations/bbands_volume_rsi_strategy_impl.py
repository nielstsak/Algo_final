from typing import Dict, Any, Optional, Union
import pandas as pd
from loguru import logger
from pydantic import Field, model_validator

from src.strategies.base_strategy import BaseStrategy
from src.core.exceptions import SignalGenerationError
from src.strategies.params import BaseFixedParams, BaseOptimizableParams
from src.data.enriched_dataframe import EnrichedDataFrame
# Importe le module refactorisé qui contient maintenant des fonctions.
from src.strategies import technical_indicators as ti

# ==============================================================================
# Définition des paramètres de la stratégie avec Pydantic pour la validation
# et la clarté.
# ==============================================================================

class BbandsVolumeRsiFixedParams(BaseFixedParams):
    """ Paramètres fixes de la stratégie, non soumis à l'optimisation. """
    indicator_frequency: str = Field(default='1h', description="Fréquence de temps pour le calcul des indicateurs (ex: '1h', '4h').")

class BbandsVolumeRsiOptimizableParams(BaseOptimizableParams):
    """ Paramètres optimisables de la stratégie. """
    bb_period: int = Field(default=20, gt=0, description="Période pour les Bandes de Bollinger.")
    bb_std_dev: float = Field(default=2.0, gt=0, description="Écart-type pour les Bandes de Bollinger.")
    rsi_period: int = Field(default=14, gt=0, description="Période pour le RSI.")
    rsi_overbought: int = Field(default=70, gt=0, lt=100, description="Seuil de surachat du RSI.")
    rsi_oversold: int = Field(default=30, gt=0, lt=100, description="Seuil de survente du RSI.")
    volume_factor: float = Field(default=1.5, ge=0, description="Facteur de multiplication pour le volume moyen de confirmation.")
    volume_period: int = Field(default=20, gt=0, description="Période pour la SMA du volume.")
    position_sizing_pct_capital: float = Field(default=0.01, gt=0, lt=1, description="Pourcentage du capital à risquer par transaction.")

    @model_validator(mode='after')
    def check_rsi_logic(self) -> 'BbandsVolumeRsiOptimizableParams':
        """ Valide que le seuil de survente est bien inférieur à celui de surachat. """
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("La logique du RSI est invalide: rsi_oversold doit être inférieur à rsi_overbought.")
        return self

# ==============================================================================
# Implémentation de la classe de stratégie
# ==============================================================================

class BbandsVolumeRsiStrategy(BaseStrategy):
    """
    Stratégie de trading qui combine les Bandes de Bollinger pour la volatilité,
    le RSI pour le momentum, et le volume comme confirmation de la force du mouvement.
    """
    name: str = "BbandsVolumeRsiStrategy"
    version: str = "2.5.0"
    description: str = "Stratégie combinant Bandes de Bollinger, RSI et confirmation par volume."

    # Association des modèles de paramètres à la classe.
    fixed_params_model = BbandsVolumeRsiFixedParams
    optimizable_params_model = BbandsVolumeRsiOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)

        # Définition des besoins de la stratégie à partir des paramètres.
        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = max(
            self.get_param('bb_period'),
            self.get_param('rsi_period'),
            self.get_param('volume_period')
        ) + 1 # +1 pour la marge de calcul
        
        # Noms de colonnes standardisés utilisés dans la stratégie.
        self.bb_lower_col, self.bb_middle_col, self.bb_upper_col = "BBL", "BBM", "BBU"
        self.rsi_col = "RSI"
        self.volume_sma_col = "volume_sma"
        
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        """
        Calcule tous les indicateurs nécessaires à la stratégie.
        """
        log_pref = self.strategy_name_log_prefix
        df = self._prepare_indicator_data(data)
        freq = self.get_param('indicator_frequency')
        
        # Détermine les noms des colonnes sources (close et volume).
        close_col = f"K_{freq}_close" if f"K_{freq}_close" in df.columns else "close"
        volume_col = f"K_{freq}_volume" if f"K_{freq}_volume" in df.columns else "volume"
        
        if close_col not in df.columns or volume_col not in df.columns:
            raise SignalGenerationError(f"Colonnes sources '{close_col}' ou '{volume_col}' sont manquantes.", self.name)

        # Utilisation des fonctions pures du module 'technical_indicators'.
        bbands = ti.calculate_bollinger_bands(df[close_col], self.get_param('bb_period'), self.get_param('bb_std_dev'))
        rsi = ti.calculate_rsi(df[close_col], self.get_param('rsi_period'))
        volume_sma = ti.calculate_sma(df[volume_col], self.get_param('volume_period'))
        volume_sma.name = self.volume_sma_col # Assigne le nom standard à la série de SMA du volume.

        # Concaténation de tous les indicateurs calculés en un seul DataFrame.
        indicators_df = pd.concat([bbands, rsi, volume_sma], axis=1)
        
        # Jointure des indicateurs avec le DataFrame original.
        final_df = df.join(indicators_df)

        # Ajout des colonnes 'close' et 'volume' sans préfixe pour un accès facile dans 'generate_signals'.
        if 'close' not in final_df.columns: final_df['close'] = final_df[close_col]
        if 'volume' not in final_df.columns: final_df['volume'] = final_df[volume_col]
            
        self._indicators_cache = final_df
        logger.info(f"{log_pref} Indicateurs calculés avec succès.")
        return final_df

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les signaux d'entrée et de sortie à partir des indicateurs calculés.
        """
        if indicators_df.empty:
            return pd.DataFrame()
        df = indicators_df.copy()

        # Vérification de la présence de toutes les colonnes requises.
        required_cols = [self.bb_lower_col, self.bb_upper_col, self.bb_middle_col, self.rsi_col, self.volume_sma_col, 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise SignalGenerationError(f"Colonnes requises manquantes pour la génération de signaux: {missing_cols}", strategy_name=self.name)

        # --- CORRECTION: Utilisation de self.get_param() au lieu de self.params_obj ---
        # Récupération des paramètres pour la lisibilité.
        rsi_overbought = self.get_param('rsi_overbought')
        rsi_oversold = self.get_param('rsi_oversold')
        volume_factor = self.get_param('volume_factor')
        
        # --- Définition des conditions logiques ---
        price_crosses_upper = (df['close'] > df[self.bb_upper_col]) & (df['close'].shift(1) <= df[self.bb_upper_col].shift(1))
        price_crosses_lower = (df['close'] < df[self.bb_lower_col]) & (df['close'].shift(1) >= df[self.bb_lower_col].shift(1))
        
        rsi_is_overbought = df[self.rsi_col] > rsi_overbought
        rsi_is_oversold = df[self.rsi_col] < rsi_oversold
        volume_confirms = df['volume'] > (df[self.volume_sma_col] * volume_factor)

        # --- Combinaison des conditions pour les signaux d'entrée ---
        entry_long_signal = price_crosses_lower & rsi_is_oversold & volume_confirms
        entry_short_signal = price_crosses_upper & rsi_is_overbought & volume_confirms
        
        # --- Définition des signaux de sortie ---
        # Sortie de position longue lorsque le prix croise la moyenne mobile centrale par le haut.
        exit_long_signal = (df['close'] > df[self.bb_middle_col]) & (df['close'].shift(1) <= df[self.bb_middle_col].shift(1))
        # Sortie de position courte lorsque le prix croise la moyenne mobile centrale par le bas.
        exit_short_signal = (df['close'] < df[self.bb_middle_col]) & (df['close'].shift(1) >= df[self.bb_middle_col].shift(1))
        
        # Création du DataFrame de signaux.
        signals = pd.DataFrame(index=df.index)
        signals['entry_long'] = entry_long_signal
        signals['entry_short'] = entry_short_signal
        signals['exit_long'] = exit_long_signal
        signals['exit_short'] = exit_short_signal
        
        self._signals = signals
        return self._signals.copy()
