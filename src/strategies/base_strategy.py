# src/strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.core.constants import Trading, Kline, DataFrameCols
from src.core.exceptions import (
    StrategyError,
    InvalidStrategyParamsError,
    SignalGenerationError,
    IndicatorCalculationError
)


class BaseStrategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading.
    Fournit une interface standardisée pour le développement de stratégies.
    """
    
    # Attributs de classe pour identifier la stratégie
    name: str = "BaseStrategy"
    version: str = "1.0.0"
    description: str = "Abstract base strategy class"
    
    # Paramètres par défaut (à surcharger dans les sous-classes)
    default_params: Dict[str, Any] = {}
    
    # Intervalles de temps requis par la stratégie
    required_timeframes: List[str] = [Kline.INTERVAL_1MINUTE]
    
    # Nombre minimum de périodes requises pour le calcul
    min_required_periods: int = 100
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialise la stratégie avec les paramètres fournis.
        
        Args:
            params: Dictionnaire des paramètres de la stratégie
            **kwargs: Paramètres additionnels passés directement
        """
        # Fusionner les paramètres
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        self.params.update(kwargs)
        
        # Valider les paramètres
        self.validate_params()
        
        # Initialiser les caches
        self._indicators_cache: Dict[str, pd.DataFrame] = {}
        self._signals_cache: Optional[pd.DataFrame] = None
        
        logger.debug(f"Initialized {self.name} with params: {self.params}")
        
    @abstractmethod
    def validate_params(self) -> None:
        """
        Valide les paramètres de la stratégie.
        Doit lever InvalidStrategyParamsError si invalides.
        """
        pass
        
    @abstractmethod
    def calculate_indicators(
        self,
        klines: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calcule les indicateurs techniques nécessaires.
        
        Args:
            klines: Dict avec les klines pour chaque timeframe
                   {timeframe: DataFrame avec colonnes OHLCV}
                   
        Returns:
            Dict avec les indicateurs calculés par timeframe
            
        Raises:
            IndicatorCalculationError: Si erreur dans le calcul
        """
        pass
        
    @abstractmethod
    def generate_signals(
        self,
        indicators: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Génère les signaux de trading basés sur les indicateurs.
        
        Args:
            indicators: Dict avec les indicateurs par timeframe
            
        Returns:
            DataFrame avec colonnes:
            - signal: 1 (long), -1 (short), 0 (neutre)
            - entry_price: Prix d'entrée suggéré
            - stop_loss: Niveau de stop loss
            - take_profit: Niveau de take profit
            - confidence: Score de confiance [0-1]
            
        Raises:
            SignalGenerationError: Si erreur dans la génération
        """
        pass
        
    def execute(
        self,
        klines: Dict[str, pd.DataFrame],
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Exécute la stratégie complète et retourne les signaux formatés.
        
        Args:
            klines: Dict avec les klines pour chaque timeframe
            current_position: Position actuelle (optionnel)
            
        Returns:
            Liste de signaux au format standardisé
        """
        try:
            # Vérifier les données d'entrée
            self._validate_klines(klines)
            
            # Calculer les indicateurs
            indicators = self.calculate_indicators(klines)
            
            # Générer les signaux
            signals_df = self.generate_signals(indicators)
            
            # Formater les signaux
            formatted_signals = self._format_signals(
                signals_df,
                klines,
                current_position
            )
            
            return formatted_signals
            
        except Exception as e:
            logger.error(f"Error executing strategy {self.name}: {e}")
            raise StrategyError(
                f"Strategy execution failed: {e}",
                strategy_name=self.name,
                original_exception=e
            )
            
    def _validate_klines(self, klines: Dict[str, pd.DataFrame]) -> None:
        """
        Valide les données klines d'entrée.
        
        Args:
            klines: Dict avec les klines par timeframe
            
        Raises:
            StrategyError: Si données invalides
        """
        # Vérifier que tous les timeframes requis sont présents
        for tf in self.required_timeframes:
            if tf not in klines:
                raise StrategyError(
                    f"Missing required timeframe: {tf}",
                    strategy_name=self.name
                )
                
        # Vérifier chaque DataFrame
        for tf, df in klines.items():
            if df is None or df.empty:
                raise StrategyError(
                    f"Empty data for timeframe: {tf}",
                    strategy_name=self.name
                )
                
            # Vérifier les colonnes requises
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'base_asset_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise StrategyError(
                    f"Missing required columns in {tf}: {missing_cols}",
                    strategy_name=self.name
                )
                
            # Vérifier le nombre minimum de périodes
            if len(df) < self.min_required_periods:
                raise StrategyError(
                    f"Insufficient data for {tf}: {len(df)} < {self.min_required_periods}",
                    strategy_name=self.name
                )
                
    def _format_signals(
        self,
        signals_df: pd.DataFrame,
        klines: Dict[str, pd.DataFrame],
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Formate les signaux au format standardisé.
        
        Args:
            signals_df: DataFrame avec les signaux bruts
            klines: Dict avec les klines originales
            current_position: Position actuelle
            
        Returns:
            Liste de signaux formatés
        """
        formatted_signals = []
        
        # Obtenir le DataFrame principal (premier timeframe requis)
        main_tf = self.required_timeframes[0]
        main_df = klines[main_tf]
        
        # Ne garder que les signaux non-neutres récents
        active_signals = signals_df[signals_df['signal'] != 0].tail(10)
        
        for idx, row in active_signals.iterrows():
            # Obtenir les prix actuels
            if idx in main_df.index:
                current_bar = main_df.loc[idx]
            else:
                # Trouver la barre la plus proche
                nearest_idx = main_df.index[main_df.index.get_indexer([idx], method='nearest')[0]]
                current_bar = main_df.loc[nearest_idx]
                
            # Créer le signal formaté
            signal = {
                'timestamp': idx,
                'strategy_name': self.name,
                'strategy_version': self.version,
                'signal_type': self._get_signal_type(row['signal']),
                'side': Trading.SIDE_BUY if row['signal'] > 0 else Trading.SIDE_SELL,
                'entry_price': row.get('entry_price', current_bar['close_price']),
                'stop_loss': row.get('stop_loss'),
                'take_profit': row.get('take_profit'),
                'confidence': row.get('confidence', 0.5),
                'metadata': {
                    'indicators': self._get_signal_metadata(idx, row),
                    'timeframe': main_tf,
                    'current_position': current_position
                }
            }
            
            # Valider et ajuster les prix si nécessaire
            signal = self._validate_signal_prices(signal, current_bar)
            
            formatted_signals.append(signal)
            
        return formatted_signals
        
    def _get_signal_type(self, signal_value: float) -> str:
        """Convertit la valeur numérique du signal en type string."""
        if signal_value > 0:
            return Trading.SIGNAL_TYPE_LONG
        elif signal_value < 0:
            return Trading.SIGNAL_TYPE_SHORT
        else:
            return Trading.SIGNAL_TYPE_NEUTRAL
            
    def _get_signal_metadata(
        self,
        timestamp: pd.Timestamp,
        signal_row: pd.Series
    ) -> Dict[str, Any]:
        """
        Extrait les métadonnées pour un signal.
        À surcharger dans les sous-classes pour ajouter des infos spécifiques.
        """
        metadata = {}
        
        # Ajouter les valeurs des indicateurs si disponibles
        if hasattr(self, '_indicators_cache'):
            for tf, indicators_df in self._indicators_cache.items():
                if timestamp in indicators_df.index:
                    tf_data = indicators_df.loc[timestamp].to_dict()
                    metadata[f'indicators_{tf}'] = tf_data
                    
        return metadata
        
    def _validate_signal_prices(
        self,
        signal: Dict[str, Any],
        current_bar: pd.Series
    ) -> Dict[str, Any]:
        """
        Valide et ajuste les prix du signal si nécessaire.
        
        Args:
            signal: Signal à valider
            current_bar: Barre de prix actuelle
            
        Returns:
            Signal avec prix validés
        """
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        # Pour un signal LONG
        if signal['signal_type'] == Trading.SIGNAL_TYPE_LONG:
            # Le SL doit être sous le prix d'entrée
            if stop_loss and stop_loss >= entry_price:
                logger.warning(f"Invalid SL for LONG: {stop_loss} >= {entry_price}")
                signal['stop_loss'] = entry_price * 0.98  # 2% par défaut
                
            # Le TP doit être au-dessus du prix d'entrée
            if take_profit and take_profit <= entry_price:
                logger.warning(f"Invalid TP for LONG: {take_profit} <= {entry_price}")
                signal['take_profit'] = entry_price * 1.02  # 2% par défaut
                
        # Pour un signal SHORT
        elif signal['signal_type'] == Trading.SIGNAL_TYPE_SHORT:
            # Le SL doit être au-dessus du prix d'entrée
            if stop_loss and stop_loss <= entry_price:
                logger.warning(f"Invalid SL for SHORT: {stop_loss} <= {entry_price}")
                signal['stop_loss'] = entry_price * 1.02  # 2% par défaut
                
            # Le TP doit être sous le prix d'entrée
            if take_profit and take_profit >= entry_price:
                logger.warning(f"Invalid TP for SHORT: {take_profit} >= {entry_price}")
                signal['take_profit'] = entry_price * 0.98  # 2% par défaut
                
        return signal
        
    def get_required_lookback(self) -> int:
        """
        Retourne le nombre de périodes historiques requises.
        
        Returns:
            Nombre de périodes
        """
        return self.min_required_periods
        
    def get_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur la stratégie.
        
        Returns:
            Dict avec les infos de la stratégie
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'required_timeframes': self.required_timeframes,
            'min_required_periods': self.min_required_periods,
            'parameters': self.params,
            'default_parameters': self.default_params
        }
        
    def reset(self) -> None:
        """Réinitialise les caches internes de la stratégie."""
        self._indicators_cache.clear()
        self._signals_cache = None
        logger.debug(f"Reset strategy {self.name} caches")
        
    def __repr__(self) -> str:
        """Représentation string de la stratégie."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', params={self.params})"