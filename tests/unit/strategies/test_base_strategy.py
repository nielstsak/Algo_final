"""
Tests unitaires pour la classe BaseStrategy.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.strategies.base import BaseStrategy
from src.core.exceptions import IndicatorCalculationError, SignalGenerationError


class SimpleTestStrategy(BaseStrategy):
    """Stratégie de test simple pour vérifier les méthodes de BaseStrategy."""
    
    def __init__(self, pair_symbol, strategy_name=None, **kwargs):
        super().__init__(pair_symbol, strategy_name or "SimpleTestStrategy", **kwargs)
        self.sma_fast = kwargs.get('sma_fast', 10)
        self.sma_slow = kwargs.get('sma_slow', 20)
    
    def calculate_indicators(self, data):
        """Calcule des indicateurs de test simples (SMA)."""
        try:
            df = data.copy()
            if hasattr(data, 'get_view'):  # EnrichedDataFrame
                df = data.get_view('1h')
            
            df['sma_fast'] = df['close'].rolling(window=self.sma_fast).mean()
            df['sma_slow'] = df['close'].rolling(window=self.sma_slow).mean()
            return df
        except Exception as e:
            raise IndicatorCalculationError(f"Erreur lors du calcul des indicateurs: {e}")
    
    def generate_signals(self, indicators_df):
        """Génère des signaux basés sur le croisement de deux moyennes mobiles."""
        try:
            signals = pd.DataFrame(index=indicators_df.index)
            signals['buy'] = (indicators_df['sma_fast'] > indicators_df['sma_slow']) & (
                indicators_df['sma_fast'].shift(1) <= indicators_df['sma_slow'].shift(1))
            signals['sell'] = (indicators_df['sma_fast'] < indicators_df['sma_slow']) & (
                indicators_df['sma_fast'].shift(1) >= indicators_df['sma_slow'].shift(1))
            return signals
        except Exception as e:
            raise SignalGenerationError(f"Erreur lors de la génération des signaux: {e}")


class TestBaseStrategy:
    """Tests pour la classe BaseStrategy."""
    
    def test_initialization(self):
        """Vérifie que l'initialisation de BaseStrategy fonctionne correctement."""
        strategy = SimpleTestStrategy(pair_symbol="BTC/USDT", sma_fast=5, sma_slow=15)
        
        assert strategy.pair_symbol == "BTC/USDT"
        assert strategy.strategy_name == "SimpleTestStrategy"
        assert strategy.sma_fast == 5
        assert strategy.sma_slow == 15
        assert strategy.required_timeframes == []
    
    def test_calculate_indicators(self, sample_ohlcv_data):
        """Vérifie que le calcul d'indicateurs fonctionne correctement."""
        strategy = SimpleTestStrategy(pair_symbol="ETH/USDC", sma_fast=5, sma_slow=10)
        
        result_df = strategy.calculate_indicators(sample_ohlcv_data)
        
        assert 'sma_fast' in result_df.columns
        assert 'sma_slow' in result_df.columns
        # Les premières valeurs doivent être NaN à cause de la fenêtre de la moyenne mobile
        assert pd.isna(result_df['sma_fast'].iloc[3])
        assert not pd.isna(result_df['sma_fast'].iloc[5])
        assert pd.isna(result_df['sma_slow'].iloc[8])
        assert not pd.isna(result_df['sma_slow'].iloc[10])
    
    def test_generate_signals(self, sample_ohlcv_data):
        """Vérifie que la génération de signaux fonctionne correctement."""
        strategy = SimpleTestStrategy(pair_symbol="BNB/USDT", sma_fast=5, sma_slow=10)
        
        indicators_df = strategy.calculate_indicators(sample_ohlcv_data)
        signals_df = strategy.generate_signals(indicators_df)
        
        assert 'buy' in signals_df.columns
        assert 'sell' in signals_df.columns
        # Vérifie que les signaux sont des booléens
        assert signals_df['buy'].dtype == bool
        assert signals_df['sell'].dtype == bool
    
    def test_indicator_calculation_error(self):
        """Vérifie que les erreurs de calcul d'indicateurs sont correctement gérées."""
        strategy = SimpleTestStrategy(pair_symbol="SOL/USDC")
        
        # Créer des données qui provoqueront une erreur
        bad_data = pd.DataFrame({
            'not_close': [1, 2, 3]  # Pas de colonne 'close'
        })
        
        with pytest.raises(IndicatorCalculationError):
            strategy.calculate_indicators(bad_data)
    
    def test_signal_generation_error(self):
        """Vérifie que les erreurs de génération de signaux sont correctement gérées."""
        strategy = SimpleTestStrategy(pair_symbol="AVAX/USDT")
        
        # Créer des indicateurs incomplets qui provoqueront une erreur
        bad_indicators = pd.DataFrame({
            'sma_fast': [1, 2, 3]  # Manque 'sma_slow'
        })
        
        with pytest.raises(SignalGenerationError):
            strategy.generate_signals(bad_indicators)
    
    def test_get_param(self):
        """Vérifie que get_param fonctionne correctement."""
        strategy = SimpleTestStrategy(
            pair_symbol="DOT/USDC", 
            sma_fast=7, 
            custom_param="test_value"
        )
        
        assert strategy.get_param('sma_fast') == 7
        assert strategy.get_param('sma_slow') == 20  # valeur par défaut
        assert strategy.get_param('custom_param') == "test_value"
        assert strategy.get_param('non_existent', 'default') == 'default'