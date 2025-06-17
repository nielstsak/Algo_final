"""
Tests unitaires pour le pattern Registry des stratégies.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.strategies.base import BaseStrategy
from src.strategies.registry import StrategyRegistry
from src.core.exceptions import StrategyLoadError


class TestStrategyRegistry:
    """Tests pour la classe StrategyRegistry."""
    
    def setup_method(self):
        """Réinitialise le registre pour chaque test."""
        # Sauvegarde l'état actuel du registre
        self._original_strategies = StrategyRegistry._strategies.copy()
        # Efface le registre pour le test
        StrategyRegistry._strategies.clear()
    
    def teardown_method(self):
        """Restaure l'état du registre après chaque test."""
        StrategyRegistry._strategies.clear()
        StrategyRegistry._strategies.update(self._original_strategies)
    
    def test_register_decorator(self):
        """Teste l'enregistrement d'une stratégie via le décorateur."""
        # Définition d'une stratégie de test
        @StrategyRegistry.register("test_strategy")
        class TestStrategy(BaseStrategy):
            name = "TestStrategy"
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, indicators_df):
                return indicators_df
        
        # Vérification que la stratégie est bien enregistrée
        assert "test_strategy" in StrategyRegistry._strategies
        assert StrategyRegistry._strategies["test_strategy"] == TestStrategy
    
    def test_register_with_default_name(self):
        """Teste l'enregistrement avec le nom par défaut (nom de la classe)."""
        @StrategyRegistry.register()
        class AnotherTestStrategy(BaseStrategy):
            name = "AnotherTestStrategy"
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, indicators_df):
                return indicators_df
        
        assert "AnotherTestStrategy" in StrategyRegistry._strategies
        assert StrategyRegistry._strategies["AnotherTestStrategy"] == AnotherTestStrategy
    
    def test_register_invalid_class(self):
        """Teste l'enregistrement d'une classe non valide."""
        with pytest.raises(TypeError):
            @StrategyRegistry.register()
            class InvalidStrategy:  # N'hérite pas de BaseStrategy
                pass
    
    def test_get_strategy_class(self):
        """Teste la récupération d'une classe de stratégie."""
        # Enregistrement d'une stratégie
        @StrategyRegistry.register("get_test")
        class GetTestStrategy(BaseStrategy):
            name = "GetTestStrategy"
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, indicators_df):
                return indicators_df
        
        # Récupération de la stratégie
        strategy_class = StrategyRegistry.get_strategy_class("get_test")
        assert strategy_class == GetTestStrategy
    
    def test_get_nonexistent_strategy_class(self):
        """Teste la récupération d'une stratégie inexistante."""
        with pytest.raises(StrategyLoadError):
            StrategyRegistry.get_strategy_class("nonexistent_strategy")
    
    def test_create_strategy(self):
        """Teste la création d'une instance de stratégie."""
        # Enregistrement d'une stratégie avec des paramètres attendus
        @StrategyRegistry.register("create_test")
        class CreateTestStrategy(BaseStrategy):
            name = "CreateTestStrategy"
            
            def __init__(self, pair_symbol="BTC/USDT", **kwargs):
                self.pair_symbol = pair_symbol
                super().__init__(**kwargs)
            
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, indicators_df):
                return indicators_df
        
        # Création d'une instance
        instance = StrategyRegistry.create("create_test", pair_symbol="ETH/USDT")
        assert isinstance(instance, CreateTestStrategy)
        assert instance.pair_symbol == "ETH/USDT"
    
    def test_create_nonexistent_strategy(self):
        """Teste la création d'une stratégie inexistante."""
        with pytest.raises(StrategyLoadError):
            StrategyRegistry.create("nonexistent_strategy")
    
    def test_list_strategies(self):
        """Teste la récupération de la liste des stratégies disponibles."""
        # Enregistrement de quelques stratégies
        @StrategyRegistry.register("list_test1")
        class ListTest1Strategy(BaseStrategy):
            name = "ListTest1Strategy"
            __doc__ = "Stratégie de test 1"
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, indicators_df):
                return indicators_df
        
        @StrategyRegistry.register("list_test2")
        class ListTest2Strategy(BaseStrategy):
            name = "ListTest2Strategy"
            __doc__ = "Stratégie de test 2"
            def calculate_indicators(self, data):
                return data
            
            def generate_signals(self, indicators_df):
                return indicators_df
        
        # Récupération de la liste
        strategies_list = StrategyRegistry.list_strategies()
        
        # Vérification que les stratégies sont bien listées
        assert "list_test1" in strategies_list
        assert "list_test2" in strategies_list
        assert strategies_list["list_test1"]["class"] == "ListTest1Strategy"
        assert strategies_list["list_test2"]["class"] == "ListTest2Strategy"
        assert "Stratégie de test 1" in strategies_list["list_test1"]["description"]