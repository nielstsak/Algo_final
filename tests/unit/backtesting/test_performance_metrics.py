"""
Tests unitaires pour les métriques de performance du backtesting.
"""
import pytest
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.backtesting.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Tests pour la classe PerformanceMetrics."""
    
    @pytest.fixture
    def mock_portfolio(self):
        """Crée un mock de portfolio VectorBT avec des données de test."""
        # Créer une série temporelle pour les données de prix
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        close = 100 + np.random.randn(100).cumsum()
        
        # Créer des signaux d'entrée et de sortie
        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)
        
        # Simuler quelques trades
        entries.iloc[10] = True  # Achat
        exits.iloc[20] = True    # Vente
        entries.iloc[40] = True  # Achat
        exits.iloc[60] = True    # Vente
        entries.iloc[70] = True  # Achat
        exits.iloc[90] = True    # Vente
        
        # Créer un portfolio VectorBT
        portfolio = vbt.Portfolio.from_signals(
            close=pd.Series(close, index=dates),
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            freq='1h'
        )
        
        return portfolio
    
    def test_calculate_returns_metrics(self, mock_portfolio):
        """Teste le calcul des métriques de rendement."""
        metrics = PerformanceMetrics.calculate_returns_metrics(mock_portfolio)
        
        # Vérifier la présence des métriques clés
        assert 'returns_total_return' in metrics
        assert 'returns_total_return_pct' in metrics
        assert 'returns_annual_return' in metrics
        assert isinstance(metrics['returns_total_return'], float)
        assert isinstance(metrics['returns_total_return_pct'], float)
        
        # Les rendements doivent être dans des plages raisonnables
        assert -1.0 <= metrics['returns_total_return_pct'] <= 10.0
    
    def test_calculate_drawdown_metrics(self, mock_portfolio):
        """Teste le calcul des métriques de drawdown."""
        metrics = PerformanceMetrics.calculate_drawdown_metrics(mock_portfolio)
        
        # Vérifier la présence des métriques clés
        assert 'drawdown_max_drawdown_pct' in metrics
        assert 'drawdown_avg_drawdown_pct' in metrics
        assert 'drawdown_max_drawdown_duration' in metrics
        
        # Les drawdowns doivent être négatifs ou nuls
        assert 0.0 <= metrics['drawdown_max_drawdown_pct'] <= 1.0
        assert 0.0 <= metrics['drawdown_avg_drawdown_pct'] <= 1.0
    
    def test_calculate_risk_metrics(self, mock_portfolio):
        """Teste le calcul des métriques de risque."""
        metrics = PerformanceMetrics.calculate_risk_metrics(mock_portfolio)
        
        # Vérifier la présence des métriques clés
        assert 'risk_volatility_annual' in metrics
        assert 'risk_ratios_sharpe_ratio' in metrics
        assert 'risk_ratios_sortino_ratio' in metrics
        
        # Les volatilités doivent être positives
        assert metrics['risk_volatility_annual'] >= 0
    
    def test_calculate_trade_metrics(self, mock_portfolio):
        """Teste le calcul des métriques de trade."""
        metrics = PerformanceMetrics.calculate_trade_metrics(mock_portfolio)
        
        # Vérifier la présence des métriques clés
        assert 'trade_stats_total_trades' in metrics
        assert 'trade_stats_win_rate' in metrics
        assert 'trade_stats_profit_factor' in metrics
        assert 'trade_stats_avg_winning_trade' in metrics
        
        # Le nombre de trades doit correspondre à ce que nous avons configuré
        assert metrics['trade_stats_total_trades'] == 3
        assert 0.0 <= metrics['trade_stats_win_rate'] <= 1.0
    
    def test_calculate_all_metrics(self, mock_portfolio):
        """Teste le calcul de toutes les métriques combinées."""
        all_metrics = PerformanceMetrics.calculate_all_metrics(mock_portfolio)
        
        # Vérifier que nous avons un bon nombre de métriques
        assert len(all_metrics) > 20
        
        # Vérifier la présence de métriques de différentes catégories
        assert any(k.startswith('returns_') for k in all_metrics.keys())
        assert any(k.startswith('drawdown_') for k in all_metrics.keys())
        assert any(k.startswith('risk_') for k in all_metrics.keys())
        assert any(k.startswith('trade_stats_') for k in all_metrics.keys())
    
    def test_calculate_all_metrics_with_benchmark(self, mock_portfolio):
        """Teste le calcul des métriques avec un benchmark."""
        # Créer des rendements benchmark
        benchmark_returns = pd.Series(
            np.random.normal(0.0001, 0.01, len(mock_portfolio.close)), 
            index=mock_portfolio.close.index
        )
        
        all_metrics = PerformanceMetrics.calculate_all_metrics(mock_portfolio, benchmark_returns)
        
        # Vérifier les métriques spécifiques au benchmark
        assert 'risk_ratios_beta' in all_metrics
        assert 'risk_ratios_alpha' in all_metrics
    
    def test_metrics_to_dataframe(self):
        """Teste la conversion des métriques en DataFrame."""
        test_metrics = {
            'returns_total_return_pct': 0.15,
            'drawdown_max_drawdown_pct': 0.05,
            'risk_ratios_sharpe_ratio': 1.8,
            'trade_stats_win_rate': 0.6
        }
        
        df = PerformanceMetrics.metrics_to_dataframe(test_metrics)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(test_metrics)
        assert 'Category' in df.columns
        assert 'Value' in df.columns
        
        # Vérifier la catégorisation correcte
        assert 'Returns' in df['Category'].values
        assert 'Risk' in df['Category'].values