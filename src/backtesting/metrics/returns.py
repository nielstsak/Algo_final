"""
Module contenant les métriques de rendement pour les backtests.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import vectorbt as vbt
from pandas.tseries.frequencies import to_offset

from src.backtesting.metrics.base import BaseMetric


class ReturnMetrics(BaseMetric):
    """
    Calcule les métriques liées aux rendements d'un portfolio.
    """
    
    @property
    def category(self) -> str:
        return "returns"
    
    def calculate(self, portfolio: vbt.Portfolio, benchmark_returns: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """
        Calcule les métriques de rendement pour un portfolio donné.
        
        Args:
            portfolio: Le portfolio VectorBT à analyser
            benchmark_returns: Série de rendements du benchmark (optionnel)
            **kwargs: Arguments supplémentaires
            
        Returns:
            Dict[str, Any]: Dictionnaire des métriques de rendement
        """
        metrics = {}
        
        # Rendement total
        metrics[self._format_metric_name('total_return')] = self._safe_calculate(
            lambda: portfolio.total_return(), 'total_return'
        )
        
        # Rendement total en pourcentage
        metrics[self._format_metric_name('total_return_pct')] = self._safe_calculate(
            lambda: portfolio.total_return() * 100, 'total_return_pct'
        )
        
        # Rendement annualisé
        metrics[self._format_metric_name('annual_return')] = self._safe_calculate(
            lambda: portfolio.annualized_return(), 'annual_return'
        )
        
        # Rendement journalier moyen
        metrics[self._format_metric_name('daily_mean_return')] = self._safe_calculate(
            lambda: self._resample_returns(portfolio.returns(), '1D').mean(), 'daily_mean_return'
        )
        
        # Rendement journalier médian
        metrics[self._format_metric_name('daily_median_return')] = self._safe_calculate(
            lambda: self._resample_returns(portfolio.returns(), '1D').median(), 'daily_median_return'
        )
        
        # Meilleur jour
        metrics[self._format_metric_name('best_day')] = self._safe_calculate(
            lambda: self._resample_returns(portfolio.returns(), '1D').max(), 'best_day'
        )
        
        # Pire jour
        metrics[self._format_metric_name('worst_day')] = self._safe_calculate(
            lambda: self._resample_returns(portfolio.returns(), '1D').min(), 'worst_day'
        )
        
        # Périodes positives et négatives
        daily_returns = self._resample_returns(portfolio.returns(), '1D')
        metrics[self._format_metric_name('positive_days')] = self._safe_calculate(
            lambda: (daily_returns > 0).sum(), 'positive_days'
        )
        metrics[self._format_metric_name('negative_days')] = self._safe_calculate(
            lambda: (daily_returns < 0).sum(), 'negative_days'
        )
        
        # Ratio de jours positifs
        metrics[self._format_metric_name('win_days_ratio')] = self._safe_calculate(
            lambda: (daily_returns > 0).sum() / len(daily_returns), 'win_days_ratio'
        )
        
        # Alpha et Beta si le benchmark est fourni
        if benchmark_returns is not None and not benchmark_returns.empty:
            metrics.update(self._calculate_benchmark_metrics(portfolio, benchmark_returns))
        
        return metrics
    
    def _resample_returns(self, returns: pd.Series, freq: str) -> pd.Series:
        """
        Rééchantillonne les rendements à une fréquence donnée.
        
        Args:
            returns: Série des rendements
            freq: Fréquence cible (ex: '1D', '1M', etc.)
            
        Returns:
            pd.Series: Rendements rééchantillonnés
        """
        if returns.empty:
            return pd.Series(dtype=float)
        
        # Vérifier si les rendements ont un index de temps approprié
        if not isinstance(returns.index, pd.DatetimeIndex):
            return returns
        
        # Rééchantillonnage avec la méthode appropriée
        resampled = returns.resample(freq).apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )
        
        return resampled
    
    def _calculate_benchmark_metrics(self, portfolio: vbt.Portfolio, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """
        Calcule les métriques relatives au benchmark.
        
        Args:
            portfolio: Le portfolio VectorBT
            benchmark_returns: Série de rendements du benchmark
            
        Returns:
            Dict[str, Any]: Métriques relatives au benchmark
        """
        metrics = {}
        
        try:
            # Aligner les deux séries de rendements
            port_returns = portfolio.returns()
            
            aligned_returns = pd.DataFrame({
                'portfolio': port_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_returns) < 2:
                return {}
            
            # Calculer beta et alpha
            cov_matrix = aligned_returns.cov()
            beta = cov_matrix.loc['portfolio', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            metrics[self._format_metric_name('beta')] = beta
            
            # Alpha annualisé
            freq = self._infer_frequency(port_returns)
            periods_per_year = self._estimate_periods_per_year(freq)
            
            portfolio_annualized = (1 + aligned_returns['portfolio'].mean()) ** periods_per_year - 1
            benchmark_annualized = (1 + aligned_returns['benchmark'].mean()) ** periods_per_year - 1
            
            alpha = portfolio_annualized - (0.01 + beta * benchmark_annualized)  # 1% risk-free rate est une approximation
            metrics[self._format_metric_name('alpha')] = alpha
            
            # Information ratio
            tracking_error = (aligned_returns['portfolio'] - aligned_returns['benchmark']).std() * np.sqrt(periods_per_year)
            if tracking_error > 0:
                info_ratio = (portfolio_annualized - benchmark_annualized) / tracking_error
                metrics[self._format_metric_name('information_ratio')] = info_ratio
            
        except Exception as e:
            print(f"Erreur lors du calcul des métriques relatives au benchmark: {e}")
        
        return metrics
    
    def _infer_frequency(self, time_series: pd.Series) -> str:
        """
        Infère la fréquence d'une série temporelle.
        
        Args:
            time_series: La série temporelle
            
        Returns:
            str: La fréquence inférée
        """
        if time_series.empty or len(time_series) < 2:
            return '1D'  # Fréquence par défaut
            
        if not isinstance(time_series.index, pd.DatetimeIndex):
            return '1D'
            
        # Calculer les différences de temps
        deltas = time_series.index[1:] - time_series.index[:-1]
        median_delta = pd.Timedelta(np.median(deltas))
        
        # Déterminer la fréquence approximative
        if median_delta <= pd.Timedelta(minutes=5):
            return '1min'
        elif median_delta <= pd.Timedelta(hours=1):
            return '1H'
        elif median_delta <= pd.Timedelta(days=1):
            return '1D'
        elif median_delta <= pd.Timedelta(days=31):
            return '1M'
        else:
            return '1Y'
    
    def _estimate_periods_per_year(self, freq: str) -> float:
        """
        Estime le nombre de périodes par an pour une fréquence donnée.
        
        Args:
            freq: La fréquence (ex: '1D', '1H', etc.)
            
        Returns:
            float: Nombre estimé de périodes par an
        """
        try:
            delta = to_offset(freq)
            one_year = pd.Timedelta(days=365.25)
            return float(one_year / delta)
        except:
            # Valeurs par défaut pour les fréquences communes
            if freq == '1min':
                return 525600  # 365.25 * 24 * 60
            elif freq == '1H':
                return 8766    # 365.25 * 24
            elif freq == '1D':
                return 365.25
            elif freq == '1W':
                return 52.18   # 365.25 / 7
            elif freq == '1M':
                return 12
            elif freq == '1Q':
                return 4
            else:
                return 252     # Jours de trading typiques