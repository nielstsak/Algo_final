"""
Module contenant les métriques de risque pour les backtests.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import vectorbt as vbt

from src.backtesting.metrics.base import BaseMetric


class RiskMetrics(BaseMetric):
    """
    Calcule les métriques liées au risque d'un portfolio.
    """
    
    @property
    def category(self) -> str:
        return "risk"
    
    def calculate(self, portfolio: vbt.Portfolio, risk_free_rate: float = 0.01, **kwargs) -> Dict[str, Any]:
        """
        Calcule les métriques de risque pour un portfolio donné.
        
        Args:
            portfolio: Le portfolio VectorBT à analyser
            risk_free_rate: Taux sans risque annualisé (défaut: 1%)
            **kwargs: Arguments supplémentaires
            
        Returns:
            Dict[str, Any]: Dictionnaire des métriques de risque
        """
        metrics = {}
        
        # Volatilité
        metrics[self._format_metric_name('volatility_annual')] = self._safe_calculate(
            lambda: portfolio.annual_volatility(), 'volatility_annual'
        )
        
        # Volatilité journalière
        daily_returns = self._resample_returns(portfolio.returns(), '1D')
        metrics[self._format_metric_name('volatility_daily')] = self._safe_calculate(
            lambda: daily_returns.std(), 'volatility_daily'
        )
        
        # Drawdowns
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio)
        metrics.update(drawdown_metrics)
        
        # Ratios de risque (Sharpe, Sortino, Calmar)
        ratio_metrics = self._calculate_risk_ratios(portfolio, risk_free_rate)
        metrics.update(ratio_metrics)
        
        # VaR (Value at Risk)
        var_metrics = self._calculate_var_metrics(portfolio)
        metrics.update(var_metrics)
        
        return metrics
    
    def _calculate_drawdown_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule les métriques liées aux drawdowns.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Métriques de drawdown
        """
        metrics = {}
        
        # Drawdown maximum
        metrics[self._format_metric_name('max_drawdown_pct')] = self._safe_calculate(
            lambda: portfolio.max_drawdown(), 'max_drawdown_pct'
        )
        
        # Période du drawdown maximum
        try:
            dd_details = self._get_max_drawdown_details(portfolio)
            if dd_details:
                start_idx, valley_idx, end_idx = dd_details
                
                # Durée du drawdown maximum
                if isinstance(portfolio.drawdown().index, pd.DatetimeIndex):
                    metrics[self._format_metric_name('max_drawdown_duration')] = portfolio.drawdown().index[end_idx] - portfolio.drawdown().index[start_idx]
                else:
                    metrics[self._format_metric_name('max_drawdown_duration')] = end_idx - start_idx
                
                # Temps de récupération
                metrics[self._format_metric_name('max_drawdown_recovery_time')] = end_idx - valley_idx
        except Exception as e:
            self._handle_calculation_error('max_drawdown_details', e)
        
        # Drawdown moyen
        metrics[self._format_metric_name('avg_drawdown_pct')] = self._safe_calculate(
            lambda: portfolio.drawdowns[portfolio.drawdowns < 0].mean(), 'avg_drawdown_pct'
        )
        
        # Nombre de drawdowns
        metrics[self._format_metric_name('drawdown_count')] = self._safe_calculate(
            lambda: len(self._find_drawdown_periods(portfolio)), 'drawdown_count'
        )
        
        return metrics
    
    def _calculate_risk_ratios(self, portfolio: vbt.Portfolio, risk_free_rate: float) -> Dict[str, Any]:
        """
        Calcule les ratios de risque/rendement.
        
        Args:
            portfolio: Le portfolio VectorBT
            risk_free_rate: Taux sans risque annualisé
            
        Returns:
            Dict[str, Any]: Ratios de risque
        """
        metrics = {}
        
        # Ratio de Sharpe (utiliser le daily_risk_free_rate)
        daily_rfr = (1 + risk_free_rate) ** (1/252) - 1
        metrics[self._format_metric_name('ratios_sharpe_ratio')] = self._safe_calculate(
            lambda: portfolio.sharpe_ratio(risk_free=daily_rfr), 'sharpe_ratio'
        )
        
        # Ratio de Sortino
        metrics[self._format_metric_name('ratios_sortino_ratio')] = self._safe_calculate(
            lambda: portfolio.sortino_ratio(risk_free=daily_rfr), 'sortino_ratio'
        )
        
        # Ratio de Calmar
        metrics[self._format_metric_name('ratios_calmar_ratio')] = self._safe_calculate(
            lambda: self._calculate_calmar_ratio(portfolio), 'calmar_ratio'
        )
        
        # Ratio d'Omega
        metrics[self._format_metric_name('ratios_omega_ratio')] = self._safe_calculate(
            lambda: self._calculate_omega_ratio(portfolio, threshold=daily_rfr), 'omega_ratio'
        )
        
        return metrics
    
    def _calculate_var_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule les métriques de Value-at-Risk (VaR).
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Métriques de VaR
        """
        metrics = {}
        
        # Resampler les rendements journaliers
        daily_returns = self._resample_returns(portfolio.returns(), '1D')
        
        # VaR 95%
        metrics[self._format_metric_name('var_95')] = self._safe_calculate(
            lambda: np.percentile(daily_returns, 5), 'var_95'
        )
        
        # VaR 99%
        metrics[self._format_metric_name('var_99')] = self._safe_calculate(
            lambda: np.percentile(daily_returns, 1), 'var_99'
        )
        
        # Conditional VaR (Expected Shortfall) 95%
        metrics[self._format_metric_name('cvar_95')] = self._safe_calculate(
            lambda: daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean(), 'cvar_95'
        )
        
        return metrics
    
    def _get_max_drawdown_details(self, portfolio: vbt.Portfolio) -> Optional[Tuple[int, int, int]]:
        """
        Trouve les détails du drawdown maximum (dates de début, creux et fin).
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Optional[Tuple[int, int, int]]: Indices (début, creux, fin) du drawdown maximum
        """
        drawdown_series = portfolio.drawdown()
        if drawdown_series.empty or drawdown_series.min() >= 0:
            return None
            
        # Trouver l'indice du drawdown maximum
        valley_idx = drawdown_series.idxmin()
        if valley_idx is None:
            return None
            
        # Trouver le début du drawdown (le dernier pic avant le creux)
        peak_idx = drawdown_series[:valley_idx][::-1].idxmax()
        if peak_idx is None:
            peak_idx = 0
            
        # Trouver la fin du drawdown (premier retour au niveau du pic ou fin de série)
        recovery_period = drawdown_series[valley_idx:]
        recovery_indices = recovery_period[recovery_period >= 0].index
        end_idx = recovery_indices[0] if len(recovery_indices) > 0 else len(drawdown_series) - 1
        
        return peak_idx, valley_idx, end_idx
    
    def _find_drawdown_periods(self, portfolio: vbt.Portfolio) -> list:
        """
        Identifie tous les périodes de drawdown distinctes.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            list: Liste des périodes de drawdown [(début, creux, fin), ...]
        """
        drawdown_series = portfolio.drawdown()
        if drawdown_series.empty:
            return []
            
        in_drawdown = False
        drawdown_periods = []
        current_peak = 0
        current_valley = 0
        
        for i, dd in enumerate(drawdown_series):
            if not in_drawdown and dd < 0:
                # Début d'un nouveau drawdown
                in_drawdown = True
                current_peak = i - 1 if i > 0 else 0
                current_valley = i
            elif in_drawdown:
                if dd < drawdown_series[current_valley]:
                    # Nouveau creux du drawdown
                    current_valley = i
                elif dd >= 0:
                    # Fin du drawdown (récupération)
                    in_drawdown = False
                    drawdown_periods.append((current_peak, current_valley, i))
        
        # Si on est toujours en drawdown à la fin
        if in_drawdown:
            drawdown_periods.append((current_peak, current_valley, len(drawdown_series) - 1))
        
        return drawdown_periods
    
    def _calculate_calmar_ratio(self, portfolio: vbt.Portfolio) -> float:
        """
        Calcule le ratio de Calmar (rendement annualisé / drawdown maximum).
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            float: Ratio de Calmar
        """
        ann_return = portfolio.annualized_return()
        max_dd = portfolio.max_drawdown()
        
        if max_dd == 0:
            return np.inf if ann_return >= 0 else -np.inf
            
        return ann_return / abs(max_dd)
    
    def _calculate_omega_ratio(self, portfolio: vbt.Portfolio, threshold: float = 0) -> float:
        """
        Calcule le ratio d'Omega (rendements supérieurs au seuil / rendements inférieurs au seuil).
        
        Args:
            portfolio: Le portfolio VectorBT
            threshold: Seuil de rendement
            
        Returns:
            float: Ratio d'Omega
        """
        returns = portfolio.returns()
        
        # Séparer les rendements supérieurs et inférieurs au seuil
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]
        
        # Calculer l'espérance des gains et des pertes
        expected_gain = returns_above.sum() / len(returns) if len(returns_above) > 0 else 0
        expected_loss = returns_below.sum() / len(returns) if len(returns_below) > 0 else 0
        
        if expected_loss == 0:
            return np.inf if expected_gain > 0 else 1.0
            
        return expected_gain / expected_loss
    
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