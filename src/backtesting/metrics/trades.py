"""
Module contenant les métriques liées aux trades individuels pour les backtests.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import vectorbt as vbt

from src.backtesting.metrics.base import BaseMetric


class TradeMetrics(BaseMetric):
    """
    Calcule les métriques liées aux trades individuels d'un portfolio.
    """
    
    @property
    def category(self) -> str:
        return "trade_stats"
    
    def calculate(self, portfolio: vbt.Portfolio, **kwargs) -> Dict[str, Any]:
        """
        Calcule les métriques de trades pour un portfolio donné.
        
        Args:
            portfolio: Le portfolio VectorBT à analyser
            **kwargs: Arguments supplémentaires
            
        Returns:
            Dict[str, Any]: Dictionnaire des métriques de trades
        """
        metrics = {}
        
        # Statistiques de base sur les trades
        metrics.update(self._calculate_basic_trade_stats(portfolio))
        
        # Statistiques sur les gains et pertes
        metrics.update(self._calculate_profit_loss_stats(portfolio))
        
        # Statistiques sur la durée des trades
        metrics.update(self._calculate_duration_stats(portfolio))
        
        # Autres métriques avancées
        metrics.update(self._calculate_advanced_trade_stats(portfolio))
        
        return metrics
    
    def _calculate_basic_trade_stats(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule les statistiques de base sur les trades.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Statistiques de base sur les trades
        """
        metrics = {}
        
        # Nombre total de trades
        metrics[self._format_metric_name('total_trades')] = self._safe_calculate(
            lambda: portfolio.trades.count(), 'total_trades'
        )
        
        # Nombre de trades gagnants et perdants
        metrics[self._format_metric_name('winning_trades')] = self._safe_calculate(
            lambda: portfolio.trades.win_count(), 'winning_trades'
        )
        metrics[self._format_metric_name('losing_trades')] = self._safe_calculate(
            lambda: portfolio.trades.loss_count(), 'losing_trades'
        )
        
        # Taux de réussite (win rate)
        metrics[self._format_metric_name('win_rate')] = self._safe_calculate(
            lambda: portfolio.trades.win_rate() if portfolio.trades.count() > 0 else 0, 'win_rate'
        )
        
        # Nombre de trades par type (entrée long, entrée short, sortie)
        metrics[self._format_metric_name('entry_trades')] = self._safe_calculate(
            lambda: portfolio.trades.entry_count(), 'entry_trades'
        )
        metrics[self._format_metric_name('exit_trades')] = self._safe_calculate(
            lambda: portfolio.trades.exit_count(), 'exit_trades'
        )
        metrics[self._format_metric_name('long_trades')] = self._safe_calculate(
            lambda: portfolio.trades.long_count(), 'long_trades'
        )
        metrics[self._format_metric_name('short_trades')] = self._safe_calculate(
            lambda: portfolio.trades.short_count(), 'short_trades'
        )
        
        return metrics
    
    def _calculate_profit_loss_stats(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule les statistiques sur les gains et pertes des trades.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Statistiques sur les gains et pertes
        """
        metrics = {}
        
        # Profit net total
        metrics[self._format_metric_name('net_profit')] = self._safe_calculate(
            lambda: portfolio.trades.profit.sum(), 'net_profit'
        )
        
        # Profit moyen par trade
        metrics[self._format_metric_name('avg_profit')] = self._safe_calculate(
            lambda: portfolio.trades.profit.mean() if portfolio.trades.count() > 0 else 0, 'avg_profit'
        )
        
        # Profit médian par trade
        metrics[self._format_metric_name('median_profit')] = self._safe_calculate(
            lambda: portfolio.trades.profit.median() if portfolio.trades.count() > 0 else 0, 'median_profit'
        )
        
        # Profit moyen des trades gagnants
        metrics[self._format_metric_name('avg_winning_trade')] = self._safe_calculate(
            lambda: portfolio.trades.winning.profit.mean() if portfolio.trades.win_count() > 0 else 0, 'avg_winning_trade'
        )
        
        # Perte moyenne des trades perdants
        metrics[self._format_metric_name('avg_losing_trade')] = self._safe_calculate(
            lambda: portfolio.trades.losing.profit.mean() if portfolio.trades.loss_count() > 0 else 0, 'avg_losing_trade'
        )
        
        # Profit maximum sur un trade
        metrics[self._format_metric_name('max_profit')] = self._safe_calculate(
            lambda: portfolio.trades.profit.max() if portfolio.trades.count() > 0 else 0, 'max_profit'
        )
        
        # Perte maximum sur un trade
        metrics[self._format_metric_name('max_loss')] = self._safe_calculate(
            lambda: portfolio.trades.profit.min() if portfolio.trades.count() > 0 else 0, 'max_loss'
        )
        
        # Profit factor
        metrics[self._format_metric_name('profit_factor')] = self._safe_calculate(
            lambda: self._calculate_profit_factor(portfolio), 'profit_factor'
        )
        
        # Profit moyen / perte moyenne (ratio P/L)
        metrics[self._format_metric_name('avg_win_loss_ratio')] = self._safe_calculate(
            lambda: self._calculate_avg_win_loss_ratio(portfolio), 'avg_win_loss_ratio'
        )
        
        # Perte consécutive maximale (maximum drawdown en nombre de trades)
        metrics[self._format_metric_name('max_trade_drawdown')] = self._safe_calculate(
            lambda: self._calculate_max_trade_drawdown(portfolio), 'max_trade_drawdown'
        )
        
        return metrics
    
    def _calculate_duration_stats(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule les statistiques sur la durée des trades.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Statistiques sur la durée des trades
        """
        metrics = {}
        
        try:
            # Durée moyenne des trades (nécessite un index temporel)
            if isinstance(portfolio.trades.records_arr["exit_idx"], np.ndarray) and len(portfolio.trades.records_arr["exit_idx"]) > 0:
                durations = portfolio.trades.records_arr["exit_idx"] - portfolio.trades.records_arr["entry_idx"]
                
                if len(durations) > 0:
                    metrics[self._format_metric_name('avg_bars_in_trade')] = float(durations.mean())
                    metrics[self._format_metric_name('median_bars_in_trade')] = float(np.median(durations))
                    metrics[self._format_metric_name('max_bars_in_trade')] = int(durations.max())
                    metrics[self._format_metric_name('min_bars_in_trade')] = int(durations.min())
                
                # Durée moyenne des trades gagnants vs perdants
                if portfolio.trades.records_arr["profit"].size > 0:
                    winning_mask = portfolio.trades.records_arr["profit"] > 0
                    if winning_mask.any():
                        win_durations = durations[winning_mask]
                        metrics[self._format_metric_name('avg_bars_in_winning_trade')] = float(win_durations.mean())
                    
                    losing_mask = portfolio.trades.records_arr["profit"] < 0
                    if losing_mask.any():
                        lose_durations = durations[losing_mask]
                        metrics[self._format_metric_name('avg_bars_in_losing_trade')] = float(lose_durations.mean())
        except Exception as e:
            self._handle_calculation_error('trade_durations', e)
        
        return metrics
    
    def _calculate_advanced_trade_stats(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule des métriques de trades avancées.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Métriques de trades avancées
        """
        metrics = {}
        
        # Expectancy (espérance mathématique par trade)
        metrics[self._format_metric_name('expectancy')] = self._safe_calculate(
            lambda: self._calculate_expectancy(portfolio), 'expectancy'
        )
        
        # System Quality Number (SQN)
        metrics[self._format_metric_name('sqn')] = self._safe_calculate(
            lambda: self._calculate_sqn(portfolio), 'sqn'
        )
        
        # Taux de réussite par type (long/short)
        if portfolio.trades.long_count() > 0:
            metrics[self._format_metric_name('long_win_rate')] = self._safe_calculate(
                lambda: portfolio.trades.long.win_rate(), 'long_win_rate'
            )
        
        if portfolio.trades.short_count() > 0:
            metrics[self._format_metric_name('short_win_rate')] = self._safe_calculate(
                lambda: portfolio.trades.short.win_rate(), 'short_win_rate'
            )
        
        # Séquences de trades
        streak_metrics = self._calculate_streak_metrics(portfolio)
        metrics.update(streak_metrics)
        
        return metrics
    
    def _calculate_profit_factor(self, portfolio: vbt.Portfolio) -> float:
        """
        Calcule le profit factor (somme des profits / somme des pertes en valeur absolue).
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            float: Profit factor
        """
        winning_trades = portfolio.trades.winning
        losing_trades = portfolio.trades.losing
        
        gross_profit = winning_trades.profit.sum() if winning_trades.count() > 0 else 0
        gross_loss = abs(losing_trades.profit.sum()) if losing_trades.count() > 0 else 0
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
            
        return gross_profit / gross_loss
    
    def _calculate_avg_win_loss_ratio(self, portfolio: vbt.Portfolio) -> float:
        """
        Calcule le ratio entre le profit moyen des trades gagnants et la perte moyenne des trades perdants.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            float: Ratio profit moyen / perte moyenne
        """
        winning_trades = portfolio.trades.winning
        losing_trades = portfolio.trades.losing
        
        avg_win = winning_trades.profit.mean() if winning_trades.count() > 0 else 0
        avg_loss = abs(losing_trades.profit.mean()) if losing_trades.count() > 0 else 0
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 1.0
            
        return avg_win / avg_loss
    
    def _calculate_max_trade_drawdown(self, portfolio: vbt.Portfolio) -> int:
        """
        Calcule la perte consécutive maximale en nombre de trades.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            int: Nombre maximum de trades perdants consécutifs
        """
        if portfolio.trades.count() == 0:
            return 0
            
        # Créer une séquence de 1 (win) et 0 (loss) pour chaque trade
        trade_results = [1 if t > 0 else 0 for t in portfolio.trades.profit]
        
        # Trouver la plus longue séquence de 0 (pertes)
        max_loss_streak = 0
        current_streak = 0
        
        for result in trade_results:
            if result == 0:  # Perte
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:  # Gain
                current_streak = 0
        
        return max_loss_streak
    
    def _calculate_expectancy(self, portfolio: vbt.Portfolio) -> float:
        """
        Calcule l'espérance mathématique par trade (win rate * avg win - (1 - win rate) * avg loss).
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            float: Expectancy
        """
        win_rate = portfolio.trades.win_rate() if portfolio.trades.count() > 0 else 0
        avg_win = portfolio.trades.winning.profit.mean() if portfolio.trades.win_count() > 0 else 0
        avg_loss = abs(portfolio.trades.losing.profit.mean()) if portfolio.trades.loss_count() > 0 else 0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def _calculate_sqn(self, portfolio: vbt.Portfolio) -> float:
        """
        Calcule le System Quality Number (SQN) = (expectancy * sqrt(n)) / stdev.
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            float: System Quality Number
        """
        if portfolio.trades.count() == 0:
            return 0
            
        profit_per_trade = portfolio.trades.profit
        expectancy = profit_per_trade.mean()
        stdev = profit_per_trade.std()
        n = len(profit_per_trade)
        
        if stdev == 0:
            return float('inf') if expectancy > 0 else float('-inf') if expectancy < 0 else 0
            
        return (expectancy * np.sqrt(n)) / stdev
    
    def _calculate_streak_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Calcule les métriques liées aux séquences de trades (winning/losing streaks).
        
        Args:
            portfolio: Le portfolio VectorBT
            
        Returns:
            Dict[str, Any]: Métriques de séquences de trades
        """
        metrics = {}
        
        if portfolio.trades.count() == 0:
            return metrics
            
        # Créer une séquence de 1 (win) et 0 (loss) pour chaque trade
        trade_results = [1 if t > 0 else 0 for t in portfolio.trades.profit]
        
        # Trouver les séquences de gains et de pertes
        win_streaks = self._find_streaks(trade_results, 1)
        loss_streaks = self._find_streaks(trade_results, 0)
        
        # Métriques de séquences de gains
        if win_streaks:
            metrics[self._format_metric_name('max_win_streak')] = max(win_streaks)
            metrics[self._format_metric_name('avg_win_streak')] = sum(win_streaks) / len(win_streaks)
        else:
            metrics[self._format_metric_name('max_win_streak')] = 0
            metrics[self._format_metric_name('avg_win_streak')] = 0
        
        # Métriques de séquences de pertes
        if loss_streaks:
            metrics[self._format_metric_name('max_loss_streak')] = max(loss_streaks)
            metrics[self._format_metric_name('avg_loss_streak')] = sum(loss_streaks) / len(loss_streaks)
        else:
            metrics[self._format_metric_name('max_loss_streak')] = 0
            metrics[self._format_metric_name('avg_loss_streak')] = 0
        
        return metrics
    
    def _find_streaks(self, results: List[int], target: int) -> List[int]:
        """
        Trouve toutes les séquences de trades du type spécifié.
        
        Args:
            results: Liste de résultats de trades (1 pour gain, 0 pour perte)
            target: Type de séquence à rechercher (1 pour gains, 0 pour pertes)
            
        Returns:
            List[int]: Longueurs de toutes les séquences trouvées
        """
        streaks = []
        current_streak = 0
        
        for result in results:
            if result == target:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # Ajouter la dernière séquence si nécessaire
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks