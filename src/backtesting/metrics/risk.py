# src/backtesting/metrics/risk.py

"""
Implementations of standard risk-based performance metrics.

This module provides concrete implementations of metrics that quantify the
risk and volatility of a trading strategy, such as Maximum Drawdown and
Annualized Volatility.
"""

import numpy as np
import pandas as pd

from src.backtesting.metrics.base import Metric
from src.backtesting.metrics.registry import register_metric
# Use a local import inside a method to avoid circular dependencies
# from .returns import _get_daily_returns, TRADING_DAYS_PER_YEAR, AnnualizedReturnMetric

# As _get_daily_returns is a private helper, we redefine it here to keep modules decoupled
def _get_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Helper function to calculate daily returns from an equity curve."""
    return equity_curve.resample('D').last().ffill().pct_change().dropna()

TRADING_DAYS_PER_YEAR = 252


@register_metric(name="max_drawdown")
class MaxDrawdownMetric(Metric):
    """Calculates the maximum drawdown of the portfolio."""
    @property
    def name(self) -> str:
        return "max_drawdown"
    @property
    def description(self) -> str:
        return "The largest peak-to-trough decline in the value of a portfolio."
    @property
    def category(self) -> str:
        return "Risk"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        
        # Return as a positive value
        return drawdown.min() * -1.0 if not drawdown.empty else 0.0


@register_metric(name="annualized_volatility")
class AnnualizedVolatilityMetric(Metric):
    """Calculates the annualized volatility of portfolio returns."""
    @property
    def name(self) -> str:
        return "annualized_volatility"
    @property
    def description(self) -> str:
        return "The annualized standard deviation of portfolio returns."
    @property
    def category(self) -> str:
        return "Risk"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        daily_returns = _get_daily_returns(equity_curve)
        if daily_returns.empty:
            return 0.0
            
        return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


@register_metric(name="calmar_ratio")
class CalmarRatioMetric(Metric):
    """Calculates the Calmar Ratio."""
    @property
    def name(self) -> str:
        return "calmar_ratio"
    @property
    def description(self) -> str:
        return "A risk-adjusted return metric based on annualized return and maximum drawdown."
    @property
    def category(self) -> str:
        return "Risk-Adjusted Returns"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> float:
        # Local import to avoid circular dependency
        from .returns import AnnualizedReturnMetric
        
        if len(equity_curve) < 2:
            return 0.0

        max_drawdown = MaxDrawdownMetric().calculate(equity_curve)
        if max_drawdown == 0:
            return np.inf

        annualized_return = AnnualizedReturnMetric().calculate(equity_curve, **kwargs)
        
        return annualized_return / max_drawdown
