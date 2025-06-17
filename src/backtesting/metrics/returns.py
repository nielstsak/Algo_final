# src/backtesting/metrics/returns.py

"""
Implementations of standard return-based performance metrics.

This module provides concrete implementations of common metrics related to
portfolio returns, such as Annualized Return and Sharpe Ratio.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.backtesting.metrics.base import Metric
from src.backtesting.metrics.registry import register_metric
from src.backtesting.portfolio.portfolio import Portfolio

# Define a constant for the number of trading days in a year for annualization.
TRADING_DAYS_PER_YEAR = 252


def _get_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Helper function to calculate daily returns from an equity curve."""
    return equity_curve.resample('D').last().ffill().pct_change().dropna()


@register_metric(name="annualized_return")
class AnnualizedReturnMetric(Metric):
    """Calculates the annualized return of the portfolio."""
    @property
    def name(self) -> str:
        return "annualized_return"
    @property
    def description(self) -> str:
        return "The geometric average amount of money earned by an investment each year over a given time period."
    @property
    def category(self) -> str:
        return "Returns"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        daily_returns = _get_daily_returns(equity_curve)
        if daily_returns.empty:
            return 0.0
            
        mean_daily_return = daily_returns.mean()
        return (1 + mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1


@register_metric(name="cumulative_return")
class CumulativeReturnMetric(Metric):
    """Calculates the total cumulative return of the portfolio."""
    @property
    def name(self) -> str:
        return "cumulative_return"
    @property
    def description(self) -> str:
        return "The total return of the investment over the entire period."
    @property
    def category(self) -> str:
        return "Returns"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> float:
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1


@register_metric(name="sharpe_ratio")
class SharpeRatioMetric(Metric):
    """Calculates the Sharpe Ratio."""
    @property
    def name(self) -> str:
        return "sharpe_ratio"
    @property
    def description(self) -> str:
        return "Measures the performance of an investment compared to a risk-free asset, after adjusting for its risk."
    @property
    def category(self) -> str:
        return "Risk-Adjusted Returns"

    def calculate(self, equity_curve: pd.Series, risk_free_rate: float = 0.0, **kwargs) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        daily_returns = _get_daily_returns(equity_curve)
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0

        excess_returns = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        # Annualize the Sharpe Ratio
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
