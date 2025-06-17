# src/backtesting/metrics/base.py

"""
Defines the abstract base class for all performance metrics.

This module provides the `Metric` interface, which establishes a standard
contract for calculating performance and risk metrics from backtest results.
This allows for a modular and extensible system where new metrics can be
easily added and calculated independently.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from src.backtesting.portfolio.portfolio import Portfolio


class Metric(ABC):
    """
    An abstract base class that defines the contract for a performance metric.
    
    A metric is a stateless function that takes the results of a backtest
    (such as the equity curve and transaction history) and computes a specific
    scalar value or a series of values.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provides the unique, machine-readable name of the metric
        (e.g., "sharpe_ratio", "max_drawdown").
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Provides a brief, human-readable description of what the metric measures.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Specifies the category of the metric (e.g., "Returns", "Risk", "Trades").
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(
        self,
        equity_curve: pd.Series,
        portfolio: Portfolio,
        benchmark_returns: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> Any:
        """
        Calculates the metric's value.

        Args:
            equity_curve (pd.Series): The portfolio's value over time.
                                      The index is a datetime, and values are floats.
            portfolio (Portfolio): The final portfolio object, containing transaction
                                   history and other state information.
            benchmark_returns (Optional[pd.Series]): A series of benchmark returns,
                                                      aligned with the equity curve's
                                                      index, for calculating relative
                                                      metrics like Beta or Alpha.
            **kwargs: Additional parameters, such as the risk-free rate, required
                      by specific metric calculations.

        Returns:
            Any: The calculated metric value. This can be a float, int, str,
                 or even a more complex object like a dictionary.
        """
        raise NotImplementedError
