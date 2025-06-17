# src/backtesting/engine/base.py

"""
Defines the abstract base class for all backtesting engines.

This module provides the `BacktestEngine` interface, which establishes a
standard contract for running backtests, regardless of the underlying
implementation (e.g., vectorized or event-driven).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from src.backtesting.portfolio.portfolio import Portfolio
from src.strategies.base import BaseStrategy


class BacktestResult:
    """
    A data class to hold the results of a backtest run.
    This will be expanded later with structured metrics, reports, etc.
    """
    def __init__(self, stats: Dict[str, Any], equity_curve: pd.Series, portfolio: Portfolio):
        self.stats = stats
        self.equity_curve = equity_curve
        self.portfolio = portfolio

    def __repr__(self) -> str:
        return f"BacktestResult(stats={self.stats})"


class BacktestEngine(ABC):
    """
    An abstract base class that defines the contract for a backtesting engine.
    """

    @abstractmethod
    def __init__(self, config: Any):
        """
        Initializes the backtesting engine with a configuration object.

        Args:
            config (Any): A dedicated configuration object containing all
                          necessary settings for the backtest (e.g., fees,
                          slippage, initial cash).
        """
        self.config = config

    @abstractmethod
    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Executes a backtest for a given strategy and dataset.

        Args:
            strategy (BaseStrategy): The strategy instance to be tested.
            data (pd.DataFrame): The historical market data (OHLCV) for the backtest,
                                 indexed by timestamp.

        Returns:
            BacktestResult: An object containing the results of the backtest.
        """
        raise NotImplementedError

    @abstractmethod
    def run_multiple(
        self,
        strategies: Dict[str, BaseStrategy],
        data: pd.DataFrame
    ) -> Dict[str, BacktestResult]:
        """
        Executes backtests for multiple strategies on the same dataset.

        Args:
            strategies (Dict[str, BaseStrategy]): A dictionary where keys are
                                                  strategy identifiers and values
                                                  are strategy instances.
            data (pd.DataFrame): The historical market data.

        Returns:
            Dict[str, BacktestResult]: A dictionary mapping strategy identifiers
                                       to their respective BacktestResult objects.
        """
        raise NotImplementedError
