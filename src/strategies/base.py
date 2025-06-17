# src/strategies/base.py

"""
Defines the abstract base class for all trading strategies.

This module provides the `BaseStrategy` class, which serves as a pure abstract
interface. Any new strategy implemented in the framework must inherit from this
class and implement its abstract methods.

This approach enforces a consistent structure for all strategies, decoupling the
strategy's core logic (indicator calculation and signal generation) from the
backtesting engine, data handling, and execution logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING

import pandas as pd

# Use TYPE_CHECKING block to avoid circular imports at runtime,
# allowing for type hints of classes that will be defined later.
if TYPE_CHECKING:
    from .parameters import ParameterSet
    from .signals import Signal


class BaseStrategy(ABC):
    """
    An abstract base class that defines the contract for a trading strategy.

    A strategy is defined by its parameters, the indicators it calculates, and
    the logic it uses to generate trading signals from those indicators.
    It is designed to be stateless regarding the backtest execution.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provides the unique, human-readable name of the strategy.
        This name is used for identification in logs, reports, and configurations.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Provides a brief description of the strategy's logic and purpose.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> 'ParameterSet':
        """
        Returns the set of parameters that configure this strategy.

        This method defines the complete set of configurable options, including
        their types, default values, and constraints. The parameter set is used
        for validation, optimization, and configuration management.

        Returns:
            ParameterSet: An object containing the definition of all strategy
                          parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculates all technical indicators required by the strategy.

        This method should receive historical market data and compute one or more
        indicator series. The results are returned as a dictionary where keys are
        indicator names and values are pandas Series aligned with the input data's
        index.

        Args:
            data (pd.DataFrame): A DataFrame containing market data (OHLCV).
                                 It must be indexed by timestamp.

        Returns:
            Dict[str, pd.Series]: A dictionary of calculated indicator series.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> List['Signal']:
        """
        Generates trading signals based on the input data and calculated indicators.

        This is the core logic of the strategy. It should analyze the indicators
        and produce a list of discrete `Signal` objects, representing trading
        decisions (e.g., enter long, exit short).

        Args:
            data (pd.DataFrame): The original market data (OHLCV).
            indicators (Dict[str, pd.Series]): The indicators calculated by the
                                               `calculate_indicators` method.

        Returns:
            List[Signal]: A list of generated Signal objects. The list can be
                          empty if no trading opportunities are found.
        """
        raise NotImplementedError
