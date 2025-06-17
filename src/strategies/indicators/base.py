# src/strategies/indicators/base.py

"""
Defines the abstract base class for all technical indicators.

This module provides the `Indicator` class, which serves as a pure abstract
interface. Any new indicator implemented in the framework must inherit from
this class and implement its abstract methods.

This approach ensures that all indicators are stateless, have a consistent API,
and can be easily registered, discovered, and used by strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pandas as pd

from src.strategies.parameters import ParameterSet


class Indicator(ABC):
    """
    An abstract base class that defines the contract for a technical indicator.

    An indicator is a stateless function that transforms market data (usually a
    pandas Series) into another Series representing the indicator's value over
    time.
    """

    def __init__(self, **params: Any):
        """
        Initializes the indicator with its parameters.

        The constructor receives keyword arguments that correspond to the
        parameters defined in `get_parameters()`. It validates them and stores
        them for the `calculate` method.

        Args:
            **params: The parameters for this specific indicator instance.
        """
        self.parameters = self.get_parameters().validate(params)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provides the unique, machine-readable name of the indicator
        (e.g., "sma", "rsi").
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Provides a brief, human-readable description of the indicator's purpose.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Specifies the category of the indicator (e.g., "Trend", "Momentum",
        "Volatility", "Volume").
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def required_input_names(self) -> List[str]:
        """
        A list of column names required from the input DataFrame.
        For example: ["close"], ["high", "low", "close"], ["volume"]
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> ParameterSet:
        """
        Returns the set of parameters that configure this indicator.

        This method defines the complete set of configurable options for the
        indicator, such as the period, standard deviation, etc.

        Returns:
            ParameterSet: An object containing the definition of all indicator
                          parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculates the indicator's value(s).

        This method takes a pandas DataFrame containing the required market data
        (as defined in `required_input_names`) and returns a dictionary of
        pandas Series representing the calculated indicator values. The keys
        of the dictionary are the output names (e.g., "sma", or "macd", "macdsignal",
        "macdhist" for MACD).

        Args:
            data (pd.DataFrame): A DataFrame with columns corresponding to
                                 `required_input_names`.

        Returns:
            Dict[str, pd.Series]: A dictionary of calculated indicator series,
                                  aligned with the input data's index.
        """
        raise NotImplementedError

    def get_param(self, key: str) -> Any:
        """
        A convenience method to safely access a parameter's value.
        """
        if key not in self.parameters:
            # This should ideally not happen if validation is correct.
            raise KeyError(f"Parameter '{key}' not found for indicator '{self.name}'.")
        return self.parameters[key]

    @property
    def min_required_periods(self) -> int:
        """
        Estimates the minimum number of data periods required for the
        indicator to produce a non-NaN value. By default, it looks for a
        'period' parameter. Subclasses should override this if the logic is
        more complex.

        Returns:
            int: The minimum lookback period.
        """
        if "period" in self.parameters:
            return int(self.get_param("period"))
        return 1

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params_str})"
