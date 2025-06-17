# src/strategies/implementations/strategy_template.py

"""
A template for creating a new custom trading strategy.

This file serves as a starting point and a guide for developers looking to
implement their own strategies within the framework. It demonstrates the
required structure and best practices.
"""

from typing import Dict, List, Any

import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.indicators.registry import IndicatorRegistry
from src.strategies.parameters import (
    ParameterSet,
    IntParameter,
    # Import other parameter types as needed, e.g., FloatParameter
)
from src.strategies.signals import (
    Signal,
    SignalDirection,
    SignalType,
)


class StrategyTemplate(BaseStrategy):
    """
    A template class for a new strategy.
    
    This strategy should be inherited from and customized. As a placeholder, it
    demonstrates how to use a Simple Moving Average (SMA) but does not generate
    any signals by default.
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        """
        Initializes the custom strategy.

        Args:
            symbol (str): The symbol to be traded (e.g., 'BTC/USDT').
            params (Dict[str, Any]): The parameters for the strategy instance,
                                     validated against the defined ParameterSet.
        """
        # Validate and store parameters
        self.params = self.get_parameters().validate(params)
        # Store the trading symbol
        self.symbol = symbol

    # --- Step 1: Define Strategy Metadata ---

    @property
    def name(self) -> str:
        """Return the unique name of your strategy."""
        return "StrategyTemplate"

    @property
    def description(self) -> str:
        """Provide a clear, concise description of what your strategy does."""
        return "A template for creating new strategies."

    # --- Step 2: Define Strategy Parameters ---

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        """
        Define the set of parameters that will configure your strategy.
        These parameters will be adjustable in backtests and optimizations.
        """
        return ParameterSet([
            IntParameter(
                name="example_period",
                default=50,
                min_value=10,
                max_value=200,
                help="An example parameter for the strategy."
            ),
            # Add other parameters here (FloatParameter, BoolParameter, etc.)
        ])

    # --- Step 3: Define and Calculate Indicators ---

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all the technical indicators your strategy needs.
        
        Use the IndicatorRegistry to create instances of indicators and run them.
        Return a dictionary where keys are descriptive names and values are the
        resulting pandas Series.
        """
        # Example: Calculate a Simple Moving Average using the parameter
        example_period = self.params["example_period"]
        sma_indicator = IndicatorRegistry.create("sma", period=example_period)
        
        # The result of calculate() is a dictionary, so we extract the series we need
        sma_series = sma_indicator.calculate(data)["sma"]

        # You can calculate as many indicators as you need
        # rsi_indicator = IndicatorRegistry.create("rsi", period=14)
        # rsi_series = rsi_indicator.calculate(data)["rsi"]

        # Return all calculated series in a single dictionary
        return {
            "sma_example": sma_series,
            # "rsi": rsi_series,
        }

    # --- Step 4: Implement the Signal Generation Logic ---

    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> List[Signal]:
        """
        This is the core logic of your strategy.
        
        Analyze the data and indicators to decide when to enter or exit a trade.
        Return a list of Signal objects for every decision made.
        """
        signals: List[Signal] = []

        # Extract the indicator series you calculated
        close_prices = data["close"]
        sma = indicators["sma_example"]
        
        # --- Example Logic: Simple Crossover ---
        # Note: This is a placeholder. Replace it with your actual trading logic.
        
        # Condition for a long entry (price crosses above SMA)
        long_condition = (close_prices.shift(1) <= sma.shift(1)) & (close_prices > sma)
        
        # Condition for a short entry (price crosses below SMA)
        short_condition = (close_prices.shift(1) >= sma.shift(1)) & (close_prices < sma)

        # Iterate through the data points where a signal might occur
        # It is more performant to check conditions on the whole series first,
        # then iterate only on the results.
        
        long_entry_points = data.loc[long_condition]
        for timestamp, row in long_entry_points.iterrows():
            # Create a Signal object for each long entry
            signal = Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                metadata={"sma_value": sma.get(timestamp)}
            )
            signals.append(signal)

        short_entry_points = data.loc[short_condition]
        for timestamp, row in short_entry_points.iterrows():
            # Create a Signal object for each short entry
            signal = Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.SHORT,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                metadata={"sma_value": sma.get(timestamp)}
            )
            signals.append(signal)
            
        # It's good practice to sort signals by date before returning
        signals.sort(key=lambda s: s.timestamp)
        
        return signals
