# src/strategies/implementations/sma_cross.py

"""
A classic trend-following strategy based on the crossover of two Simple Moving
Averages (SMAs).
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.indicators.registry import IndicatorRegistry
from src.strategies.parameters import (
    ParameterSet,
    IntParameter,
)
from src.strategies.signals import (
    Signal,
    SignalDirection,
    SignalType,
)


class SMACrossStrategy(BaseStrategy):
    """
    Implements the SMA Crossover strategy.

    Generates a LONG entry signal when a short-period SMA crosses above a
    long-period SMA.
    Generates a SHORT entry signal when a short-period SMA crosses below a
    long-period SMA.
    This version generates entry signals only and does not manage exits.
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        """
        Initializes the SMA Crossover strategy.

        Args:
            symbol (str): The symbol to be traded (e.g., 'BTC/USDT').
            params (Dict[str, Any]): The parameters for the strategy instance,
                                     validated against the defined ParameterSet.
        """
        self.params = self.get_parameters().validate(params)
        self.symbol = symbol

    @property
    def name(self) -> str:
        return "SMACross"

    @property
    def description(self) -> str:
        return "A simple strategy based on the crossover of two SMAs."

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        """Defines the parameters required for this strategy."""
        return ParameterSet([
            IntParameter(
                name="fast_period",
                default=10,
                min_value=2,
                max_value=100,
                help="The period for the fast Simple Moving Average."
            ),
            IntParameter(
                name="slow_period",
                default=30,
                min_value=5,
                max_value=300,
                help="The period for the slow Simple Moving Average."
            ),
        ])

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculates the fast and slow SMAs required by the strategy.
        """
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        # Ensure slow period is greater than fast period
        if slow_period <= fast_period:
            # In a real-world scenario, this might raise an error.
            # Here we adjust it silently to ensure valid calculation.
            slow_period = fast_period + 1

        # Create indicator instances via the registry
        fast_sma_indicator = IndicatorRegistry.create("sma", period=fast_period)
        slow_sma_indicator = IndicatorRegistry.create("sma", period=slow_period)

        # Calculate indicators
        fast_sma_series = fast_sma_indicator.calculate(data)["sma"]
        slow_sma_series = slow_sma_indicator.calculate(data)["sma"]

        return {"sma_fast": fast_sma_series, "sma_slow": slow_sma_series}

    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> List[Signal]:
        """
        Generates trading signals based on SMA crossovers.
        """
        sma_fast = indicators["sma_fast"]
        sma_slow = indicators["sma_slow"]

        # Find crossover points using vectorized operations for performance
        # A "golden cross" (long entry) happens when fast SMA crosses above slow SMA
        golden_cross = (
            (sma_fast.shift(1) <= sma_slow.shift(1)) &
            (sma_fast > sma_slow)
        )
        # A "death cross" (short entry) happens when fast SMA crosses below slow SMA
        death_cross = (
            (sma_fast.shift(1) >= sma_slow.shift(1)) &
            (sma_fast < sma_slow)
        )

        long_entry_points = data.loc[golden_cross]
        short_entry_points = data.loc[death_cross]

        signals = []
        for timestamp, row in long_entry_points.iterrows():
            signals.append(Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                metadata={"fast_sma": sma_fast.loc[timestamp], "slow_sma": sma_slow.loc[timestamp]}
            ))

        for timestamp, row in short_entry_points.iterrows():
            signals.append(Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.SHORT,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                metadata={"fast_sma": sma_fast.loc[timestamp], "slow_sma": sma_slow.loc[timestamp]}
            ))
        
        # Sort signals by timestamp to ensure chronological order
        signals.sort(key=lambda s: s.timestamp)

        return signals
