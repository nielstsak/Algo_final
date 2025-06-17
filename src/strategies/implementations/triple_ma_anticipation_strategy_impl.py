# src/strategies/implementations/triple_ma_anticipation_strategy_impl.py

"""
A trend-following strategy that uses three Exponential Moving Averages (EMAs)
to anticipate entries in an established trend.
"""

from typing import Dict, List, Any

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


class TripleMAAnticipationStrategy(BaseStrategy):
    """
    Implements the Triple Moving Average Anticipation strategy.

    This strategy uses three EMAs (fast, medium, slow) to identify a trend
    and enter on a pullback.
    - An uptrend is confirmed when medium EMA is above slow EMA. A LONG entry
      is triggered when the fast EMA crosses above the medium EMA.
    - A downtrend is confirmed when medium EMA is below slow EMA. A SHORT entry
      is triggered when the fast EMA crosses below the medium EMA.
    - This version only generates entry signals.
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        """
        Initializes the TripleMAAnticipationStrategy.

        Args:
            symbol (str): The symbol to be traded (e.g., 'BTC/USDT').
            params (Dict[str, Any]): The parameters for the strategy instance.
        """
        self.params = self.get_parameters().validate(params)
        self.symbol = symbol

    @property
    def name(self) -> str:
        return "TripleMAAnticipation"

    @property
    def description(self) -> str:
        return "A trend-following strategy using three EMAs to time entries."

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        """Defines the parameters required for this strategy."""
        return ParameterSet([
            IntParameter("fast_period", 9, 2, 50, help="Period for the fastest EMA."),
            IntParameter("medium_period", 21, 5, 100, help="Period for the medium EMA."),
            IntParameter("slow_period", 50, 10, 200, help="Period for the slowest EMA (trend filter)."),
        ])

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates the three EMAs required by the strategy."""
        fast_period = self.params["fast_period"]
        medium_period = self.params["medium_period"]
        slow_period = self.params["slow_period"]

        # Basic validation to ensure logical periods
        if not (fast_period < medium_period < slow_period):
            # In a real scenario, this should raise a configuration error.
            # For this implementation, we will log a warning but proceed.
            # logger.warning("EMA periods are not logical: fast < medium < slow is expected.")
            pass

        # Create indicator instances
        fast_ema_ind = IndicatorRegistry.create("ema", period=fast_period)
        medium_ema_ind = IndicatorRegistry.create("ema", period=medium_period)
        slow_ema_ind = IndicatorRegistry.create("ema", period=slow_period)

        # Calculate indicators and return them in a dictionary
        return {
            "ema_fast": fast_ema_ind.calculate(data)["ema"],
            "ema_medium": medium_ema_ind.calculate(data)["ema"],
            "ema_slow": slow_ema_ind.calculate(data)["ema"],
        }

    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> List[Signal]:
        """Generates trading signals based on the triple EMA logic."""
        ema_fast = indicators["ema_fast"]
        ema_medium = indicators["ema_medium"]
        ema_slow = indicators["ema_slow"]

        # --- Conditions for Long Entry ---
        # 1. Trend confirmation: Medium EMA is above Slow EMA.
        uptrend = ema_medium > ema_slow
        # 2. Entry trigger: Fast EMA crosses above Medium EMA.
        long_entry_trigger = (ema_fast.shift(1) <= ema_medium.shift(1)) & (ema_fast > ema_medium)
        # 3. Combined condition
        long_condition = uptrend & long_entry_trigger

        # --- Conditions for Short Entry ---
        # 1. Trend confirmation: Medium EMA is below Slow EMA.
        downtrend = ema_medium < ema_slow
        # 2. Entry trigger: Fast EMA crosses below Medium EMA.
        short_entry_trigger = (ema_fast.shift(1) >= ema_medium.shift(1)) & (ema_fast < ema_medium)
        # 3. Combined condition
        short_condition = downtrend & short_entry_trigger
        
        # Filter data points where signals occur
        long_entry_points = data.loc[long_condition]
        short_entry_points = data.loc[short_condition]

        signals = []
        for timestamp, row in long_entry_points.iterrows():
            signals.append(Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                metadata={
                    "ema_fast": ema_fast.get(timestamp),
                    "ema_medium": ema_medium.get(timestamp),
                    "ema_slow": ema_slow.get(timestamp)
                }
            ))

        for timestamp, row in short_entry_points.iterrows():
            signals.append(Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.SHORT,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                metadata={
                    "ema_fast": ema_fast.get(timestamp),
                    "ema_medium": ema_medium.get(timestamp),
                    "ema_slow": ema_slow.get(timestamp)
                }
            ))

        signals.sort(key=lambda s: s.timestamp)
        return signals
