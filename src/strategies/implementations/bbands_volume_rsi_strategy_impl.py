# src/strategies/implementations/bbands_volume_rsi_strategy_impl.py

"""
A mean-reversion strategy combining Bollinger Bands, RSI, and volume.
"""

from typing import Dict, List, Any

import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.indicators.registry import IndicatorRegistry
from src.strategies.parameters import (
    ParameterSet,
    IntParameter,
    FloatParameter,
    BoolParameter,
)
from src.strategies.signals import (
    Signal,
    SignalDirection,
    SignalType,
)


class BBandsVolumeRSIStrategy(BaseStrategy):
    """
    Implements a mean-reversion strategy using Bollinger Bands and RSI.

    - Entry Conditions:
        - LONG: Price closes below the lower Bollinger Band and RSI is oversold.
        - SHORT: Price closes above the upper Bollinger Band and RSI is overbought.
    - Exit Conditions:
        - This implementation only generates entry signals. Exits are expected
          to be managed by the execution engine (e.g., via a trailing stop,
          take profit, or a separate exit signal logic).
    - Stop-Loss:
        - Can optionally be calculated using ATR for dynamic risk management.
    """

    def __init__(self, symbol: str, params: Dict[str, Any]):
        """
        Initializes the BBandsVolumeRSIStrategy.

        Args:
            symbol (str): The symbol to be traded (e.g., 'BTC/USDT').
            params (Dict[str, Any]): The parameters for the strategy instance.
        """
        self.params = self.get_parameters().validate(params)
        self.symbol = symbol

    @property
    def name(self) -> str:
        return "BBandsVolumeRSI"

    @property
    def description(self) -> str:
        return "A mean-reversion strategy using Bollinger Bands, RSI, and optional ATR stops."

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        """Defines the parameters required for this strategy."""
        return ParameterSet([
            # Bollinger Bands parameters
            IntParameter("bb_period", 20, 5, 100, help="Period for the Bollinger Bands SMA."),
            FloatParameter("bb_std_dev", 2.0, 1.0, 4.0, help="Standard deviations for the Bollinger Bands."),
            # RSI parameters
            IntParameter("rsi_period", 14, 5, 50, help="Period for the RSI calculation."),
            IntParameter("rsi_oversold", 30, 10, 40, help="RSI level for oversold condition."),
            IntParameter("rsi_overbought", 70, 60, 90, help="RSI level for overbought condition."),
            # ATR Stop-Loss parameters
            BoolParameter("use_atr_stop", True, help="Whether to use ATR for stop-loss calculation."),
            IntParameter("atr_period", 14, 5, 50, help="Period for the ATR calculation."),
            FloatParameter("atr_multiplier", 2.0, 1.0, 5.0, help="Multiplier for the ATR stop-loss distance."),
        ])

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates all indicators required by the strategy."""
        indicators = {}

        # Bollinger Bands
        bb_indicator = IndicatorRegistry.create(
            "bbands", period=self.params["bb_period"], std_dev=self.params["bb_std_dev"]
        )
        indicators.update(bb_indicator.calculate(data))

        # RSI
        rsi_indicator = IndicatorRegistry.create("rsi", period=self.params["rsi_period"])
        indicators.update(rsi_indicator.calculate(data))
        
        # ATR for stop-loss
        if self.params["use_atr_stop"]:
            atr_indicator = IndicatorRegistry.create("atr", period=self.params["atr_period"])
            indicators.update(atr_indicator.calculate(data))

        return indicators

    def generate_signals(
        self, data: pd.DataFrame, indicators: Dict[str, pd.Series]
    ) -> List[Signal]:
        """Generates trading signals based on the strategy's logic."""
        signals = []
        close = data["close"]
        
        # Extract indicator series
        bbands_lower = indicators["bbands_lower"]
        bbands_upper = indicators["bbands_upper"]
        rsi = indicators["rsi"]
        
        # Define conditions using vectorized operations
        long_condition = (close < bbands_lower) & (rsi < self.params["rsi_oversold"])
        short_condition = (close > bbands_upper) & (rsi > self.params["rsi_overbought"])

        # Filter data points where signals occur
        long_entry_points = data.loc[long_condition]
        short_entry_points = data.loc[short_condition]

        # Create LONG signals
        for timestamp, row in long_entry_points.iterrows():
            stop_loss = None
            if self.params["use_atr_stop"] and "atr" in indicators:
                atr_value = indicators["atr"].get(timestamp)
                if atr_value is not None:
                    stop_loss = row["close"] - (atr_value * self.params["atr_multiplier"])
            
            signals.append(Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                stop_loss=stop_loss,
                metadata={k: v.get(timestamp) for k, v in indicators.items()}
            ))

        # Create SHORT signals
        for timestamp, row in short_entry_points.iterrows():
            stop_loss = None
            if self.params["use_atr_stop"] and "atr" in indicators:
                atr_value = indicators["atr"].get(timestamp)
                if atr_value is not None:
                    stop_loss = row["close"] + (atr_value * self.params["atr_multiplier"])

            signals.append(Signal(
                strategy_name=self.name,
                symbol=self.symbol,
                timestamp=timestamp,
                direction=SignalDirection.SHORT,
                signal_type=SignalType.ENTRY,
                price=row["close"],
                stop_loss=stop_loss,
                metadata={k: v.get(timestamp) for k, v in indicators.items()}
            ))
            
        signals.sort(key=lambda s: s.timestamp)
        return signals
