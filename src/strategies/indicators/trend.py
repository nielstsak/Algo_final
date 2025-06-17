# src/strategies/indicators/technical/trend.py

"""
Implementations of standard trend-following technical indicators.

This module provides concrete implementations of common trend indicators like
Simple Moving Average (SMA) and Exponential Moving Average (EMA), built upon
the base `Indicator` interface.

Each indicator is self-contained, defines its own parameters, and is registered
with the `IndicatorRegistry` for dynamic instantiation by strategies.
The calculations are delegated to the robust `pandas-ta` library.
"""

from typing import Dict, List, Any

import pandas as pd
import pandas_ta as ta

from src.core.exceptions import IndicatorError
from src.strategies.indicators.base import Indicator
from src.strategies.indicators.registry import register_indicator
from src.strategies.parameters import ParameterSet, IntParameter


# --- Simple Moving Average (SMA) ---

@register_indicator(name="sma")
class SMAIndicator(Indicator):
    """Calculates the Simple Moving Average (SMA)."""

    @property
    def name(self) -> str:
        return "sma"

    @property
    def description(self) -> str:
        return "Calculates the Simple Moving Average (SMA) of a series."

    @property
    def category(self) -> str:
        return "Trend"

    @property
    def required_input_names(self) -> List[str]:
        return ["close"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            IntParameter(
                name="period",
                default=20,
                min_value=2,
                max_value=500,
                help="The time period window for the moving average."
            )
        ])

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.get_param("period")
        try:
            sma_series = ta.sma(data["close"], length=period)
            if sma_series is None:
                 raise IndicatorError("pandas_ta.sma returned None.")
        except Exception as e:
            raise IndicatorError(f"Error calculating SMA for period {period}.") from e

        # The output key should match the indicator's registered name for consistency.
        return {self.name: sma_series}


# --- Exponential Moving Average (EMA) ---

@register_indicator(name="ema")
class EMAIndicator(Indicator):
    """Calculates the Exponential Moving Average (EMA)."""

    @property
    def name(self) -> str:
        return "ema"

    @property
    def description(self) -> str:
        return "Calculates the Exponential Moving Average (EMA) of a series."

    @property
    def category(self) -> str:
        return "Trend"

    @property
    def required_input_names(self) -> List[str]:
        return ["close"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            IntParameter(
                name="period",
                default=20,
                min_value=2,
                max_value=500,
                help="The time period window for the moving average."
            )
        ])

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.get_param("period")
        try:
            ema_series = ta.ema(data["close"], length=period)
            if ema_series is None:
                raise IndicatorError("pandas_ta.ema returned None.")
        except Exception as e:
            raise IndicatorError(f"Error calculating EMA for period {period}.") from e

        return {self.name: ema_series}

# --- Moving Average Convergence Divergence (MACD) ---

@register_indicator(name="macd")
class MACDIndicator(Indicator):
    """Calculates the Moving Average Convergence Divergence (MACD)."""

    @property
    def name(self) -> str:
        return "macd"

    @property
    def description(self) -> str:
        return "Calculates MACD, Signal Line, and Histogram."

    @property
    def category(self) -> str:
        return "Trend"

    @property
    def required_input_names(self) -> List[str]:
        return ["close"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            IntParameter("fast", 12, 2, 100, help="The period for the fast EMA."),
            IntParameter("slow", 26, 5, 200, help="The period for the slow EMA."),
            IntParameter("signal", 9, 2, 100, help="The period for the signal line EMA."),
        ])

    @property
    def min_required_periods(self) -> int:
        return self.get_param("slow") + self.get_param("signal")

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        fast = self.get_param("fast")
        slow = self.get_param("slow")
        signal = self.get_param("signal")

        try:
            macd_df = ta.macd(data["close"], fast=fast, slow=slow, signal=signal)
            if macd_df is None or macd_df.empty:
                raise IndicatorError("pandas_ta.macd returned None or an empty DataFrame.")
        except Exception as e:
            raise IndicatorError(
                f"Error calculating MACD with params fast={fast}, slow={slow}, signal={signal}."
            ) from e

        # pandas-ta returns a df with columns like 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
        # We rename them to simple, predictable names for strategies to use.
        output = {
            "macd": macd_df.iloc[:, 0],      # The MACD line
            "macdsignal": macd_df.iloc[:, 1], # The Signal line
            "macdhist": macd_df.iloc[:, 2],   # The Histogram
        }
        return output
