# src/strategies/indicators/technical/momentum.py

"""
Implementations of standard momentum-based technical indicators.

This module provides concrete implementations of common momentum indicators like
the Relative Strength Index (RSI) and the Stochastic Oscillator, built upon
the base `Indicator` interface.

Each indicator is self-contained, defines its own parameters, and is registered
with the `IndicatorRegistry` for dynamic instantiation by strategies.
The calculations are delegated to the robust `pandas-ta` library.
"""

from typing import Dict, List

import pandas as pd
import pandas_ta as ta

from src.core.exceptions import IndicatorError
from src.strategies.indicators.base import Indicator
from src.strategies.indicators.registry import register_indicator
from src.strategies.parameters import ParameterSet, IntParameter


# --- Relative Strength Index (RSI) ---

@register_indicator(name="rsi")
class RSIIndicator(Indicator):
    """Calculates the Relative Strength Index (RSI)."""

    @property
    def name(self) -> str:
        return "rsi"

    @property
    def description(self) -> str:
        return "Calculates the Relative Strength Index (RSI)."

    @property
    def category(self) -> str:
        return "Momentum"

    @property
    def required_input_names(self) -> List[str]:
        return ["close"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            IntParameter(
                name="period",
                default=14,
                min_value=2,
                max_value=100,
                help="The time period for RSI calculation."
            )
        ])

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.get_param("period")
        try:
            rsi_series = ta.rsi(data["close"], length=period)
            if rsi_series is None:
                raise IndicatorError("pandas_ta.rsi returned None.")
        except Exception as e:
            raise IndicatorError(f"Error calculating RSI for period {period}.") from e

        return {self.name: rsi_series}


# --- Stochastic Oscillator ---

@register_indicator(name="stoch")
class StochasticIndicator(Indicator):
    """Calculates the Stochastic Oscillator (%K and %D)."""

    @property
    def name(self) -> str:
        return "stoch"

    @property
    def description(self) -> str:
        return "Calculates the Stochastic Oscillator (%K and %D)."

    @property
    def category(self) -> str:
        return "Momentum"

    @property
    def required_input_names(self) -> List[str]:
        return ["high", "low", "close"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            IntParameter("k", 14, 1, 100, help="The look-back period for the K line."),
            IntParameter("d", 3, 1, 100, help="The smoothing period for the D line."),
            IntParameter("smooth_k", 3, 1, 100, help="The smoothing period for the K line."),
        ])

    @property
    def min_required_periods(self) -> int:
        return self.get_param("k") + self.get_param("d")

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        k = self.get_param("k")
        d = self.get_param("d")
        smooth_k = self.get_param("smooth_k")

        try:
            stoch_df = ta.stoch(data["high"], data["low"], data["close"], k=k, d=d, smooth_k=smooth_k)
            if stoch_df is None or stoch_df.empty:
                raise IndicatorError("pandas_ta.stoch returned None or an empty DataFrame.")
        except Exception as e:
            raise IndicatorError(
                f"Error calculating Stochastic with params k={k}, d={d}, smooth_k={smooth_k}."
            ) from e

        # pandas-ta returns columns like 'STOCHk_14_3_3', 'STOCHd_14_3_3'
        # We rename them to simple, predictable names.
        output = {
            "stoch_k": stoch_df.iloc[:, 0],  # The %K line
            "stoch_d": stoch_df.iloc[:, 1],  # The %D line
        }
        return output
