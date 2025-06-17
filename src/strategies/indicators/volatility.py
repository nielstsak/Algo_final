# src/strategies/indicators/technical/volatility.py

"""
Implementations of standard volatility-based technical indicators.

This module provides concrete implementations of common volatility indicators
like Average True Range (ATR) and Bollinger Bands, built upon the base
`Indicator` interface.

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
from src.strategies.parameters import ParameterSet, IntParameter, FloatParameter


# --- Average True Range (ATR) ---

@register_indicator(name="atr")
class ATRIndicator(Indicator):
    """Calculates the Average True Range (ATR)."""

    @property
    def name(self) -> str:
        return "atr"

    @property
    def description(self) -> str:
        return "Calculates the Average True Range (ATR)."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def required_input_names(self) -> List[str]:
        return ["high", "low", "close"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            IntParameter(
                name="period",
                default=14,
                min_value=2,
                max_value=100,
                help="The time period for ATR calculation."
            )
        ])

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.get_param("period")
        try:
            atr_series = ta.atr(data["high"], data["low"], data["close"], length=period)
            if atr_series is None:
                raise IndicatorError("pandas_ta.atr returned None.")
        except Exception as e:
            raise IndicatorError(f"Error calculating ATR for period {period}.") from e

        return {self.name: atr_series}


# --- Bollinger Bands ---

@register_indicator(name="bbands")
class BollingerBandsIndicator(Indicator):
    """Calculates Bollinger Bands."""

    @property
    def name(self) -> str:
        return "bbands"

    @property
    def description(self) -> str:
        return "Calculates Bollinger Bands (upper, middle, lower)."

    @property
    def category(self) -> str:
        return "Volatility"

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
                max_value=100,
                help="The time period for the moving average."
            ),
            FloatParameter(
                name="std_dev",
                default=2.0,
                min_value=0.5,
                max_value=5.0,
                help="The number of standard deviations for the bands."
            )
        ])

    @property
    def min_required_periods(self) -> int:
        return self.get_param("period")

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.get_param("period")
        std_dev = self.get_param("std_dev")

        try:
            bbands_df = ta.bbands(data["close"], length=period, std=std_dev)
            if bbands_df is None or bbands_df.empty:
                raise IndicatorError("pandas_ta.bbands returned None or an empty DataFrame.")
        except Exception as e:
            raise IndicatorError(
                f"Error calculating Bollinger Bands with period={period}, std_dev={std_dev}."
            ) from e

        # pandas-ta returns columns like 'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0', etc.
        # We rename them to simple, predictable names.
        output = {
            "bbands_upper": bbands_df.iloc[:, 2],  # Upper band
            "bbands_middle": bbands_df.iloc[:, 1], # Middle band (SMA)
            "bbands_lower": bbands_df.iloc[:, 0],  # Lower band
        }
        return output
