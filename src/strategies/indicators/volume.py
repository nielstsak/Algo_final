# src/strategies/indicators/technical/volume.py

"""
Implementations of standard volume-based technical indicators.

This module provides concrete implementations of common volume indicators
like On-Balance Volume (OBV), built upon the base `Indicator` interface.

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
from src.strategies.parameters import ParameterSet


# --- On-Balance Volume (OBV) ---

@register_indicator(name="obv")
class OBVIndicator(Indicator):
    """Calculates the On-Balance Volume (OBV)."""

    @property
    def name(self) -> str:
        return "obv"

    @property
    def description(self) -> str:
        return "Calculates the On-Balance Volume (OBV)."

    @property
    def category(self) -> str:
        return "Volume"

    @property
    def required_input_names(self) -> List[str]:
        return ["close", "volume"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        """OBV typically does not require any parameters."""
        return ParameterSet([])

    @property
    def min_required_periods(self) -> int:
        """OBV can be calculated from the first period."""
        return 1

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        try:
            obv_series = ta.obv(data["close"], data["volume"])
            if obv_series is None:
                raise IndicatorError("pandas_ta.obv returned None.")
        except Exception as e:
            raise IndicatorError("Error calculating OBV.") from e

        return {self.name: obv_series}
