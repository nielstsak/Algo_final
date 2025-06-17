# src/strategies/implementations/psar_reversal_otoco_strategy_impl.py

"""
A reversal strategy based on the Parabolic SAR (PSAR) indicator.
"""

from typing import Dict, List, Any

import pandas as pd
import pandas_ta as ta

from src.core.exceptions import IndicatorError
from src.strategies.base import BaseStrategy
from src.strategies.indicators.base import Indicator
from src.strategies.indicators.registry import register_indicator, IndicatorRegistry
from src.strategies.parameters import (
    ParameterSet,
    FloatParameter,
    IntParameter,
    BoolParameter,
)
from src.strategies.signals import (
    Signal,
    SignalDirection,
    SignalType,
)


# NOTE: The PSAR indicator is defined here temporarily for self-containment,
# as it was not in the initial indicator files. In a final structure, this
# would be in its own file (e.g., `strategies/indicators/technical/trend.py`).
@register_indicator(name="psar")
class PSARIndicator(Indicator):
    """Calculates the Parabolic Stop and Reverse (PSAR)."""

    @property
    def name(self) -> str:
        return "psar"
    @property
    def description(self) -> str:
        return "Calculates Parabolic SAR and its reversal points."
    @property
    def category(self) -> str:
        return "Trend"
    @property
    def required_input_names(self) -> List[str]:
        return ["high", "low"]

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        return ParameterSet([
            FloatParameter("initial_af", 0.02, 0.01, 0.1, help="Initial acceleration factor."),
            FloatParameter("af_increment", 0.02, 0.01, 0.1, help="Acceleration factor increment."),
            FloatParameter("max_af", 0.2, 0.1, 0.5, help="Maximum acceleration factor."),
        ])

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        try:
            psar_df = ta.psar(
                data["high"],
                data["low"],
                af0=self.get_param("initial_af"),
                af=self.get_param("af_increment"),
                max_af=self.get_param("max_af"),
            )
            if psar_df is None or psar_df.empty:
                raise IndicatorError("pandas_ta.psar returned None or an empty DataFrame.")
        except Exception as e:
            raise IndicatorError("Error calculating PSAR.") from e
        
        # We are interested in the reversal column `PSARr...`
        reversal_col = next((col for col in psar_df.columns if 'PSARr' in col), None)
        if reversal_col is None:
            raise IndicatorError("Could not find reversal column in PSAR output.")

        return {"psar_reversal": psar_df[reversal_col]}


class PSARReversalStrategy(BaseStrategy):
    """
    Implements a strategy based on PSAR reversals for entries.
    It uses ATR to set a Stop-Loss and Take-Profit (OTOCO).
    """
    def __init__(self, symbol: str, params: Dict[str, Any]):
        self.params = self.get_parameters().validate(params)
        self.symbol = symbol

    @property
    def name(self) -> str:
        return "PSARReversal"

    @property
    def description(self) -> str:
        return "A reversal strategy using PSAR flips and ATR for SL/TP."

    @classmethod
    def get_parameters(cls) -> ParameterSet:
        """Defines the parameters for this strategy."""
        return ParameterSet([
            # PSAR parameters
            FloatParameter("initial_af", 0.02, 0.01, 0.1, help="PSAR Initial acceleration factor."),
            FloatParameter("af_increment", 0.02, 0.01, 0.1, help="PSAR Acceleration factor increment."),
            FloatParameter("max_af", 0.2, 0.1, 0.5, help="PSAR Maximum acceleration factor."),
            # OTOCO (SL/TP) parameters
            BoolParameter("use_otoco", True, help="Whether to use ATR for SL/TP calculation."),
            IntParameter("atr_period", 14, 5, 50, help="Period for the ATR calculation."),
            FloatParameter("atr_multiplier_sl", 2.0, 1.0, 5.0, help="ATR multiplier for stop-loss."),
            FloatParameter("atr_multiplier_tp", 4.0, 1.0, 10.0, help="ATR multiplier for take-profit."),
        ])

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates all indicators required by the strategy."""
        indicators = {}
        
        psar_indicator = IndicatorRegistry.create(
            "psar",
            initial_af=self.params["initial_af"],
            af_increment=self.params["af_increment"],
            max_af=self.params["max_af"],
        )
        indicators.update(psar_indicator.calculate(data))
        
        if self.params["use_otoco"]:
            atr_indicator = IndicatorRegistry.create("atr", period=self.params["atr_period"])
            indicators.update(atr_indicator.calculate(data))
            
        return indicators

    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> List[Signal]:
        """Generates trading signals based on PSAR reversals."""
        signals = []
        reversals = indicators["psar_reversal"]

        long_condition = (reversals == 1)
        short_condition = (reversals == -1)

        long_entry_points = data.loc[long_condition]
        short_entry_points = data.loc[short_condition]

        for timestamp, row in long_entry_points.iterrows():
            stop_loss, take_profit = None, None
            if self.params["use_otoco"] and "atr" in indicators:
                atr_value = indicators["atr"].get(timestamp)
                if atr_value is not None:
                    stop_loss = row["close"] - (atr_value * self.params["atr_multiplier_sl"])
                    take_profit = row["close"] + (atr_value * self.params["atr_multiplier_tp"])
            
            signals.append(Signal(
                strategy_name=self.name, symbol=self.symbol, timestamp=timestamp,
                direction=SignalDirection.LONG, signal_type=SignalType.ENTRY,
                price=row["close"], stop_loss=stop_loss, take_profit=take_profit,
                metadata={"psar_reversal": 1}
            ))

        for timestamp, row in short_entry_points.iterrows():
            stop_loss, take_profit = None, None
            if self.params["use_otoco"] and "atr" in indicators:
                atr_value = indicators["atr"].get(timestamp)
                if atr_value is not None:
                    stop_loss = row["close"] + (atr_value * self.params["atr_multiplier_sl"])
                    take_profit = row["close"] - (atr_value * self.params["atr_multiplier_tp"])

            signals.append(Signal(
                strategy_name=self.name, symbol=self.symbol, timestamp=timestamp,
                direction=SignalDirection.SHORT, signal_type=SignalType.ENTRY,
                price=row["close"], stop_loss=stop_loss, take_profit=take_profit,
                metadata={"psar_reversal": -1}
            ))

        signals.sort(key=lambda s: s.timestamp)
        return signals
