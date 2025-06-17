# src/backtesting/execution/fees.py

"""
Defines pluggable models for calculating transaction fees.

This module provides an abstract interface `FeeModel` and concrete
implementations for common fee structures (e.g., a fixed percentage).
This allows the backtesting engine to easily swap different fee models
to simulate various exchange or broker commission schemes.
"""

from abc import ABC, abstractmethod

from src.backtesting.execution.models import Order, Fill


class FeeModel(ABC):
    """
    An abstract base class for all transaction fee models.
    """

    @abstractmethod
    def calculate(self, order: Order, fill: Fill) -> float:
        """
        Calculates the transaction fee for a given trade.

        Args:
            order (Order): The original order that was placed.
            fill (Fill): The resulting fill from the execution.

        Returns:
            float: The calculated transaction fee in the quote currency.
                   The value should always be positive.
        """
        raise NotImplementedError


class PercentageFeeModel(FeeModel):
    """
    A fee model that calculates fees as a fixed percentage of the trade value.
    """

    def __init__(self, fee_percentage: float):
        """
        Initializes the PercentageFeeModel.

        Args:
            fee_percentage (float): The fee as a percentage (e.g., 0.1 for 0.1%).
                                    The value must be non-negative.
        """
        if fee_percentage < 0:
            raise ValueError("Fee percentage cannot be negative.")
        self.fee_rate = fee_percentage / 100.0

    def calculate(self, order: Order, fill: Fill) -> float:
        """
        Calculates the fee based on the fill's total value.
        """
        trade_value = fill.quantity * fill.price
        return trade_value * self.fee_rate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fee_percentage={self.fee_rate * 100:.4f}%)"


class NoFeeModel(FeeModel):
    """
    A fee model that applies zero fees. Useful for baseline backtests.
    """

    def calculate(self, order: Order, fill: Fill) -> float:
        """
        Returns a fee of zero.
        """
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
