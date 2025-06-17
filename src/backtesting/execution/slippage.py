# src/backtesting/execution/slippage.py

"""
Defines pluggable models for simulating price slippage.

Slippage is the difference between the expected price of a trade and the price
at which the trade is actually executed. This module provides an abstract
interface `SlippageModel` and concrete implementations for common slippage
scenarios. This allows the backtesting engine to simulate more realistic
market conditions.
"""

from abc import ABC, abstractmethod

import pandas as pd

from src.backtesting.execution.models import Order, OrderSide
from src.core.exceptions import ExecutionError


class SlippageModel(ABC):
    """
    An abstract base class for all price slippage models.
    """

    @abstractmethod
    def calculate_fill_price(self, order: Order, market_price: float) -> float:
        """
        Adjusts the market price to account for slippage.

        Args:
            order (Order): The order being executed.
            market_price (float): The ideal market price at the moment of execution
                                  (e.g., the 'close' price of the current bar).

        Returns:
            float: The adjusted fill price after applying slippage.
        """
        raise NotImplementedError


class PercentageSlippageModel(SlippageModel):
    """
    A slippage model that adjusts the fill price by a fixed percentage.

    For BUY orders, the price is increased. For SELL orders, it is decreased.
    This simulates an unfavorable price movement caused by the order's impact
    on the market.
    """

    def __init__(self, slippage_percentage: float):
        """
        Initializes the PercentageSlippageModel.

        Args:
            slippage_percentage (float): The slippage as a percentage (e.g., 0.05 for 0.05%).
                                         The value must be non-negative.
        """
        if slippage_percentage < 0:
            raise ValueError("Slippage percentage cannot be negative.")
        self.slippage_rate = slippage_percentage / 100.0

    def calculate_fill_price(self, order: Order, market_price: float) -> float:
        """
        Applies a percentage-based slippage to the market price.
        """
        if order.side == OrderSide.BUY:
            # For a buy, slippage increases the price
            return market_price * (1 + self.slippage_rate)
        elif order.side == OrderSide.SELL:
            # For a sell, slippage decreases the price
            return market_price * (1 - self.slippage_rate)
        else:
            # Should not happen with valid OrderSide enum
            raise ExecutionError(f"Unknown order side: {order.side}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(slippage_percentage={self.slippage_rate * 100:.4f}%)"


class NoSlippageModel(SlippageModel):
    """
    A model that simulates zero slippage. The fill price is the market price.
    Useful for ideal-condition backtests.
    """

    def calculate_fill_price(self, order: Order, market_price: float) -> float:
        """
        Returns the market price unmodified.
        """
        return market_price

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
