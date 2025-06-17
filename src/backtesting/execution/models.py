# src/backtesting/execution/models.py

"""
Defines immutable data structures for the trade execution lifecycle.

This module provides clear, type-safe, and immutable dataclasses for
representing Orders, Fills, and Positions. These structures are the fundamental
building blocks for both event-driven and vectorized backtesting engines, as
well as for live trading execution handlers.
"""

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

from src.core.exceptions import OrderError


class OrderSide(str, Enum):
    """Enumerates the side of an order."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Enumerates the type of an order."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Enumerates the possible statuses of an order."""
    PENDING = "PENDING"      # The order has been created but not yet sent.
    OPEN = "OPEN"            # The order is active on the exchange.
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class Order:
    """
    An immutable representation of a trading order.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Required for LIMIT orders
    stop_price: Optional[float] = None # Required for STOP_LIMIT orders
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        """Performs validation after initialization."""
        if self.quantity <= 0:
            raise OrderError("Order quantity must be positive.", context={"quantity": self.quantity})
        
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise OrderError("A price must be specified for LIMIT orders.")
        
        if self.order_type == OrderType.STOP_LIMIT and (self.price is None or self.stop_price is None):
            raise OrderError("Both price and stop_price must be specified for STOP_LIMIT orders.")
            
        if self.price is not None and self.price <= 0:
            raise OrderError("Order price must be positive.", context={"price": self.price})

        if self.stop_price is not None and self.stop_price <= 0:
            raise OrderError("Order stop_price must be positive.", context={"stop_price": self.stop_price})


@dataclass(frozen=True)
class Fill:
    """
    An immutable representation of a single trade execution (a fill).

    A single order can result in multiple fills.
    """
    symbol: str
    timestamp: datetime.datetime
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_currency: str
    order_id: str
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        """Performs validation after initialization."""
        if self.quantity <= 0:
            raise OrderError("Fill quantity must be positive.", context={"quantity": self.quantity})
        if self.price <= 0:
            raise OrderError("Fill price must be positive.", context={"price": self.price})
        if self.fee < 0:
            raise OrderError("Fill fee cannot be negative.", context={"fee": self.fee})


@dataclass(frozen=True)
class Position:
    """
    An immutable snapshot of an open position for a single asset.
    """
    symbol: str
    quantity: float  # Positive for long, negative for short
    average_entry_price: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)

    @property
    def is_long(self) -> bool:
        """Returns True if the position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Returns True if the position is short."""
        return self.quantity < 0

    @property
    def market_value(self) -> float:
        """Calculates the absolute market value of the position."""
        return abs(self.quantity) * self.average_entry_price

    def __post_init__(self):
        """Performs validation after initialization."""
        if self.quantity == 0:
            raise ValueError("Position quantity cannot be zero.")
        if self.average_entry_price <= 0:
            raise ValueError("Position average_entry_price must be positive.")
