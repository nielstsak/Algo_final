# src/backtesting/portfolio/portfolio.py

"""
Defines the Portfolio class, responsible for managing the state of assets,
positions, and cash during a backtest.
"""

import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.backtesting.execution.models import Fill, Position, OrderSide
from src.core.exceptions import PortfolioError


class Portfolio:
    """
    Manages the state of a trading portfolio throughout a backtest.

    The Portfolio is a stateful object that tracks cash, open positions, and
    the history of transactions. It provides methods to update its state based
    on trade executions (fills) and to calculate its value over time.
    """

    def __init__(self, initial_cash: float, quote_currency: str = "USD"):
        """
        Initializes the Portfolio.

        Args:
            initial_cash (float): The starting cash balance.
            quote_currency (str): The currency for accounting (e.g., "USDT", "USD").
        """
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive.")

        self.initial_cash = initial_cash
        self.quote_currency = quote_currency
        self.cash = initial_cash

        self._positions: Dict[str, Position] = {}  # symbol -> Position
        self.transaction_history: List[Fill] = []
        
        # Track equity over time
        self._equity_curve: Dict[datetime.datetime, float] = {}
        self._last_equity_update_time: Optional[datetime.datetime] = None

    def get_position(self, symbol: str) -> Optional[Position]:
        """Retrieves the current open position for a given symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Returns a list of all current open positions."""
        return list(self._positions.values())

    def update_from_fill(self, fill: Fill) -> None:
        """
        Updates the portfolio's state based on a new trade execution (fill).

        This is the core method that adjusts cash and positions.

        Args:
            fill (Fill): The Fill object representing the trade.
        """
        self.transaction_history.append(fill)
        
        # --- Update Cash ---
        trade_value = fill.quantity * fill.price
        if fill.side == OrderSide.BUY:
            self.cash -= trade_value
        else:  # SELL
            self.cash += trade_value
        self.cash -= fill.fee

        if self.cash < 0:
            raise PortfolioError(
                "Cash balance fell below zero. Possible over-leveraging or insufficient funds.",
                context={"cash": self.cash, "fill_id": fill.fill_id}
            )

        # --- Update Position ---
        current_position = self.get_position(fill.symbol)
        
        if fill.side == OrderSide.BUY:
            if current_position is None:
                # Open a new long position
                new_position = Position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    average_entry_price=fill.price,
                    timestamp=fill.timestamp
                )
                self._positions[fill.symbol] = new_position
            elif current_position.is_long:
                # Increase existing long position
                new_quantity = current_position.quantity + fill.quantity
                new_avg_price = (
                    (current_position.market_value + trade_value) / new_quantity
                )
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=new_quantity,
                    average_entry_price=new_avg_price,
                    timestamp=fill.timestamp
                )
            else: # is_short
                # Reduce or close existing short position
                self._close_or_reduce_short(fill, trade_value)

        else:  # SELL
            if current_position is None:
                # Open a new short position
                new_position = Position(
                    symbol=fill.symbol,
                    quantity=-fill.quantity,
                    average_entry_price=fill.price,
                    timestamp=fill.timestamp
                )
                self._positions[fill.symbol] = new_position
            elif current_position.is_short:
                 # Increase existing short position
                new_quantity = current_position.quantity - fill.quantity
                new_avg_price = (
                    (abs(current_position.market_value) + trade_value) / abs(new_quantity)
                )
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=new_quantity,
                    average_entry_price=new_avg_price,
                    timestamp=fill.timestamp
                )
            else: # is_long
                # Reduce or close existing long position
                self._close_or_reduce_long(fill, trade_value)
                
        # Update the equity at the time of the transaction
        self.update_equity(fill.timestamp)

    def _close_or_reduce_long(self, fill: Fill, trade_value: float):
        """Helper to handle selling from a long position."""
        pos = self._positions[fill.symbol]
        if fill.quantity > pos.quantity:
            raise PortfolioError("Attempted to sell more than the quantity held in a long position.")
        
        new_quantity = pos.quantity - fill.quantity
        if new_quantity == 0:
            # Position closed
            del self._positions[fill.symbol]
        else:
            # Position reduced
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=new_quantity,
                average_entry_price=pos.average_entry_price, # Avg price doesn't change on partial close
                timestamp=fill.timestamp
            )

    def _close_or_reduce_short(self, fill: Fill, trade_value: float):
        """Helper to handle buying to cover a short position."""
        pos = self._positions[fill.symbol]
        if fill.quantity > abs(pos.quantity):
            raise PortfolioError("Attempted to buy more than the quantity held in a short position.")
        
        new_quantity = pos.quantity + fill.quantity
        if new_quantity == 0:
            # Position closed
            del self._positions[fill.symbol]
        else:
            # Position reduced
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=new_quantity,
                average_entry_price=pos.average_entry_price,
                timestamp=fill.timestamp
            )

    def update_equity(self, timestamp: datetime.datetime, market_prices: Optional[Dict[str, float]] = None):
        """
        Calculates and records the total portfolio value (equity) at a given timestamp.
        """
        # To avoid redundant calculations, only update if time has passed
        if timestamp == self._last_equity_update_time:
            return

        holdings_value = 0.0
        if market_prices:
            for symbol, position in self._positions.items():
                current_price = market_prices.get(symbol)
                if current_price:
                    # For long positions, value is qty * price.
                    # For short positions, value is market_value + (entry_price - current_price) * abs(qty)
                    if position.is_long:
                        holdings_value += position.quantity * current_price
                    else: # is_short
                        pnl = (position.average_entry_price - current_price) * abs(position.quantity)
                        holdings_value += position.market_value + pnl
                else:
                    # If price is not available, use the average entry price as a fallback
                    holdings_value += position.market_value
        else:
            # If no market prices are provided, value holdings at their entry cost
            holdings_value = sum(pos.market_value for pos in self._positions.values())

        self._equity_curve[timestamp] = self.cash + holdings_value
        self._last_equity_update_time = timestamp
    
    def get_equity_curve(self) -> pd.Series:
        """
        Returns the portfolio's equity curve as a pandas Series.
        """
        if not self._equity_curve:
            return pd.Series(dtype=float)
        
        series = pd.Series(self._equity_curve)
        series.sort_index(inplace=True)
        return series
