# src/backtesting/metrics/trades.py

"""
Implementations of metrics related to trade analysis.

These metrics typically require a list of closed, round-trip trades.
As a simple vectorized backtest does not easily provide this, some metrics
here are placeholders that depend on a future trade reconstruction layer.
"""

from typing import Any

from src.backtesting.metrics.base import Metric
from src.backtesting.metrics.registry import register_metric
from src.backtesting.portfolio.portfolio import Portfolio


@register_metric(name="total_transactions")
class TotalTransactionsMetric(Metric):
    """Calculates the total number of transactions (fills)."""
    @property
    def name(self) -> str:
        return "total_transactions"
    @property
    def description(self) -> str:
        return "The total number of individual executed buy/sell transactions (fills)."
    @property
    def category(self) -> str:
        return "Trades"

    def calculate(self, portfolio: Portfolio, **kwargs) -> int:
        """
        This metric simply counts the number of Fill objects.
        It does not represent round-trip trades.
        """
        return len(portfolio.transaction_history)


# NOTE: The metrics below are placeholders. A full implementation requires
# a trade analysis component that can reconstruct round-trip trades from
# the flat list of transaction fills in the portfolio.

@register_metric(name="win_rate")
class WinRateMetric(Metric):
    """
    Calculates the win rate of closed trades. (Placeholder)
    """
    @property
    def name(self) -> str:
        return "win_rate"
    @property
    def description(self) -> str:
        return "The percentage of profitable trades out of all closed trades. (Not implemented)"
    @property
    def category(self) -> str:
        return "Trades"

    def calculate(self, portfolio: Portfolio, **kwargs) -> float:
        """
        This metric requires a list of closed trades with calculated PnL.
        Returning 0.0 as a placeholder.
        """
        # A real implementation would iterate through a list of closed trades,
        # count the number of trades with PnL > 0, and divide by the total.
        return 0.0


@register_metric(name="profit_factor")
class ProfitFactorMetric(Metric):
    """
    Calculates the profit factor. (Placeholder)
    """
    @property
    def name(self) -> str:
        return "profit_factor"
    @property
    def description(self) -> str:
        return "The ratio of gross profits to gross losses. (Not implemented)"
    @property
    def category(self) -> str:
        return "Trades"

    def calculate(self, portfolio: Portfolio, **kwargs) -> float:
        """
        This metric requires summing profits from winning trades and losses
        from losing trades. Returning 0.0 as a placeholder.
        """
        # A real implementation would calculate gross_profit and gross_loss
        # from a list of closed trades and return gross_profit / abs(gross_loss).
        return 0.0
