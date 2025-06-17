# src/core/exceptions.py

"""
Custom exception types for the trading framework.

This module defines a hierarchy of specific exceptions to allow for more precise
error handling and clearer, more informative error messages throughout the application.
Using specific exceptions helps in debugging and makes the control flow more robust.
"""

from typing import Any, Dict, Optional


class TradingFrameworkException(Exception):
    """
    Base exception for all application-specific errors.

    Attributes:
        message (str): The primary error message.
        context (Optional[Dict[str, Any]]): A dictionary containing contextual
                                             information about the error.
    """
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initializes the TradingFrameworkException.

        Args:
            message (str): The error message.
            context (Optional[Dict[str, Any]]): A dictionary of contextual data
                                                 to aid in debugging.
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Formats the exception message with its context."""
        if not self.context:
            return self.message
        context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} (Context: {context_str})"

    def __str__(self) -> str:
        """Returns the formatted string representation of the exception."""
        return self._format_message()


# --- Core & Configuration ---

class ConfigurationError(TradingFrameworkException):
    """Raised for configuration-related errors (e.g., invalid format, missing keys)."""
    pass


# --- Data Module ---

class DataError(TradingFrameworkException):
    """Base exception for data processing and retrieval errors."""
    pass


class DataSourceError(DataError):
    """Raised for issues with external data sources (e.g., exchange API failure)."""
    pass


class NoDataFoundError(DataError):
    """Raised when expected data is missing or cannot be located in storage."""
    pass


# --- Strategies Module ---

class StrategyError(TradingFrameworkException):
    """Base exception for strategy-related errors."""
    pass


class StrategyInvalidParameterError(StrategyError):
    """Raised when a parameter provided to a strategy is invalid or out of bounds."""
    pass


class IndicatorError(StrategyError):
    """Raised for errors within an indicator calculation or configuration."""
    pass


class SignalError(StrategyError):
    """Raised for errors related to signal generation or validation."""
    pass


# --- Backtesting Module ---

class BacktestError(TradingFrameworkException):
    """Base exception for backtesting-related errors."""
    pass


class BacktestFailureError(BacktestError):
    """Raised for critical, unrecoverable failures during a backtest run."""
    pass


class PortfolioError(BacktestError):
    """Base exception for portfolio management errors (e.g., insufficient funds)."""
    pass


class RiskManagementViolationError(PortfolioError):
    """Raised when a proposed action violates established risk management rules."""
    pass


class OrderError(BacktestError):
    """Raised for errors related to order creation, validation, or lifecycle."""
    pass


class ExecutionError(BacktestError):
    """Raised for errors during the simulation of trade execution (e.g., slippage model failure)."""
    pass


# --- Optimization Module ---

class OptimizationError(TradingFrameworkException):
    """Base exception for optimization-related errors."""
    pass


class WFOSplitError(OptimizationError):
    """Raised for errors during the splitting of data for Walk-Forward Optimization."""
    pass


class ObjectiveFunctionError(OptimizationError):
    """Raised for errors during the calculation of the optimization objective function."""
    pass


class SearchSpaceError(OptimizationError):
    """Raised for errors related to the definition or validation of the optimization search space."""
    pass
