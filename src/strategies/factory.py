# src/strategies/factory.py

"""
A factory for discovering and instantiating trading strategies.

This module provides a centralized mechanism to create strategy instances from
configuration data, without needing to hardcode imports for each strategy class.
It automatically discovers all available strategies defined in the
`strategies.implementations` package.
"""

import importlib
import inspect
import pkgutil
from typing import Dict, Type, List, Any, Optional

from src.core.exceptions import StrategyError
from src.strategies.base import BaseStrategy
from src.strategies.implementations import (
    bbands_volume_rsi_strategy_impl,
    psar_reversal_otoco_strategy_impl,
    sma_cross,
    strategy_template,
    triple_ma_anticipation_strategy_impl,
)


class _StrategyFactory:
    """
    A singleton factory to discover, manage, and instantiate strategy classes.
    This class should not be used directly; interact with it via the
    `StrategyFactory` instance.
    """

    def __init__(self):
        self._registry: Dict[str, Type[BaseStrategy]] = {}
        self.discover_strategies()

    def discover_strategies(self):
        """
        Automatically discovers and registers all BaseStrategy subclasses
        from the `strategies.implementations` package.
        """
        # A list of modules to scan for strategies. This is more explicit
        # than dynamic path scanning and avoids complex import issues.
        implementations_modules = [
            bbands_volume_rsi_strategy_impl,
            psar_reversal_otoco_strategy_impl,
            sma_cross,
            strategy_template,
            triple_ma_anticipation_strategy_impl
        ]

        for module in implementations_modules:
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj) and
                    issubclass(obj, BaseStrategy) and
                    obj is not BaseStrategy
                ):
                    # Use the strategy's own `name` property as the key
                    # We instantiate it without params just to get the name.
                    # This relies on `name` being a property on the class itself.
                    try:
                        strategy_name = obj.name.fget(obj) 
                        if strategy_name in self._registry:
                             # This can happen if names are duplicated across files
                            continue
                        self._registry[strategy_name] = obj
                    except Exception:
                        # Could fail if a strategy is badly defined.
                        # We can log this event if a logger is available.
                        continue

    def create(
        self,
        strategy_name: str,
        symbol: str,
        params: Dict[str, Any]
    ) -> BaseStrategy:
        """
        Instantiates a strategy by its registered name.

        Args:
            strategy_name (str): The name of the strategy to create.
            symbol (str): The trading symbol for the strategy instance.
            params (Dict[str, Any]): The parameters for the strategy.

        Returns:
            BaseStrategy: An instance of the requested strategy.

        Raises:
            StrategyError: If the strategy is not found or if instantiation fails due
                           to invalid parameters.
        """
        strategy_class = self.get_strategy_class(strategy_name)
        if not strategy_class:
            raise StrategyError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {self.list_available()}"
            )

        try:
            # The strategy __init__ is now expected to handle validation
            return strategy_class(symbol=symbol, params=params)
        except Exception as e:
            # This will catch validation errors from ParameterSet
            raise StrategyError(
                f"Failed to create instance of strategy '{strategy_name}'."
            ) from e

    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Retrieves a strategy class by its registered name."""
        return self._registry.get(name)

    def list_available(self) -> List[str]:
        """Returns a sorted list of names of all registered strategies."""
        return sorted(list(self._registry.keys()))

    def get_strategy_parameters(self, name: str) -> ParameterSet:
        """
        Returns the ParameterSet for a given strategy.

        Raises:
            StrategyError: If the strategy name is not found.
        """
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            raise StrategyError(f"Strategy '{name}' not found.")
        return strategy_class.get_parameters()


# Singleton instance of the factory. Import this to use it.
StrategyFactory = _StrategyFactory()
