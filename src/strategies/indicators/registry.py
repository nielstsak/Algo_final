# src/strategies/indicators/registry.py

"""
Provides a central registry for discovering and instantiating indicators.

This module implements the IndicatorRegistry, which uses a factory and
registration pattern to manage all available `Indicator` subclasses.
Strategies can then create indicator instances by name without needing to
import their specific modules directly, promoting loose coupling.

The registry is implemented as a singleton, exposed via a decorator for ease of use.
"""

from typing import Dict, Type, List, Any, Callable

from src.core.exceptions import IndicatorError
from .base import Indicator


class _IndicatorRegistry:
    """
    A singleton class to register and manage technical indicators.
    This class should not be used directly; interact with it via the
    `IndicatorRegistry` instance and the `@register_indicator` decorator.
    """

    def __init__(self):
        self._registry: Dict[str, Type[Indicator]] = {}

    def register(self, name: str, indicator_class: Type[Indicator]) -> None:
        """
        Registers an indicator class in the registry with a given name.

        Args:
            name (str): The unique name to register the indicator under (e.g., "sma").
            indicator_class (Type[Indicator]): The indicator class to register.

        Raises:
            TypeError: If the provided class is not a subclass of Indicator.
            ValueError: If an indicator with the same name is already registered.
        """
        if not issubclass(indicator_class, Indicator):
            raise TypeError(
                f"Class {indicator_class.__name__} is not a subclass of Indicator."
            )

        if name in self._registry:
            raise ValueError(
                f"An indicator with the name '{name}' is already registered."
            )
        self._registry[name] = indicator_class

    def create(self, name: str, **params: Any) -> Indicator:
        """
        Creates an instance of an indicator by its registered name (Factory method).

        Args:
            name (str): The name of the indicator to create.
            **params: The parameters required to initialize the indicator.

        Returns:
            Indicator: An instance of the requested indicator.

        Raises:
            IndicatorError: If the indicator name is not found or if instantiation fails.
        """
        indicator_class = self.get_indicator_class(name)
        if indicator_class is None:
            raise IndicatorError(
                f"Indicator '{name}' not found in registry. "
                f"Available indicators: {self.list_available()}"
            )

        try:
            instance = indicator_class(**params)
            # Sanity check to ensure the instance's self-reported name matches
            if instance.name != name:
                raise IndicatorError(
                    f"Instance name ('{instance.name}') does not match registered name "
                    f"('{name}'). Ensure the 'name' property in the class "
                    f"'{indicator_class.__name__}' returns the correct registered value."
                )
            return instance
        except Exception as e:
            # Catches validation errors from ParameterSet or other init issues
            raise IndicatorError(
                f"Failed to create instance of indicator '{name}' with params {params}."
            ) from e

    def get_indicator_class(self, name: str) -> Optional[Type[Indicator]]:
        """Retrieves the class for a registered indicator by name."""
        return self._registry.get(name)

    def list_available(self) -> List[str]:
        """Returns a sorted list of names of all registered indicators."""
        return sorted(list(self._registry.keys()))

    def get_indicator_details(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary with details for all registered indicators.

        Note: As `description` and `category` are instance properties on the
        base class, they cannot be retrieved here without instantiating the class,
        which requires parameters. They are therefore omitted.
        """
        details = {}
        for name, indicator_class in self._registry.items():
            try:
                params = indicator_class.get_parameters()
                details[name] = {
                    "parameters": {p.name: p.help for p in params}
                }
            except Exception as e:
                details[name] = {"error": f"Could not retrieve details: {e}"}
        return details


# Singleton instance of the registry. Import this instance to use the registry.
IndicatorRegistry = _IndicatorRegistry()


def register_indicator(name: str) -> Callable[[Type[Indicator]], Type[Indicator]]:
    """
    A class decorator to automatically register an indicator subclass.

    Args:
        name (str): The unique name to register the indicator with.

    Example:
        @register_indicator(name="sma")
        class SMAIndicator(Indicator):
            ...
    """
    def decorator(cls: Type[Indicator]) -> Type[Indicator]:
        IndicatorRegistry.register(name, cls)
        return cls
    return decorator
