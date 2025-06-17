# src/backtesting/metrics/registry.py

"""
Provides a central registry for discovering and instantiating metrics.

This module implements the MetricRegistry, which uses a factory and
registration pattern to manage all available `Metric` subclasses. This allows
an analysis engine to dynamically calculate a suite of metrics without needing
to import their specific modules directly.
"""

from typing import Dict, Type, List, Any, Callable, Optional

from src.backtesting.metrics.base import Metric
from src.core.exceptions import BacktestError


class _MetricRegistry:
    """
    A singleton class to register and manage performance metrics.
    This class should not be used directly; interact with it via the
    `MetricRegistry` instance and the `@register_metric` decorator.
    """

    def __init__(self):
        self._registry: Dict[str, Type[Metric]] = {}

    def register(self, name: str, metric_class: Type[Metric]) -> None:
        """
        Registers a metric class in the registry with a given name.

        Args:
            name (str): The unique name to register the metric under (e.g., "sharpe_ratio").
            metric_class (Type[Metric]): The metric class to register.
        """
        if not issubclass(metric_class, Metric):
            raise TypeError(f"Class {metric_class.__name__} is not a subclass of Metric.")

        if name in self._registry:
            raise ValueError(f"A metric with the name '{name}' is already registered.")
        self._registry[name] = metric_class

    def create(self, name: str) -> Metric:
        """
        Creates an instance of a metric by its registered name (Factory method).

        Args:
            name (str): The name of the metric to create.

        Returns:
            Metric: An instance of the requested metric.
        """
        metric_class = self.get_metric_class(name)
        if metric_class is None:
            raise BacktestError(
                f"Metric '{name}' not found in registry. "
                f"Available metrics: {self.list_available()}"
            )
        try:
            return metric_class()
        except Exception as e:
            raise BacktestError(f"Failed to create instance of metric '{name}'.") from e

    def get_metric_class(self, name: str) -> Optional[Type[Metric]]:
        """Retrieves the class for a registered metric by name."""
        return self._registry.get(name)

    def list_available(self) -> List[str]:
        """Returns a sorted list of names of all registered metrics."""
        return sorted(list(self._registry.keys()))

    def list_by_category(self) -> Dict[str, List[str]]:
        """Returns a dictionary of metrics grouped by category."""
        by_category: Dict[str, List[str]] = {}
        for name, metric_class in self._registry.items():
            # We need to instantiate the class to get its category property
            instance = metric_class()
            category = instance.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(name)
        return by_category


# Singleton instance of the registry. Import this instance to use the registry.
MetricRegistry = _MetricRegistry()


def register_metric(name: str) -> Callable[[Type[Metric]], Type[Metric]]:
    """
    A class decorator to automatically register a metric subclass.

    Args:
        name (str): The unique name to register the metric with.

    Example:
        @register_metric(name="sharpe_ratio")
        class SharpeRatioMetric(Metric):
            ...
    """
    def decorator(cls: Type[Metric]) -> Type[Metric]:
        MetricRegistry.register(name, cls)
        return cls
    return decorator
