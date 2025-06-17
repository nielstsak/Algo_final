# src/optimization/search_space/builder.py

"""
Provides a builder for creating optimization search spaces from strategies.
"""

from typing import Dict, Any

from src.core.exceptions import SearchSpaceError, StrategyError
from src.strategies.factory import StrategyFactory


class SearchSpaceBuilder:
    """
    A class dedicated to building search space definitions for optimization
    libraries like Optuna.
    """

    @staticmethod
    def from_strategy(strategy_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Generates the search space for a given strategy.

        This method fetches the strategy's `ParameterSet` and instructs it to
        generate a search space definition compatible with optimization tools.

        Args:
            strategy_name (str): The name of the strategy for which to build
                                 the search space.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary defining the search space,
                                       ready to be used by an optimizer.

        Raises:
            SearchSpaceError: If the search space cannot be generated.
            StrategyError: If the strategy is not found.
        """
        try:
            # Use the StrategyFactory to get the ParameterSet for the strategy
            parameter_set = StrategyFactory.get_strategy_parameters(strategy_name)

            # The ParameterSet itself knows how to generate the space definition
            return parameter_set.generate_optuna_space()

        except StrategyError:
            # Re-raise the original, more specific error
            raise
        except Exception as e:
            raise SearchSpaceError(
                f"Failed to build search space for strategy '{strategy_name}'."
            ) from e

