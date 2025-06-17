# src/optimization/algorithms/base.py

"""
Defines the abstract base class for all optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

from src.optimization.objectives.objective import ObjectiveFunction


class OptimizationResult:
    """
    A data class to hold the results of an optimization run.
    """
    def __init__(self, best_params: Dict[str, Any], best_value: float, study: Any = None):
        """
        Initializes the OptimizationResult.

        Args:
            best_params (Dict[str, Any]): The set of best parameters found.
            best_value (float): The objective function score for the best parameters.
            study (Any): The underlying study object from the optimization library
                         (e.g., an Optuna study), for more detailed analysis.
        """
        self.best_params = best_params
        self.best_value = best_value
        self.study = study

    def __repr__(self) -> str:
        return f"OptimizationResult(best_value={self.best_value}, best_params={self.best_params})"


class OptimizationAlgorithm(ABC):
    """
    An abstract base class that defines the contract for an optimization algorithm.
    """

    @abstractmethod
    def optimize(
        self,
        objective_function: ObjectiveFunction,
        search_space: Dict[str, Any],
        n_trials: int,
        callbacks: list[Callable] | None = None
    ) -> OptimizationResult:
        """
        Runs the optimization process.

        Args:
            objective_function (ObjectiveFunction): The function to be optimized.
            search_space (Dict[str, Any]): A dictionary defining the search space
                                            for the hyperparameters.
            n_trials (int): The number of trials to run.
            callbacks (list[Callable] | None): A list of callback functions to be
                                               called after each trial.

        Returns:
            OptimizationResult: An object containing the results of the optimization.
        """
        raise NotImplementedError
