# src/optimization/algorithms/optuna.py

"""
A concrete implementation of the OptimizationAlgorithm interface using Optuna.
"""

from typing import Dict, Any, Callable, List

import optuna
from optuna import Trial

from src.core.exceptions import OptimizationError
from src.optimization.algorithms.base import OptimizationAlgorithm, OptimizationResult
from src.optimization.objectives.objective import ObjectiveFunction


class OptunaOptimization(OptimizationAlgorithm):
    """
    An optimization algorithm that uses the Optuna framework to find the best
    hyperparameters for a strategy.
    """

    def __init__(self, study_name: str, direction: str = "maximize", storage: str | None = None):
        """
        Initializes the Optuna optimization algorithm.

        Args:
            study_name (str): The name for the Optuna study.
            direction (str): The direction of optimization ("maximize" or "minimize").
            storage (str | None): The database URL for study storage (e.g., "sqlite:///db.sqlite3").
                                  If None, an in-memory storage is used.
        """
        self.study_name = study_name
        self.direction = direction
        self.storage = storage

    def optimize(
        self,
        objective_function: ObjectiveFunction,
        search_space: Dict[str, Any],
        n_trials: int,
        callbacks: list[Callable] | None = None
    ) -> OptimizationResult:
        """
        Runs the optimization process using Optuna.
        """
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                direction=self.direction
            )

            # Wrapper function to be passed to Optuna's optimize method
            def objective_wrapper(trial: Trial) -> float:
                params = self._suggest_params(trial, search_space)
                return objective_function(params)

            study.optimize(
                objective_wrapper,
                n_trials=n_trials,
                callbacks=callbacks
            )

            # The best_value from Optuna needs to be un-adjusted if we are minimizing
            # by maximizing the negative. The ObjectiveFunction already handles this.
            best_value = study.best_value

            return OptimizationResult(
                best_params=study.best_params,
                best_value=best_value,
                study=study
            )

        except Exception as e:
            raise OptimizationError("Optuna optimization process failed.") from e

    def _suggest_params(self, trial: Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates the search space definition into Optuna's `trial.suggest_*` calls.
        """
        params = {}
        for name, spec in search_space.items():
            param_type = spec.get("type")
            if param_type == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
            elif param_type == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise OptimizationError(f"Unsupported parameter type '{param_type}' in search space for '{name}'.")
        return params
