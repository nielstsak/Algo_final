# src/optimization/objectives/objective.py

"""
Defines the objective function for hyperparameter optimization.
"""

from typing import Dict, Any

import pandas as pd

from src.backtesting.analysis.analyzer import BacktestAnalyzer
from src.backtesting.engine.base import BacktestEngine
from src.core.exceptions import ObjectiveFunctionError
from src.strategies.factory import StrategyFactory


class ObjectiveFunction:
    """
    A class that encapsulates the logic for evaluating a single trial in an
    optimization process.
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        data: pd.DataFrame,
        backtest_engine: BacktestEngine,
        metric_to_optimize: str,
        higher_is_better: bool = True
    ):
        """
        Initializes the ObjectiveFunction.

        Args:
            strategy_name (str): The name of the strategy to optimize.
            symbol (str): The trading symbol to use for the backtest.
            data (pd.DataFrame): The historical data for the backtest.
            backtest_engine (BacktestEngine): The backtest engine instance to run simulations.
            metric_to_optimize (str): The name of the metric to use as the optimization score.
            higher_is_better (bool): True if a higher metric score is better, False otherwise.
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.data = data
        self.backtest_engine = backtest_engine
        self.metric_to_optimize = metric_to_optimize
        # Multiplier to handle maximization vs. minimization
        self.optimization_direction = 1.0 if higher_is_better else -1.0

    def __call__(self, params: Dict[str, Any]) -> float:
        """
        Executes one evaluation of the objective function with a given set of parameters.

        This method is designed to be called by an optimization library like Optuna.

        Args:
            params (Dict[str, Any]): A dictionary of hyperparameter values for this trial.

        Returns:
            float: The score of this trial. The optimizer will try to maximize this value.
        """
        try:
            # 1. Create a strategy instance with the trial's parameters
            strategy_instance = StrategyFactory.create(
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                params=params
            )

            # 2. Run the backtest
            result = self.backtest_engine.run(strategy_instance, self.data)

            # 3. Analyze the results
            analyzer = BacktestAnalyzer(result)
            report = analyzer.generate_report()

            # 4. Extract the score
            score = report.get(self.metric_to_optimize)
            if score is None:
                raise ObjectiveFunctionError(
                    f"Metric '{self.metric_to_optimize}' not found in backtest report."
                )
            if not isinstance(score, (int, float)):
                 raise ObjectiveFunctionError(
                    f"Metric '{self.metric_to_optimize}' must be a number, but got {type(score)}."
                )
            
            # 5. Return the score, adjusted for optimization direction
            return float(score) * self.optimization_direction

        except Exception as e:
            # Optuna and other libraries can handle failed trials gracefully if they return a poor score.
            # We return a very bad score to signal that this parameter set should be avoided.
            # logger.warning(f"Trial failed for params {params}: {e}")
            return -1e9 # Return a very large negative number for failed trials
