# src/optimization/validation/walk_forward.py

"""
Provides an engine for performing Walk-Forward Optimization (WFO).
"""

from typing import List, Dict, Any

import pandas as pd

from src.backtesting.engine.base import BacktestEngine, BacktestResult
from src.optimization.algorithms.base import OptimizationAlgorithm
from src.optimization.objectives.objective import ObjectiveFunction
from src.optimization.search_space.builder import SearchSpaceBuilder
from src.optimization.validation.splitters import BaseTimeSeriesSplitter
from src.strategies.factory import StrategyFactory


class WalkForwardResult:
    """A data class to hold the results of a full WFO run."""
    def __init__(self, oos_results: List[BacktestResult], best_params: List[Dict[str, Any]]):
        """
        Args:
            oos_results (List[BacktestResult]): List of backtest results from each
                                                 out-of-sample (test) period.
            best_params (List[Dict[str, Any]]): List of the best parameters found
                                                for each corresponding in-sample period.
        """
        self.oos_results = oos_results
        self.best_params = best_params
        # In the future, this can include aggregated metrics, stability analysis, etc.


class WFOEngine:
    """
    Orchestrates the Walk-Forward Optimization process.

    This engine ties together the data splitter, the optimization algorithm,
    and the backtesting engine to validate a strategy's robustness over time.
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        backtest_engine: BacktestEngine,
        optimizer: OptimizationAlgorithm,
        splitter: BaseTimeSeriesSplitter,
        metric_to_optimize: str,
    ):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.backtest_engine = backtest_engine
        self.optimizer = optimizer
        self.splitter = splitter
        self.metric_to_optimize = metric_to_optimize

    def run(self, data: pd.DataFrame, n_trials: int) -> WalkForwardResult:
        """
        Executes the full Walk-Forward Optimization.

        For each split:
        1. Define the objective function for the training data.
        2. Run the optimizer on the training data to find the best parameters.
        3. Create a new strategy instance with these best parameters.
        4. Run a simple backtest with this strategy on the unseen test data.
        5. Collect the out-of-sample (OOS) results.

        Args:
            data (pd.DataFrame): The entire dataset for the WFO process.
            n_trials (int): The number of optimization trials to run per split.

        Returns:
            WalkForwardResult: An object containing the collected results.
        """
        oos_results: List[BacktestResult] = []
        best_params_per_fold: List[Dict[str, Any]] = []

        search_space = SearchSpaceBuilder.from_strategy(self.strategy_name)

        for i, (train_idx, test_idx) in enumerate(self.splitter.split(data)):
            # print(f"--- WFO Fold {i+1} ---") # Or use logger
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # 1. Define objective for this training fold
            objective_function = ObjectiveFunction(
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                data=train_data,
                backtest_engine=self.backtest_engine,
                metric_to_optimize=self.metric_to_optimize,
            )

            # 2. Run optimization on the training data
            opt_result = self.optimizer.optimize(objective_function, search_space, n_trials)
            best_params = opt_result.best_params
            best_params_per_fold.append(best_params)

            # 3. Create strategy with best params
            oos_strategy = StrategyFactory.create(
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                params=best_params,
            )
            
            # 4. Run backtest on the out-of-sample test data
            oos_result = self.backtest_engine.run(oos_strategy, test_data)
            oos_results.append(oos_result)

        return WalkForwardResult(oos_results=oos_results, best_params=best_params_per_fold)
