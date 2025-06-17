# src/optimization/optimizer.py

"""
Provides a high-level orchestrator for running strategy optimizations.

This module contains the `StrategyOptimizer` class, which serves as a simplified
interface (facade) for executing complex optimization workflows like simple
hyperparameter tuning or Walk-Forward Optimization (WFO). It abstracts away the
details of wiring together the various components (engines, algorithms, data splitters).
"""

from typing import Dict, Any

import pandas as pd

from src.backtesting.engine.base import BacktestEngine
from src.backtesting.engine.vectorized import VectorizedBacktestEngine, VectorizedEngineConfig
from src.optimization.algorithms.base import OptimizationAlgorithm, OptimizationResult
from src.optimization.algorithms.optuna import OptunaOptimization
from src.optimization.validation.splitters import WalkForwardSplitter
from src.optimization.validation.walk_forward import WFOEngine, WalkForwardResult


class StrategyOptimizer:
    """
    A high-level facade for running different types of strategy optimizations.
    """

    def __init__(self, strategy_name: str, symbol: str, config: Dict[str, Any]):
        """
        Initializes the StrategyOptimizer.

        Args:
            strategy_name (str): The name of the strategy to optimize.
            symbol (str): The trading symbol to use.
            config (Dict[str, Any]): A dictionary containing all configuration
                                     for the optimization process.
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.config = config

    def run_simple_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """
        Runs a standard hyperparameter optimization over the entire dataset.

        Args:
            data (pd.DataFrame): The full dataset for the optimization.

        Returns:
            OptimizationResult: The result of the optimization.
        """
        # 1. Instantiate components from config
        backtest_config = VectorizedEngineConfig(
            initial_cash=self.config.get("initial_cash", 100_000)
        )
        backtest_engine = VectorizedBacktestEngine(backtest_config)

        optuna_config = self.config.get("optimizer", {})
        optimizer = OptunaOptimization(
            study_name=f"{self.strategy_name}_{self.symbol}",
            direction=optuna_config.get("direction", "maximize")
        )

        # 2. Build the objective function
        from src.optimization.objectives.objective import ObjectiveFunction
        objective_function = ObjectiveFunction(
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            data=data,
            backtest_engine=backtest_engine,
            metric_to_optimize=self.config.get("metric_to_optimize", "sharpe_ratio"),
        )

        # 3. Build the search space
        from src.optimization.search_space.builder import SearchSpaceBuilder
        search_space = SearchSpaceBuilder.from_strategy(self.strategy_name)
        
        # 4. Run optimization
        n_trials = self.config.get("n_trials", 100)
        return optimizer.optimize(objective_function, search_space, n_trials)

    def run_walk_forward_optimization(self, data: pd.DataFrame) -> WalkForwardResult:
        """
        Runs a full Walk-Forward Optimization.

        Args:
            data (pd.DataFrame): The entire dataset for the WFO process.

        Returns:
            WalkForwardResult: The aggregated results of the WFO.
        """
        # 1. Instantiate components from config
        backtest_config = VectorizedEngineConfig(
            initial_cash=self.config.get("initial_cash", 100_000)
        )
        backtest_engine = VectorizedBacktestEngine(backtest_config)
        
        optuna_config = self.config.get("optimizer", {})
        optimizer = OptunaOptimization(
            study_name=f"{self.strategy_name}_{self.symbol}_wfo",
            direction=optuna_config.get("direction", "maximize")
        )

        wfo_config = self.config.get("wfo", {})
        splitter = WalkForwardSplitter(
            train_size=wfo_config.get("train_size"),
            test_size=wfo_config.get("test_size"),
            n_splits=wfo_config.get("n_splits"),
        )
        
        # 2. Instantiate the WFO Engine
        wfo_engine = WFOEngine(
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            backtest_engine=backtest_engine,
            optimizer=optimizer,
            splitter=splitter,
            metric_to_optimize=self.config.get("metric_to_optimize", "sharpe_ratio"),
        )
        
        # 3. Run WFO
        n_trials = self.config.get("n_trials", 50)
        return wfo_engine.run(data, n_trials)
