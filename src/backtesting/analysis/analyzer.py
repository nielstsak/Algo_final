# src/backtesting/analysis/analyzer.py

"""
Provides a centralized analyzer for processing and reporting backtest results.
"""

from typing import Dict, Any, Optional

import pandas as pd

from src.backtesting.engine.base import BacktestResult
from src.backtesting.metrics.registry import MetricRegistry
from src.core.exceptions import BacktestError


class BacktestAnalyzer:
    """
    Orchestrates the analysis of a completed backtest run.

    This class takes the raw results from a backtest engine and uses the
    MetricRegistry to calculate a comprehensive suite of performance and
    risk metrics, presenting them in a structured report.
    """

    def __init__(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ):
        """
        Initializes the BacktestAnalyzer.

        Args:
            result (BacktestResult): The result object from a backtest engine run.
            benchmark_returns (Optional[pd.Series]): A series of benchmark returns
                                                     for comparative analysis.
            risk_free_rate (float): The risk-free rate for calculating metrics
                                    like the Sharpe Ratio.
        """
        if result is None or result.equity_curve.empty:
            raise BacktestError("Cannot analyze an empty or invalid backtest result.")

        self.result = result
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.metric_registry = MetricRegistry

    def generate_report(self) -> Dict[str, Any]:
        """
        Calculates all registered metrics and compiles them into a single report.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated metrics,
                            potentially grouped by category.
        """
        report: Dict[str, Any] = {}
        
        # In the future, this could be parallelized if metric calculations are slow.
        for metric_name in self.metric_registry.list_available():
            try:
                metric_instance = self.metric_registry.create(metric_name)
                
                # Prepare arguments for the calculate method
                calculation_args = {
                    "equity_curve": self.result.equity_curve,
                    "portfolio": self.result.portfolio,
                    "benchmark_returns": self.benchmark_returns,
                    "risk_free_rate": self.risk_free_rate,
                }
                
                # The 'stats' dict from vectorbt might contain pre-calculated metrics
                # that we can pass along.
                if self.result.stats:
                    calculation_args.update(self.result.stats)

                value = metric_instance.calculate(**calculation_args)
                report[metric_name] = value

            except Exception as e:
                # Log the error but don't stop the whole report generation
                report[metric_name] = f"Error: {e}"
        
        return report

    def to_series(self) -> pd.Series:
        """
        Generates the report and returns it as a pandas Series for easy viewing.
        
        Returns:
            pd.Series: A Series with metric names as the index and their
                       calculated values.
        """
        report_dict = self.generate_report()
        return pd.Series(report_dict, name="Performance Report")

