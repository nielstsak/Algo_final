# src/backtesting/engine/vectorized.py

"""
A vectorized backtesting engine powered by the `vectorbt` library.

This engine processes entire arrays of data and signals at once, making it
extremely fast for strategies that do not require intra-bar logic or complex
path-dependent operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.backtesting.engine.base import BacktestEngine, BacktestResult
from src.backtesting.execution.fees import FeeModel, NoFeeModel
from src.backtesting.execution.slippage import SlippageModel, NoSlippageModel
from src.backtesting.portfolio.portfolio import Portfolio
from src.core.exceptions import BacktestError
from src.strategies.base import BaseStrategy
from src.strategies.signals import Signal, SignalType, SignalDirection


@dataclass
class VectorizedEngineConfig:
    """Configuration for the VectorizedBacktestEngine."""
    initial_cash: float = 100_000.0
    fee_model: FeeModel = field(default_factory=NoFeeModel)
    slippage_model: SlippageModel = field(default_factory=NoSlippageModel)
    quote_currency: str = "USD"


class VectorizedBacktestEngine(BacktestEngine):
    """
    An implementation of the BacktestEngine using vectorbt for high-speed,
    vectorized backtesting.
    """

    def __init__(self, config: VectorizedEngineConfig):
        """
        Initializes the vectorized backtesting engine.

        Args:
            config (VectorizedEngineConfig): The configuration object for the engine.
        """
        super().__init__(config)

    def run(self, strategy: BaseStrategy, data: pd.DataFrame) -> BacktestResult:
        """
        Executes a vectorized backtest for a single strategy.
        """
        if data.empty:
            raise BacktestError("Input data for backtest cannot be empty.")

        try:
            # 1. Generate strategy signals
            indicators = strategy.calculate_indicators(data)
            signals = strategy.generate_signals(data, indicators)

            # 2. Adapt signals for vectorbt
            entries, exits, sl_stops, tp_stops = self._adapt_signals_for_vbt(signals, data.index)

            # 3. Configure and run the vectorbt portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=data["close"],
                entries=entries,
                exits=exits,
                sl_stops=sl_stops,
                tp_stops=tp_stops,
                init_cash=self.config.initial_cash,
                fees=getattr(self.config.fee_model, 'fee_rate', 0.0),
                slippage=getattr(self.config.slippage_model, 'slippage_rate', 0.0),
                freq=data.index.freq or pd.infer_freq(data.index),
            )
            
            # 4. Create a consistent result object
            stats = portfolio.stats(settings=dict(freq=data.index.freq))
            equity_curve = portfolio.value()
            
            # The portfolio object from our framework is not fully populated in a
            # vectorized backtest. We return a simplified version for consistency.
            final_portfolio = Portfolio(self.config.initial_cash)
            # In a future version, we could attempt to reconstruct fills from vbt's trade log.
            
            return BacktestResult(stats=stats.to_dict(), equity_curve=equity_curve, portfolio=final_portfolio)

        except Exception as e:
            raise BacktestError(f"Vectorized backtest for strategy '{strategy.name}' failed.") from e

    def run_multiple(
        self,
        strategies: Dict[str, BaseStrategy],
        data: pd.DataFrame
    ) -> Dict[str, BacktestResult]:
        """
        Executes backtests for multiple strategies on the same dataset.
        This default implementation runs them sequentially. A more optimized
        version could try to batch indicator calculations if they are shared.
        """
        results = {}
        for name, strategy in strategies.items():
            results[name] = self.run(strategy, data)
        return results

    def _adapt_signals_for_vbt(self, signals: List[Signal], index: pd.Index):
        """
        Converts a list of Signal objects into boolean Series that vectorbt can use.
        """
        entries = pd.Series(False, index=index)
        exits = pd.Series(False, index=index)
        
        # vectorbt's `from_signals` uses sl_stops/tp_stops on a per-signal basis if passed as Series.
        sl_stops = pd.Series(np.nan, index=index)
        tp_stops = pd.Series(np.nan, index=index)

        for signal in signals:
            if signal.timestamp not in index:
                continue # Ignore signals outside the data range
            
            if signal.signal_type == SignalType.ENTRY:
                # For now, we assume SHORT signals are entries, not exits.
                # vectorbt handles position flipping automatically.
                entries.loc[signal.timestamp] = True

                if signal.stop_loss is not None:
                    sl_stops.loc[signal.timestamp] = signal.stop_loss
                if signal.take_profit is not None:
                    tp_stops.loc[signal.timestamp] = signal.take_profit

            elif signal.signal_type == SignalType.EXIT:
                # If strategies provide explicit exit signals, mark them here.
                exits.loc[signal.timestamp] = True
        
        # If no explicit exits are given, vectorbt will exit a position when an
        # entry signal for the opposite direction occurs.
        
        return entries, exits, sl_stops, tp_stops

