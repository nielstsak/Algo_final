# src/backtesting/signal_adapter.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from loguru import logger

class SignalAdapter:
    """
    Adapts strategy signals into a format suitable for vectorbt.Portfolio.
    """

    def __init__(self):
        logger.debug("SignalAdapter initialized.")

    def adapt_signals(
        self,
        signals_df: pd.DataFrame,
        allow_shorting: bool = True,
        size_type: str = 'percent', # 'percent', 'amount', 'value'
        default_size: float = 0.1,
        sl_col: str = 'sl', # Column name for stop-loss prices
        tp_col: str = 'tp', # Column name for take-profit prices
        price_col_for_sl_tp: str = 'close' # Price column used as reference for SL/TP if they are relative
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Converts a strategy's signal DataFrame into vectorbt compatible signals.

        Args:
            signals_df: DataFrame from the strategy. Expected columns:
                - 'entry_long': boolean Series for long entries
                - 'exit_long': boolean Series for long exits
                - 'entry_short': boolean Series for short entries (if allow_shorting)
                - 'exit_short': boolean Series for short exits (if allow_shorting)
                - 'sl': Optional float Series for stop-loss levels
                - 'tp': Optional float Series for take-profit levels
                - 'size': Optional float Series for position size (overrides default_size)
            allow_shorting: Whether shorting is allowed.
            size_type: How 'default_size' or 'size' column is interpreted by vectorbt.
            default_size: Default position size if not provided in signals_df.
            sl_col: Name of the stop-loss column in signals_df.
            tp_col: Name of the take-profit column in signals_df.
            price_col_for_sl_tp: The price column (e.g., 'close', 'open') from the main data
                                 that SL/TP values in signals_df might be relative to or should be compared against.
                                 Vectorbt handles SL/TP based on entry price, so this is more for context if needed.

        Returns:
            Tuple of (entries, exits, size_series):
                - entries: Boolean Series for all entry signals.
                - exits: Boolean Series for all exit signals.
                - size_series: Optional Series for position sizing.
        """
        if not isinstance(signals_df, pd.DataFrame):
            raise ValueError("signals_df must be a pandas DataFrame.")

        entries = pd.Series(False, index=signals_df.index)
        exits = pd.Series(False, index=signals_df.index)
        size_series: Optional[pd.Series] = None

        # --- Process Long Signals ---
        if 'entry_long' in signals_df.columns:
            entries = entries | signals_df['entry_long'].fillna(False)
        if 'exit_long' in signals_df.columns:
            exits = exits | signals_df['exit_long'].fillna(False)

        # --- Process Short Signals ---
        if allow_shorting:
            if 'entry_short' in signals_df.columns:
                # vectorbt uses negative size for short entries if `direction='both'`
                # For `Portfolio.from_signals`, entries are just true/false.
                # The direction of trade (long/short) is determined by `short_entries` and `short_exits`
                # or by passing `direction='shortonly'` or `direction='both'`.
                # Here, we combine them and rely on VectorBTEngine to pass correct direction or specific short signals.
                # For simplicity, if VectorBTEngine uses `from_signals` with `entries` and `exits` only,
                # it might need a size array that is negative for shorts.
                # However, `Portfolio.from_signals` also accepts `short_entries` and `short_exits`.
                # Let's assume VectorBTEngine will handle the distinction.
                # For now, `entries` will mark any kind of entry.
                entries = entries | signals_df['entry_short'].fillna(False)
            if 'exit_short' in signals_df.columns:
                exits = exits | signals_df['exit_short'].fillna(False)
        
        # --- Process Size ---
        # Vectorbt's `size` parameter in `Portfolio.from_signals` can be:
        # - A scalar (applied to all trades)
        # - A Series (size for each signal, must align with entries/exits)
        # If `size_type` is 'percent', this is a percentage of current equity.
        # If 'amount', it's a fixed amount of base currency.
        # If 'value', it's a fixed amount of quote currency.
        if 'size' in signals_df.columns and signals_df['size'].notna().any():
            size_series = signals_df['size'].copy()
            # For short entries, vectorbt might expect negative sizes if not using short_entries/exits
            if allow_shorting and 'entry_short' in signals_df.columns:
                size_series = np.where(signals_df['entry_short'], -size_series.abs(), size_series)
            logger.debug("Using 'size' column from signals_df for position sizing.")
        else:
            # Create a default size series if no size column is provided
            # This scalar will be passed to vectorbt, which applies it according to size_type
            # No need to create a full series here if it's just a scalar default.
            # VectorBTEngine will pass this default_size to Portfolio.from_signals.
            # So, size_series can remain None if we are using the scalar default_size.
            # However, if we want to explicitly make short sizes negative:
            if allow_shorting and 'entry_short' in signals_df.columns:
                size_series = pd.Series(default_size, index=signals_df.index)
                size_series = np.where(signals_df['entry_short'], -size_series.abs(), size_series)
            else:
                # If not shorting or no entry_short, size_series can be None
                # and vectorbt will use the scalar default_size.
                # Or, we can create a series with the default_size for consistency.
                # For now, let it be None so VBT uses its scalar default.
                pass


        # --- Process Stop-Loss and Take-Profit ---
        # Vectorbt's `Portfolio.from_signals` directly accepts `sl_stop` and `tp_stop`.
        # These can be scalars (percentage) or Series (price levels).
        # The `VectorBTEngine` should handle passing these.
        # This adapter primarily focuses on entries, exits, and size.
        # If sl/tp are price levels, they should be in signals_df.
        # If they are percentages, VectorBTEngine can handle that.

        # Example: If sl_col and tp_col contain price levels
        # sl_prices = signals_df[sl_col] if sl_col in signals_df else None
        # tp_prices = signals_df[tp_col] if tp_col in signals_df else None
        # These would then be passed to Portfolio.from_signals

        # Clean up: ensure no NaNs in boolean series
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        if entries.sum() == 0:
            logger.warning("No entry signals found after adaptation.")
        
        # For vectorbt, entries and exits should not be true at the same time for the same bar.
        # Resolve conflicts: if entry and exit are true on the same bar, prioritize exit.
        common_signals = entries & exits
        if common_signals.any():
            logger.debug(f"Found {common_signals.sum()} common entry/exit signals. Prioritizing exits.")
            entries[common_signals] = False


        return entries, exits, size_series
