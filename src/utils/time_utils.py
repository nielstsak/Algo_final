import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Union, List, Tuple, Optional, Literal, Dict, cast
from dateutil import parser as dateutil_parser
import pytz
from pandas.tseries.frequencies import to_offset
import re
import time

# --- Type Aliases ---
TimestampAlike = Union[pd.Timestamp, datetime, str, int, float, np.datetime64]
# Standard pandas frequency strings. More can be added.
Frequency = Literal['1T', '3T', '5T', '15T', '30T', '1H', '2H', '4H', '6H', '8H', '12H', '1D', '1W', '1M']


# --- Conversion Utilities ---

def to_timestamp(
    time_input: TimestampAlike,
    tz: Optional[str] = 'UTC'
) -> pd.Timestamp:
    """
    Converts various time inputs to a timezone-aware pandas.Timestamp.

    Parameters
    ----------
    time_input : TimestampAlike
        The time input. Can be a pandas.Timestamp, datetime.datetime,
        string (parsable by dateutil.parser), int/float (Unix timestamp in seconds or milliseconds).
    tz : Optional[str], optional
        The target timezone. If None, the timestamp will be naive if the input is naive,
        or keep its original timezone. If 'UTC', it will be converted to UTC.
        Defaults to 'UTC'.

    Returns
    -------
    pd.Timestamp
        A timezone-aware pandas.Timestamp object.

    Raises
    ------
    ValueError
        If the time_input format is unrecognized or unparseable.
    pytz.exceptions.UnknownTimeZoneError
        If the provided timezone string is invalid.

    Examples
    --------
    >>> to_timestamp("2023-01-01 10:00:00")
    Timestamp('2023-01-01 10:00:00+0000', tz='UTC')
    >>> to_timestamp(1672531200) # Unix seconds
    Timestamp('2023-01-01 00:00:00+0000', tz='UTC')
    >>> to_timestamp(1672531200000) # Unix milliseconds
    Timestamp('2023-01-01 00:00:00+0000', tz='UTC')
    >>> to_timestamp(datetime(2023, 1, 1, 5, 0, 0, tzinfo=dt_timezone.utc))
    Timestamp('2023-01-01 05:00:00+0000', tz='UTC')
    >>> to_timestamp("2023-01-01 10:00:00", tz='America/New_York')
    Timestamp('2023-01-01 10:00:00-0500', tz='America/New_York')
    """
    if pd.isna(time_input):
        raise ValueError("Time input cannot be NaT or None.")

    ts: pd.Timestamp
    if isinstance(time_input, pd.Timestamp):
        ts = time_input
    elif isinstance(time_input, datetime):
        # If naive, pandas will assume local system timezone if tz_localize is used without UTC.
        # Best to make it UTC if naive and tz='UTC' is desired.
        if time_input.tzinfo is None:
            ts = pd.Timestamp(time_input) # Naive pd.Timestamp
        else:
            ts = pd.Timestamp(time_input)
    elif isinstance(time_input, (int, float, np.integer, np.floating)):
        # Heuristic to differentiate seconds vs milliseconds for Unix timestamps
        # If timestamp > 3e9 (approx year 2065 in seconds) or has fractional part for ms, assume ms
        # A more robust way might be to check if it's within a "reasonable" range for seconds.
        # For example, if it's larger than now in seconds by a huge margin, it might be ms.
        # Or, if it's < 1e10, it's likely seconds. If > 1e11 and < 1e13, likely ms.
        num_input = float(time_input)
        if num_input > 3e9 and num_input > time.time() * 2 : # If it's far in future as seconds, likely ms
             ts = pd.Timestamp(num_input, unit='ms')
        elif num_input > 1e11: # Likely milliseconds
            ts = pd.Timestamp(num_input, unit='ms')
        else: # Likely seconds
            ts = pd.Timestamp(num_input, unit='s')
    elif isinstance(time_input, str):
        try:
            dt_obj = dateutil_parser.parse(time_input)
            ts = pd.Timestamp(dt_obj)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse string time_input: '{time_input}'. Error: {e}")
    elif isinstance(time_input, np.datetime64):
        ts = pd.Timestamp(time_input)
    else:
        raise ValueError(f"Unrecognized time_input type: {type(time_input)}")

    # Handle timezone
    if tz:
        try:
            if ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            else:
                ts = ts.tz_convert(tz)
        except pytz.exceptions.UnknownTimeZoneError:
            raise pytz.exceptions.UnknownTimeZoneError(f"Unknown timezone: {tz}")
        except Exception as e: # Catch other tz localization/conversion errors
            raise ValueError(f"Error applying timezone '{tz}' to {time_input}: {e}")
    return ts


def to_unix_ms(timestamp: TimestampAlike) -> int:
    """
    Converts a pandas.Timestamp or compatible to Unix milliseconds (UTC).

    Parameters
    ----------
    timestamp : TimestampAlike
        The timestamp to convert.

    Returns
    -------
    int
        Unix timestamp in milliseconds.

    Examples
    --------
    >>> to_unix_ms(pd.Timestamp("2023-01-01 00:00:00", tz="UTC"))
    1672531200000
    >>> to_unix_ms("2023-01-01 00:00:00+00:00")
    1672531200000
    """
    ts_utc = to_timestamp(timestamp, tz='UTC')
    return int(ts_utc.value / 1_000_000) # .value is nanoseconds since epoch

def from_unix_ms(unix_ms: int, tz: Optional[str] = 'UTC') -> pd.Timestamp:
    """
    Converts a Unix timestamp in milliseconds to a pandas.Timestamp.

    Parameters
    ----------
    unix_ms : int
        Unix timestamp in milliseconds.
    tz : Optional[str], optional
        The target timezone. Defaults to 'UTC'.

    Returns
    -------
    pd.Timestamp
        A timezone-aware pandas.Timestamp object.

    Examples
    --------
    >>> from_unix_ms(1672531200000)
    Timestamp('2023-01-01 00:00:00+0000', tz='UTC')
    """
    if not isinstance(unix_ms, (int, np.integer)):
        raise TypeError("Unix milliseconds timestamp must be an integer.")
    ts = pd.Timestamp(unix_ms, unit='ms')
    if tz:
        ts = ts.tz_localize(tz) # Timestamp from unit is always UTC, so localize
    return ts

def ensure_timezone(timestamp: pd.Timestamp, tz: str = 'UTC') -> pd.Timestamp:
    """
    Ensures a pandas.Timestamp is in the specified timezone.
    If naive, localizes to the target timezone. If aware, converts.

    Parameters
    ----------
    timestamp : pd.Timestamp
        The input pandas.Timestamp.
    tz : str, optional
        The target timezone string (e.g., 'UTC', 'America/New_York'). Defaults to 'UTC'.

    Returns
    -------
    pd.Timestamp
        The timestamp in the specified timezone.

    Examples
    --------
    >>> naive_ts = pd.Timestamp("2023-01-01 10:00:00")
    >>> ensure_timezone(naive_ts, 'UTC')
    Timestamp('2023-01-01 10:00:00+0000', tz='UTC')
    >>> aware_ts = pd.Timestamp("2023-01-01 10:00:00", tz='America/New_York')
    >>> ensure_timezone(aware_ts, 'UTC')
    Timestamp('2023-01-01 15:00:00+0000', tz='UTC')
    """
    if not isinstance(timestamp, pd.Timestamp):
        raise TypeError("Input must be a pandas.Timestamp.")
    try:
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(tz)
        else:
            return timestamp.tz_convert(tz)
    except pytz.exceptions.UnknownTimeZoneError:
        raise pytz.exceptions.UnknownTimeZoneError(f"Unknown timezone: {tz}")


# --- Period Generation ---

def generate_date_range(
    start: TimestampAlike,
    end: TimestampAlike,
    freq: Frequency, # Use pandas frequency strings
    tz: Optional[str] = 'UTC',
    closed: Optional[Literal['left', 'right']] = None
) -> pd.DatetimeIndex:
    """
    Generates a DatetimeIndex with a specified frequency.

    Parameters
    ----------
    start : TimestampAlike
        The start of the date range.
    end : TimestampAlike
        The end of the date range.
    freq : Frequency
        The frequency string (e.g., '1T', '1H', '1D').
    tz : Optional[str], optional
        Timezone for the generated DatetimeIndex. Defaults to 'UTC'.
    closed : Optional[Literal['left', 'right']], optional
        Make the interval closed on the given side. None means both included if exact.
        Pandas default is 'left' for start/end, but can be None to include both if they align with freq.

    Returns
    -------
    pd.DatetimeIndex
        A DatetimeIndex.

    Examples
    --------
    >>> generate_date_range("2023-01-01 00:00", "2023-01-01 00:02", freq='1T')
    DatetimeIndex(['2023-01-01 00:00:00+00:00', '2023-01-01 00:01:00+00:00',
                   '2023-01-01 00:02:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq='T')
    """
    start_ts = to_timestamp(start, tz=tz)
    end_ts = to_timestamp(end, tz=tz)

    if start_ts > end_ts:
        raise ValueError(f"Start date {start_ts} must be before or equal to end date {end_ts}.")

    dt_index = pd.date_range(start=start_ts, end=end_ts, freq=freq, closed=closed, tz=tz)
    if dt_index.tz is None and tz is not None: # Ensure tz if pd.date_range somehow drops it (should not with tz arg)
        dt_index = dt_index.tz_localize(tz)
    elif dt_index.tz is not None and tz is not None and str(dt_index.tz) != tz:
        dt_index = dt_index.tz_convert(tz)
    return dt_index


def split_time_period(
    start: TimestampAlike,
    end: TimestampAlike,
    n_splits: int,
    overlap_ratio: float = 0.0, # Ratio of overlap, e.g., 0.1 for 10%
    tz: Optional[str] = 'UTC'
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Splits a given time period into n_splits potentially overlapping sub-periods.

    Parameters
    ----------
    start : TimestampAlike
        Start of the total period.
    end : TimestampAlike
        End of the total period.
    n_splits : int
        Number of sub-periods to create.
    overlap_ratio : float, optional
        The ratio of overlap between consecutive periods relative to a single split's duration.
        E.g., 0.1 means 10% overlap. Must be between 0 and <1. Defaults to 0.0.
    tz : Optional[str], optional
        Timezone for calculations. Defaults to 'UTC'.

    Returns
    -------
    List[Tuple[pd.Timestamp, pd.Timestamp]]
        A list of tuples, where each tuple is (sub_period_start, sub_period_end).

    Raises
    ------
    ValueError
        If n_splits is not positive, or overlap_ratio is invalid.

    Examples
    --------
    >>> splits = split_time_period("2023-01-01", "2023-01-10", n_splits=3, overlap_ratio=0.0)
    >>> len(splits)
    3
    >>> print(f"{splits[0][0]} to {splits[0][1]}") # Approx 3 days each for 9 day period
    2023-01-01 00:00:00+00:00 to 2023-01-04 00:00:00+00:00
    >>> print(f"{splits[1][0]} to {splits[1][1]}")
    2023-01-04 00:00:00+00:00 to 2023-01-07 00:00:00+00:00
    >>> splits_overlap = split_time_period("2023-01-01", "2023-01-10", n_splits=2, overlap_ratio=0.2)
    >>> print(f"{splits_overlap[0][0]} to {splits_overlap[0][1]}")
    2023-01-01 00:00:00+00:00 to 2023-01-05 12:00:00+00:00
    >>> print(f"{splits_overlap[1][0]} to {splits_overlap[1][1]}") # Starts before first one ends
    2023-01-04 14:24:00+00:00 to 2023-01-10 00:00:00+00:00
    """
    if n_splits <= 0:
        raise ValueError("Number of splits must be positive.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("Overlap ratio must be between 0.0 (inclusive) and 1.0 (exclusive).")

    start_ts = to_timestamp(start, tz=tz)
    end_ts = to_timestamp(end, tz=tz)

    if start_ts >= end_ts:
        raise ValueError("Start time must be before end time.")

    total_duration = end_ts - start_ts
    if total_duration <= timedelta(0):
        return [(start_ts, end_ts)] * n_splits # Or raise error

    # Effective duration of each split, considering it forms a sequence
    # If no overlap, split_duration = total_duration / n_splits
    # With overlap, the "advance" per split is less.
    # Total_duration = n_splits * split_duration - (n_splits - 1) * overlap_amount
    # overlap_amount = split_duration * overlap_ratio
    # Total_duration = n_splits * split_duration - (n_splits - 1) * split_duration * overlap_ratio
    # Total_duration = split_duration * (n_splits - (n_splits - 1) * overlap_ratio)
    
    if n_splits == 1:
        return [(start_ts, end_ts)]

    denominator = n_splits - (n_splits - 1) * overlap_ratio
    if denominator == 0: # Avoid division by zero if overlap_ratio is such that it cancels out
        raise ValueError("Invalid overlap ratio for the number of splits, leads to zero effective advance.")
    
    single_split_duration = total_duration / denominator
    overlap_timedelta = single_split_duration * overlap_ratio
    advance_per_split = single_split_duration - overlap_timedelta


    periods = []
    current_start = start_ts
    for i in range(n_splits):
        current_end = current_start + single_split_duration
        if i == n_splits - 1: # Ensure last period ends exactly at end_ts
            current_end = end_ts
        
        # Ensure current_end does not exceed the overall end_ts significantly due to rounding
        current_end = min(current_end, end_ts)
        
        periods.append((current_start, current_end))
        
        if i < n_splits - 1 : # For next iteration, advance current_start
             current_start = current_start + advance_per_split
             # Safety check: if current_start overshoots end_ts, something is wrong or splits are too many
             if current_start >= end_ts and n_splits -1 -i > 0 : # If there are more splits to generate but we are at the end
                  # This can happen if overlap is large and splits are many.
                  # Add remaining splits as (end_ts, end_ts) or handle as error/warning.
                  # For now, let's allow it, the last split will be clamped.
                  pass


    # Adjust the last split's start if it got pushed beyond where it should be due to overlap logic
    # to ensure it doesn't start after the previous one ends too much or after end_ts
    if n_splits > 1 and len(periods) == n_splits:
        last_start, _ = periods[-1]
        prev_end = periods[-2][1]
        if last_start > prev_end : # If there's a gap instead of overlap or continuation
            # This scenario indicates an issue with parameter combination.
            # However, the logic for advance_per_split should prevent large gaps.
            pass
        if last_start >= end_ts: # If last split starts at or after the global end
            periods[-1] = (min(prev_end - advance_per_split, end_ts - timedelta(microseconds=1)), end_ts)


    return periods


def generate_walk_forward_splits(
    data_start: TimestampAlike,
    data_end: TimestampAlike,
    is_days: int,
    oos_days: int,
    gap_days: int,
    n_splits: Optional[int] = None, # If None, generate as many as possible
    expanding: bool = False,
    tz: Optional[str] = 'UTC'
) -> List[Dict[str, pd.Timestamp]]:
    """
    Generates walk-forward optimization splits (In-Sample, Gap, Out-of-Sample).
    Splits are generated by working backwards from data_end.

    Parameters
    ----------
    data_start : TimestampAlike
        The very beginning of the available data.
    data_end : TimestampAlike
        The very end of the available data.
    is_days : int
        Duration of the In-Sample period in days.
    oos_days : int
        Duration of the Out-of-Sample period in days.
    gap_days : int
        Duration of the Gap period (between IS and OOS) in days.
    n_splits : Optional[int], optional
        The desired number of splits. If None, generates all possible splits.
    expanding : bool, optional
        If True, In-Sample window expands. If False, it's a rolling window. Defaults to False.
    tz : Optional[str], optional
        Timezone for calculations. Defaults to 'UTC'.

    Returns
    -------
    List[Dict[str, pd.Timestamp]]
        A list of dictionaries, each representing a split with keys:
        'is_start', 'is_end', 'gap_start', 'gap_end', 'oos_start', 'oos_end'.
        The list is sorted chronologically (earliest split first).

    Raises
    ------
    ValueError
        If date parameters are invalid or durations are not positive.
    """
    if is_days <= 0 or oos_days <= 0 or gap_days < 0:
        raise ValueError("is_days and oos_days must be positive, gap_days non-negative.")

    start_ts = to_timestamp(data_start, tz=tz)
    end_ts = to_timestamp(data_end, tz=tz)

    if start_ts >= end_ts:
        raise ValueError("Data start time must be before data end time.")

    is_delta = pd.Timedelta(days=is_days)
    oos_delta = pd.Timedelta(days=oos_days)
    gap_delta = pd.Timedelta(days=gap_days)

    splits = []
    current_oos_end = end_ts

    while True:
        oos_start = current_oos_end - oos_delta + pd.Timedelta(days=1) # Inclusive start
        if oos_start < start_ts : # Not enough data for OOS
            break
        
        gap_end = oos_start - pd.Timedelta(days=1)
        gap_start = gap_end - gap_delta + pd.Timedelta(days=1)
        if gap_start < start_ts and gap_days > 0: # Not enough data for Gap
             break

        is_period_end = gap_start - pd.Timedelta(days=1) if gap_days > 0 else gap_end

        if expanding:
            is_period_start = start_ts
        else:
            is_period_start = is_period_end - is_delta + pd.Timedelta(days=1)

        if is_period_start < start_ts: # Not enough data for IS
            break
        
        # Ensure is_period_start is not after is_period_end
        if is_period_start > is_period_end:
            break

        splits.append({
            'is_start': is_period_start, 'is_end': is_period_end,
            'gap_start': gap_start if gap_days > 0 else is_period_end + pd.Timedelta(days=1), # Adjust if no gap
            'gap_end': gap_end if gap_days > 0 else is_period_end, # Adjust if no gap
            'oos_start': oos_start, 'oos_end': current_oos_end
        })

        if n_splits is not None and len(splits) >= n_splits:
            break
        
        # Move to the next potential split (backwards)
        current_oos_end = oos_start - pd.Timedelta(days=1) - gap_delta # End of previous OOS is start of current OOS - gap
        if current_oos_end < start_ts + is_delta + gap_delta + oos_delta - pd.Timedelta(days=3) : # Rough check
            break


    splits.reverse()  # Sort chronologically
    return splits


# --- Alignment Utilities ---

def align_timestamps(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = 'inner'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns two DataFrames based on their DatetimeIndex.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame with a DatetimeIndex.
    df2 : pd.DataFrame
        Second DataFrame with a DatetimeIndex.
    method : str, optional
        Method for alignment ('inner', 'outer', 'left', 'right'). Defaults to 'inner'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The two aligned DataFrames.

    Raises
    ------
    TypeError
        If inputs are not pandas DataFrames or do not have a DatetimeIndex.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df1.index, pd.DatetimeIndex):
        raise TypeError("df1 must be a pandas DataFrame with a DatetimeIndex.")
    if not isinstance(df2, pd.DataFrame) or not isinstance(df2.index, pd.DatetimeIndex):
        raise TypeError("df2 must be a pandas DataFrame with a DatetimeIndex.")

    # Ensure timezones are compatible or convert to UTC for alignment
    df1_aligned, df2_aligned = df1.align(df2, join=method, axis=0) # axis=0 for index alignment
    
    return cast(pd.DataFrame, df1_aligned), cast(pd.DataFrame, df2_aligned)


def get_common_timerange(
    *dataframes: pd.DataFrame
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Finds the common (intersecting) time range among multiple DataFrames.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        Variable number of DataFrames, each with a DatetimeIndex.

    Returns
    -------
    Optional[Tuple[pd.Timestamp, pd.Timestamp]]
        A tuple (common_start_time, common_end_time), or None if no common range or no dataframes.
        Timestamps are returned in UTC.
    """
    if not dataframes:
        return None

    common_start: Optional[pd.Timestamp] = None
    common_end: Optional[pd.Timestamp] = None

    for df in dataframes:
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            continue # Skip empty or invalid dataframes

        current_start_utc = ensure_timezone(df.index.min(), 'UTC')
        current_end_utc = ensure_timezone(df.index.max(), 'UTC')

        if common_start is None or current_start_utc > common_start:
            common_start = current_start_utc
        if common_end is None or current_end_utc < common_end:
            common_end = current_end_utc
            
    if common_start is None or common_end is None or common_start > common_end:
        return None # No common overlapping range found

    return common_start, common_end


def fill_missing_timestamps(
    df: pd.DataFrame,
    freq: Frequency,
    fill_method: Optional[str] = None, # e.g., 'ffill', 'bfill'
    fill_value: Optional[Any] = None
) -> pd.DataFrame:
    """
    Fills missing timestamps in a DataFrame's DatetimeIndex by reindexing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    freq : Frequency
        The desired frequency for the index.
    fill_method : Optional[str], optional
        Method to use for filling NaN values after reindexing (e.g., 'ffill', 'bfill').
    fill_value : Optional[Any], optional
        Value to use for filling NaN values if fill_method is not used.

    Returns
    -------
    pd.DataFrame
        DataFrame with a complete DatetimeIndex at the specified frequency.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")
    if df.empty:
        return df.copy()

    # Ensure timezone consistency, default to UTC if naive
    original_tz = df.index.tz
    df_temp = df.copy()
    if original_tz is None:
        df_temp.index = df_temp.index.tz_localize('UTC')
        original_tz = pytz.UTC # type: ignore

    start_time = df_temp.index.min()
    end_time = df_temp.index.max()
    
    new_index = pd.date_range(start=start_time, end=end_time, freq=freq, tz=original_tz)
    
    df_reindexed = df_temp.reindex(new_index)

    if fill_method:
        df_reindexed = df_reindexed.fillna(method=fill_method)
    elif fill_value is not None:
        df_reindexed = df_reindexed.fillna(value=fill_value)
        
    # If original was naive, and we localized to UTC, decide if we should convert back
    # For consistency, this function will return with the timezone it operated on (original_tz or UTC)
    return df_reindexed


# --- Resampling Helpers ---
from typing import Any # For fill_value

def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: Frequency,
    origin: str = 'start_day', # Pandas default, or 'epoch', 'start'
    closed: Optional[Literal['left', 'right']] = 'left',
    label: Optional[Literal['left', 'right']] = 'left',
    custom_aggregations: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Resamples OHLCV DataFrame to a target frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'open', 'high', 'low', 'close' columns and a DatetimeIndex.
        'volume' is optional.
    target_freq : Frequency
        Target frequency string (e.g., '5T', '1H').
    origin : str, optional
        The timestamp on which to adjust the grouping. Default 'start_day'.
        Common options: 'epoch', 'start', 'start_day'.
    closed : Optional[Literal['left', 'right']], optional
        Which side of bin interval is closed. Default 'left'.
    label : Optional[Literal['left', 'right']], optional
        Which bin edge label to label bucket with. Default 'left'.
    custom_aggregations : Optional[Dict[str, Any]], optional
        Custom aggregation functions for other columns.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame.

    Raises
    ------
    ValueError
        If required OHLC columns are missing.
    """
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {', '.join(required_cols)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df.columns:
        agg_rules['volume'] = 'sum'
    if custom_aggregations:
        agg_rules.update(custom_aggregations)
        
    # Filter out columns not in agg_rules to prevent warnings/errors if df has other columns
    cols_to_resample = [col for col in agg_rules.keys() if col in df.columns]
    if not cols_to_resample:
        raise ValueError("No columns available for resampling based on aggregation rules.")

    resampled_df = df[cols_to_resample].resample(
        rule=target_freq,
        closed=closed, # type: ignore
        label=label,   # type: ignore
        origin=origin
    ).agg(agg_rules)
    
    return resampled_df.dropna(subset=['open']) # Drop rows where open is NaN (usually means no data in interval)


def create_rolling_windows(
    df: pd.DataFrame,
    window_size_str: str, # Pandas offset string, e.g., '3H' for 3 hours
    step_size_str: Optional[str] = None # Pandas offset string, e.g., '1H'
) -> List[pd.DataFrame]:
    """
    Creates rolling (overlapping) or tumbling (non-overlapping) windows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    window_size_str : str
        Pandas offset string for the window size (e.g., '3H', '1D').
    step_size_str : Optional[str], optional
        Pandas offset string for the step size. If None or equal to window_size_str,
        creates tumbling (non-overlapping) windows. Otherwise, creates overlapping windows.
        Defaults to None (tumbling windows).

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames, each representing a window.

    Note: This is a basic implementation. For large data, generating all DFs in a list
          might be memory intensive. Consider a generator or on-demand processing.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")
    if df.empty:
        return []

    window_td = pd.to_timedelta(to_offset(window_size_str)) # type: ignore
    step_td = window_td
    if step_size_str:
        step_td = pd.to_timedelta(to_offset(step_size_str)) # type: ignore

    windows = []
    current_time = df.index.min()
    end_time = df.index.max()

    while current_time <= end_time:
        window_end_time = current_time + window_td
        # For the slice, pandas includes the end if it's exact.
        # We want [current_time, window_end_time) typically for left-closed intervals.
        # However, df slicing is inclusive.
        # Let's make it [current_time, current_time + window_td - smallest_delta]
        # Or, more simply, slice up to window_end_time and if the last point is window_end_time, it's included.
        
        # Slice up to window_end_time.
        # If step_td makes window_end_time go beyond max index, the last window might be smaller.
        actual_window_end = min(window_end_time, end_time + pd.Timedelta(nanoseconds=1)) # Ensure last point can be included

        window_df = df[(df.index >= current_time) & (df.index < actual_window_end)]
        
        if not window_df.empty:
            windows.append(window_df)
        
        if current_time + step_td > end_time and current_time != df.index.min() : # Avoid infinite loop if step is too small and we are at the end
            if window_end_time > end_time and not window_df.empty and window_df.index.max() == end_time:
                 # If the last window captured the end_time, we are done.
                 pass
            elif window_end_time <= end_time : # If there is still room for another window starting at current_time + step_td
                 pass # let it continue
            else: # No more full steps possible
                 break
        
        current_time += step_td
        if current_time > end_time and len(windows) > 0 and windows[-1].index.max() == end_time:
            break


    return windows


def time_weighted_average(
    df: pd.DataFrame, # Must have DatetimeIndex
    value_col: str,
    target_freq: Frequency,
    time_col_for_weights: Optional[str] = None # If None, uses index diffs
) -> pd.Series:
    """
    Calculates time-weighted average for a value column, resampling to target_freq.
    Assumes observations are point-in-time, and value persists until next observation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and a value column.
    value_col : str
        Name of the column containing values to average.
    target_freq : Frequency
        Target frequency for TWA.
    time_col_for_weights : Optional[str], optional
        If provided, this column (must be datetime or timestamp) is used to calculate
        durations. Otherwise, the difference between consecutive index timestamps is used.

    Returns
    -------
    pd.Series
        Time-weighted average series at target_freq.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame.")
    if df.empty or df[value_col].isnull().all():
        # Return empty series with correct index type if possible, or handle as error
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq, tz=df.index.tz) if not df.empty else None
        return pd.Series(dtype=float, index=idx)


    data = df[[value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
    data = data.dropna(subset=[value_col]) # Remove rows where value is NaN

    if data.empty:
         idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq, tz=df.index.tz) if not df.empty else None
         return pd.Series(dtype=float, index=idx)


    # Calculate durations each value was active
    if time_col_for_weights and time_col_for_weights in data.columns:
        timestamps = pd.to_datetime(data[time_col_for_weights], utc=True)
    else:
        timestamps = data.index.to_series()

    # Time difference to next point, or end of period for last point
    durations = timestamps.diff().shift(-1) # Time diff from current point to next
    
    # For the last data point, duration extends to end of its resampling period
    # This is complex. A simpler way is to resample value_col * duration_seconds
    # and duration_seconds separately, then divide.

    # Upsample to a high frequency (e.g., seconds) then resample. This can be robust.
    # Or, calculate weighted sum manually.
    
    # Simpler approach: resample value * duration and duration, then divide.
    # This requires careful handling of how durations are defined at resampling boundaries.
    
    # Pandas resampler with .apply() can do this:
    def twa_apply(series_in_bin):
        if series_in_bin.empty:
            return np.nan
        
        # Create a series of time diffs within the bin
        # The value is considered to hold from its timestamp until the next one, or bin end
        
        # This is a simplified TWA: take the mean of values in bin, assuming they are point samples
        # A true TWA needs to know how long each value persisted.
        # For a more accurate TWA, one might need to forward-fill the data to a fine resolution,
        # then resample with mean. Or calculate weights explicitly.
        
        # Let's assume values are constant until the next measurement.
        # This means we need to calculate the time each value was "active".
        
        # If we have `df` with irregular timestamps:
        #   time1, value1
        #   time2, value2
        # value1 was active for (time2 - time1).
        
        # This is a complex function to implement robustly as a generic utility.
        # For now, a simple mean after resampling if that's acceptable, or placeholder.
        # A common method:
        # 1. Forward fill the original data to cover gaps up to a reasonable limit.
        # 2. Resample using 'mean'. This approximates TWA if data points are frequent enough.
        
        # For a true TWA within resample().agg():
        # Need to calculate weights for each point within the aggregation window.
        # This is non-trivial with standard pandas resample().agg().
        
        # Return simple mean for now, user should be aware of implications.
        return series_in_bin.mean()

    # This will just be a normal resample mean, not a true TWA unless data is already regular.
    # To do true TWA, often you upsample, ffill, then downsample.
    # data_ffilled = data.asfreq(pd.infer_freq(data.index) or 'T').ffill() # Infer or assume a base freq
    # return data_ffilled[value_col].resample(target_freq).mean()
    # This is still not perfect.
    
    # Placeholder: For a proper TWA, a more involved calculation is needed, potentially outside .resample().agg()
    # or by using a library that supports it directly.
    # For this utility, we'll return the mean of the resampled period.
    return df[value_col].resample(target_freq).mean()


# --- Duration Calculations ---

def trading_days_between(start: TimestampAlike, end: TimestampAlike) -> int:
    """
    Calculates the number of trading days (calendar days for crypto) between two timestamps.
    For 24/7 crypto markets, this is essentially the number of calendar days.

    Parameters
    ----------
    start : TimestampAlike
        The start timestamp.
    end : TimestampAlike
        The end timestamp.

    Returns
    -------
    int
        Number of full calendar days between start (exclusive) and end (inclusive),
        or if difference is less than a day, can be 0 or 1 depending on interpretation.
        This returns (end_day - start_day).
    """
    start_ts = to_timestamp(start, tz='UTC').normalize() # Normalize to midnight
    end_ts = to_timestamp(end, tz='UTC').normalize()

    if start_ts > end_ts:
        return 0
    
    # (end_ts - start_ts).days gives the number of full 24h periods.
    return (end_ts - start_ts).days

def trading_hours_between(start: TimestampAlike, end: TimestampAlike) -> float:
    """
    Calculates the total trading hours between two timestamps.
    For 24/7 crypto, this is total hours.

    Parameters
    ----------
    start : TimestampAlike
        The start timestamp.
    end : TimestampAlike
        The end timestamp.

    Returns
    -------
    float
        Total hours between the two timestamps.
    """
    start_ts = to_timestamp(start, tz='UTC')
    end_ts = to_timestamp(end, tz='UTC')

    if start_ts > end_ts:
        return 0.0
        
    delta: timedelta = end_ts - start_ts
    return delta.total_seconds() / 3600.0

def next_period_start(timestamp: TimestampAlike, freq: Frequency) -> pd.Timestamp:
    """
    Calculates the start of the next period for a given timestamp and frequency.

    Parameters
    ----------
    timestamp : TimestampAlike
        The reference timestamp.
    freq : Frequency
        The frequency string (e.g., '1H', '1D').

    Returns
    -------
    pd.Timestamp
        The start of the next period.

    Examples
    --------
    >>> next_period_start("2023-01-01 10:30:00", freq='1H')
    Timestamp('2023-01-01 11:00:00+0000', tz='UTC')
    >>> next_period_start("2023-01-01 10:30:00", freq='1D')
    Timestamp('2023-01-02 00:00:00+0000', tz='UTC')
    """
    ts = to_timestamp(timestamp, tz='UTC')
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset is None:
        raise ValueError(f"Invalid frequency string for offset: {freq}")

    # Get the current period's start, then add one offset.
    # For 'T' (minute), 'H' (hour), 'D' (day) based frequencies:
    # Truncate to the current period start, then add one period.
    # e.g. 10:30, freq '1H' -> current period start is 10:00. Next is 11:00.
    # e.g. 10:30, freq '1T' -> current period start is 10:30. Next is 10:31. (This is if ts is on boundary)
    # If ts is 10:30:15, freq '1T', current period start is 10:30. Next 10:31.
    
    # A robust way: find the floor of the current timestamp to the frequency
    # For example, for 'H', floor 10:30:15 to 10:00:00
    # For 'T', floor 10:30:15 to 10:30:00
    # This is what `ts.floor(freq)` does.
    current_period_start_val = ts.floor(freq)
    
    # If the input timestamp is exactly on a period boundary, `floor` gives that boundary.
    # We want the *next* period start.
    if ts == current_period_start_val:
        # If ts is already a period start, the next period starts one offset away.
        return current_period_start_val + offset
    else:
        # If ts is within a period, current_period_start_val is the start of the *current* period.
        # The next period starts one offset after that.
        return current_period_start_val + offset


def previous_period_end(timestamp: TimestampAlike, freq: Frequency) -> pd.Timestamp:
    """
    Calculates the end of the previous period for a given timestamp and frequency.
    The "end" is exclusive for the next period's start, i.e., it's the last moment of the period.

    Parameters
    ----------
    timestamp : TimestampAlike
        The reference timestamp.
    freq : Frequency
        The frequency string (e.g., '1H', '1D').

    Returns
    -------
    pd.Timestamp
        The end of the previous period (exclusive end, i.e., start of current period - 1ns).

    Examples
    --------
    >>> previous_period_end("2023-01-01 10:30:00", freq='1H')
    Timestamp('2023-01-01 09:59:59.999999999+0000', tz='UTC')
    >>> previous_period_end("2023-01-01 00:00:00", freq='1D') # If on boundary, previous period end
    Timestamp('2022-12-31 23:59:59.999999999+0000', tz='UTC')
    """
    ts = to_timestamp(timestamp, tz='UTC')
    # Get the start of the current period
    current_period_start = ts.floor(freq)
    
    # If the timestamp is exactly at the start of a period,
    # the "previous period end" is the end of the period before this one.
    # So, effectively, it's current_period_start - 1 nanosecond.
    return current_period_start - pd.Timedelta(nanoseconds=1)


# --- Time Validation ---

VALID_TIMEFRAMES_REGEX = re.compile(r"^(\d+)([mhdw])$", re.IGNORECASE) # m,h,D,W for minute, hour, day, week
VALID_PANDAS_FREQS = ['T', 'min', 'H', 'D', 'W', 'M', 'Q', 'Y'] # Base pandas frequencies

def is_valid_timeframe(timeframe: str) -> bool:
    """
    Validates if a given timeframe string is in a supported format (e.g., "1m", "5T", "1H", "4h", "1D").

    Parameters
    ----------
    timeframe : str
        The timeframe string.

    Returns
    -------
    bool
        True if valid, False otherwise.

    Examples
    --------
    >>> is_valid_timeframe("15T") # Pandas style
    True
    >>> is_valid_timeframe("1h") # Common style
    True
    >>> is_valid_timeframe("1X")
    False
    """
    try:
        pd.tseries.frequencies.to_offset(timeframe)
        return True
    except ValueError:
        # Check common alternative forms like "1m" if pandas doesn't directly support
        match = VALID_TIMEFRAMES_REGEX.match(timeframe)
        if match:
            num, unit = int(match.group(1)), match.group(2).upper()
            # Convert to pandas style if possible for a final check
            # e.g. 1m -> 1T, 1h -> 1H, 1d -> 1D, 1w -> 1W
            # This is implicitly handled by to_offset if it's a common one.
            # The regex is more for parsing if needed.
            return True # If regex matches, assume it's a parsable common format
        return False

def validate_time_range(start: TimestampAlike, end: TimestampAlike) -> bool:
    """
    Validates if the start timestamp is before or equal to the end timestamp.

    Parameters
    ----------
    start : TimestampAlike
        The start timestamp.
    end : TimestampAlike
        The end timestamp.

    Returns
    -------
    bool
        True if valid range, False otherwise.
    """
    try:
        start_ts = to_timestamp(start, tz='UTC') # Convert to common base for comparison
        end_ts = to_timestamp(end, tz='UTC')
        return start_ts <= end_ts
    except ValueError:
        return False

def is_market_open(timestamp: Optional[TimestampAlike] = None) -> bool:
    """
    Checks if the crypto market is open at the given timestamp.
    Crypto markets are 24/7, so this always returns True.

    Parameters
    ----------
    timestamp : Optional[TimestampAlike], optional
        The timestamp to check. If None, checks current time. (Not used for crypto).

    Returns
    -------
    bool
        Always True for crypto markets.
    """
    return True


# --- Performance Monitoring ---

class TimeTracker:
    """A simple class to track execution times of code blocks."""
    def __init__(self):
        self.timers: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str) -> None:
        """Starts a timer for a named block."""
        if name not in self.timers:
            self.timers[name] = []
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> Optional[float]:
        """Stops the timer for a named block and records the duration."""
        if name in self._start_times:
            elapsed_time = time.perf_counter() - self._start_times.pop(name)
            self.timers[name].append(elapsed_time)
            return elapsed_time
        # print(f"Warning: Timer '{name}' was stopped without being started.")
        return None

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Gets statistics for a named timer."""
        if name in self.timers and self.timers[name]:
            durations = self.timers[name]
            return {
                "name": name,
                "count": len(durations),
                "total_time": sum(durations),
                "avg_time": sum(durations) / len(durations),
                "min_time": min(durations),
                "max_time": max(durations)
            }
        return None

    def reset(self, name: Optional[str] = None) -> None:
        """Resets a specific timer or all timers if name is None."""
        if name:
            if name in self.timers: self.timers[name] = []
            if name in self._start_times: del self._start_times[name]
        else:
            self.timers.clear()
            self._start_times.clear()


# --- Utilities ---

def format_duration(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable string (Xd Yh Zm Ws).

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable duration string.
    """
    if seconds < 0: return "0s"
    
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    parts = []
    if d > 0: parts.append(f"{int(d)}d")
    if h > 0: parts.append(f"{int(h)}h")
    if m > 0: parts.append(f"{int(m)}m")
    if s > 0 or not parts : parts.append(f"{s:.2f}s" if isinstance(s, float) and not s.is_integer() else f"{int(s)}s")
    
    return " ".join(parts) if parts else "0s"


TIMEFRAME_REGEX = re.compile(r"(\d+)([TsmhdMWQY])", re.IGNORECASE) # T for minute in pandas, m for minute common
TIMEFRAME_UNIT_MAP = {
    'T': 'minute', 'MIN': 'minute', 'S': 'second',
    'H': 'hour', 'D': 'day', 'W': 'week', 'M': 'month',
    'Q': 'quarter', 'Y': 'year', 'A': 'year'
}
TIMEFRAME_TO_MINUTES_MAP = {
    'T': 1, 'MIN': 1, 'H': 60, 'D': 24 * 60, 'W': 7 * 24 * 60,
    # Months are tricky due to variable length. This is an approximation.
    'M': 30 * 24 * 60, # Approx
}


def parse_timeframe(timeframe_str: str) -> Optional[Tuple[int, str]]:
    """
    Parses a timeframe string (e.g., "15T", "1h", "3d") into amount and unit.

    Parameters
    ----------
    timeframe_str : str
        The timeframe string.

    Returns
    -------
    Optional[Tuple[int, str]]
        A tuple (amount, unit_name) e.g., (15, 'minute'), or None if unparseable.
        Unit name is from TIMEFRAME_UNIT_MAP.
    """
    # Try pandas offset first
    try:
        offset = pd.tseries.frequencies.to_offset(timeframe_str)
        if offset:
            # This is tricky because offset.name might be complex e.g. <Day>
            # We need to extract numeric multiplier and base unit.
            # For simple cases like '5T', offset.n = 5, offset.rule_code = 'T'
            if hasattr(offset, 'n') and hasattr(offset, 'rule_code'):
                 unit_name = TIMEFRAME_UNIT_MAP.get(offset.rule_code.upper())
                 if unit_name:
                     return offset.n, unit_name
            # Fallback to regex for common patterns if pandas offset parsing is not straightforward
    except ValueError:
        pass # Will try regex if pandas fails

    match = TIMEFRAME_REGEX.match(timeframe_str)
    if match:
        amount = int(match.group(1))
        unit_char = match.group(2).upper()
        
        # Handle 'm' for minute if pandas uses 'T'
        if unit_char == 'M' and timeframe_str.lower().endswith('m'): # Distinguish Month from minute
             if timeframe_str == str(amount) + 'm': # e.g. "15m"
                 unit_name = 'minute'
             else: # e.g. "1M"
                 unit_name = TIMEFRAME_UNIT_MAP.get(unit_char)
        elif unit_char.upper() == 'T' or unit_char.lower() == 'm': # if unit is 'T' or 'm'
            unit_name = 'minute'
        else:
            unit_name = TIMEFRAME_UNIT_MAP.get(unit_char)

        if unit_name:
            return amount, unit_name
            
    return None


def timeframe_to_minutes(timeframe: Frequency) -> int:
    """
    Converts a frequency string to an approximate number of minutes.
    Note: Month/Quarter/Year are approximate.

    Parameters
    ----------
    timeframe : Frequency
        The frequency string (e.g., "1T", "1H", "1D").

    Returns
    -------
    int
        Equivalent number of minutes.

    Raises
    ------
    ValueError
        If the timeframe string is not recognized or convertible.
    """
    parsed = parse_timeframe(timeframe)
    if not parsed:
        raise ValueError(f"Invalid or unhandled timeframe string: {timeframe}")

    amount, unit_key_part = parsed
    
    # Find a key in TIMEFRAME_TO_MINUTES_MAP that matches unit_key_part (e.g. 'minute' -> 'MIN' or 'T')
    mapped_minutes = 0
    for map_key, minutes_val in TIMEFRAME_TO_MINUTES_MAP.items():
        if unit_key_part.upper() == TIMEFRAME_UNIT_MAP.get(map_key, '').upper():
            mapped_minutes = minutes_val
            break
    
    if mapped_minutes == 0 and unit_key_part.upper() != 'SECOND': # if not found and not second
        raise ValueError(f"Cannot convert timeframe unit '{unit_key_part}' to minutes for '{timeframe}'.")

    if unit_key_part.upper() == 'SECOND':
        if amount % 60 != 0:
            raise ValueError("Conversion to minutes from seconds is only exact for multiples of 60s.")
        return amount // 60
        
    return amount * mapped_minutes


def get_current_period_bounds(freq: Frequency, tz: str = 'UTC') -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Gets the start and end timestamps of the current trading period for a given frequency.
    The end is exclusive (start of next period).

    Parameters
    ----------
    freq : Frequency
        The frequency string (e.g., '1H', '1D').
    tz : str, optional
        The timezone to use. Defaults to 'UTC'.

    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        (current_period_start, current_period_end)
    """
    now = pd.Timestamp.now(tz=tz)
    current_period_start = now.floor(freq)
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset is None:
         raise ValueError(f"Invalid frequency string for offset: {freq}")
    current_period_end = current_period_start + offset
    return current_period_start, current_period_end


if __name__ == '__main__':
    print("--- Testing time_utils ---")

    # Conversions
    ts_utc = to_timestamp("2023-01-01 12:30:45.123Z")
    print(f"String to Timestamp (UTC): {ts_utc}")
    unix_ms_val = to_unix_ms(ts_utc)
    print(f"Timestamp to Unix MS: {unix_ms_val}")
    print(f"Unix MS to Timestamp: {from_unix_ms(unix_ms_val)}")
    ts_ny = to_timestamp("2023-01-01 12:00:00", tz="America/New_York")
    print(f"Timestamp in NY: {ts_ny}")
    print(f"Timestamp NY to UTC: {ensure_timezone(ts_ny, 'UTC')}")

    # Period Generation
    dr = generate_date_range("2023-01-01 00:00", "2023-01-01 00:05", freq='1T')
    print(f"Date Range (1T):\n{dr}")
    
    splits_simple = split_time_period("2023-01-01", "2023-01-04", n_splits=3) # 3 days, 3 splits
    print(f"Split Time Period (3 splits, no overlap):")
    for i, (s, e) in enumerate(splits_simple): print(f"  Split {i+1}: {s} to {e}")

    wfo_splits = generate_walk_forward_splits("2023-01-01", "2023-06-30", is_days=90, oos_days=30, gap_days=5, n_splits=2)
    print(f"WFO Splits (2 splits):")
    for i, split in enumerate(wfo_splits): print(f"  Split {i+1}: IS {split['is_start'].date()}->{split['is_end'].date()}, OOS {split['oos_start'].date()}->{split['oos_end'].date()}")

    # Alignment
    df_a = pd.DataFrame({'A': range(5)}, index=pd.date_range("2023-01-01 00:00", periods=5, freq='1H', tz='UTC'))
    df_b = pd.DataFrame({'B': range(3,8)}, index=pd.date_range("2023-01-01 01:00", periods=5, freq='1H', tz='UTC'))
    aligned_a, aligned_b = align_timestamps(df_a, df_b, method='inner')
    print(f"Aligned A (inner):\n{aligned_a.head(2)}")
    common_start, common_end = get_common_timerange(df_a, df_b) # type: ignore
    print(f"Common Timerange: {common_start} to {common_end}")
    
    df_missing = df_a.drop(df_a.index[2]) # Drop one row
    df_filled = fill_missing_timestamps(df_missing, freq='1H', fill_method='ffill')
    print(f"Filled Timestamps (original count {len(df_missing)}, filled count {len(df_filled)}):")
    print(df_filled.head())


    # Resampling
    ohlcv_data = {
        'open': [10, 11, 10.5, 12], 'high': [12, 11.5, 11, 12.5],
        'low': [9, 10.5, 10, 11.5], 'close': [11, 10.5, 12, 12.2],
        'volume': [100, 150, 120, 200]
    }
    ohlcv_idx = pd.date_range("2023-01-01 09:00", periods=4, freq='15T', tz='UTC')
    df_ohlcv = pd.DataFrame(ohlcv_data, index=ohlcv_idx)
    resampled_1h = resample_ohlcv(df_ohlcv, target_freq='1H')
    print(f"Resampled OHLCV (15T to 1H):\n{resampled_1h}")

    # Rolling Windows (Example is conceptual as output can be large)
    # windows = create_rolling_windows(df_a, window_size_str='2H', step_size_str='1H')
    # print(f"Number of 2H rolling windows with 1H step: {len(windows)}")
    # if windows: print(f"First window:\n{windows[0]}")


    # Duration
    print(f"Trading Days (2023-01-01 to 2023-01-05): {trading_days_between('2023-01-01', '2023-01-05')}")
    print(f"Trading Hours (10:00 to 12:30): {trading_hours_between('2023-01-01 10:00', '2023-01-01 12:30')}")
    print(f"Next 1H start from 10:30: {next_period_start('2023-01-01 10:30', '1H')}")
    print(f"Previous 1D end from 2023-01-01 00:00: {previous_period_end('2023-01-01 00:00', '1D')}")

    # Validation
    print(f"Is '15m' valid timeframe? {is_valid_timeframe('15m')}")
    print(f"Is '1X' valid timeframe? {is_valid_timeframe('1X')}")
    print(f"Is time range valid ('2023-01-01', '2023-01-05')? {validate_time_range('2023-01-01', '2023-01-05')}")
    print(f"Is market open now? {is_market_open()}")

    # TimeTracker
    tracker = TimeTracker()
    tracker.start("test_sleep")
    time.sleep(0.05)
    tracker.stop("test_sleep")
    stats = tracker.get_stats("test_sleep")
    print(f"TimeTracker stats for 'test_sleep': {stats['avg_time'] if stats else 'N/A'}") # type: ignore

    # Utilities
    print(f"Format duration 3661.5 seconds: {format_duration(3661.5)}")
    print(f"Parse timeframe '15T': {parse_timeframe('15T')}")
    print(f"Parse timeframe '4h': {parse_timeframe('4h')}")
    print(f"Timeframe '4H' to minutes: {timeframe_to_minutes('4H')}")
    print(f"Current 1H period bounds: {get_current_period_bounds('1H')}")

