import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone # Ajout de timezone pour example_usage
from loguru import logger
import re # Added for _freq_to_minutes

from src.core.constants import Kline 
from src.core.exceptions import DataError, KlineValidationError


class KlineProcessor:
    """
    Processeur pour la manipulation et le resampling des klines.
    Gère la génération de klines multi-fréquences (standard et glissantes).
    """
    
    def __init__(self):
        """Initialise le processeur de klines."""
        self.interval_mapping = {
            '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M' 
        }
        
        if hasattr(Kline, 'AGGREGATION_RULES_STANDARD') and isinstance(Kline.AGGREGATION_RULES_STANDARD, dict):
            self.aggregation_rules = Kline.AGGREGATION_RULES_STANDARD.copy()
        else:
            logger.warning(
                "Kline.AGGREGATION_RULES_STANDARD not found or not a dict in constants. "
                "Using a default set of aggregation rules for standard klines, "
                "referencing Kline constants for column names."
            )
            self.aggregation_rules = {
                Kline.OHLCV_OPEN: "first",
                Kline.OHLCV_HIGH: "max",
                Kline.OHLCV_LOW: "min",
                Kline.OHLCV_CLOSE: "last",
                Kline.OHLCV_VOLUME: "sum",
                Kline.OHLCV_QUOTE_ASSET_VOLUME: "sum",
                Kline.OHLCV_NUMBER_OF_TRADES: "sum",
                Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: "sum",
                Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: "sum"
            }
            
        logger.debug("KlineProcessor initialized")
        
    def resample_klines(
        self,
        df: pd.DataFrame,
        target_freq: str,
        rolling: bool = False,
        window_size: Optional[int] = None 
    ) -> pd.DataFrame:
        """
        Génère des klines de fréquence supérieure ou des klines glissantes.
        
        Args:
            df: DataFrame avec klines source (doit avoir un index temporel DatetimeIndex).
                Les noms de colonnes doivent correspondre à ceux de src.core.constants.Kline.
            target_freq: Fréquence cible ('1m', '5m', '1h', etc.) pour les klines standard,
                         ou la durée de la fenêtre pour les klines glissantes si window_size n'est pas fourni.
            rolling: Si True, génère des klines glissantes.
            window_size: Taille de la fenêtre pour les klines glissantes, en nombre de périodes de base du df.
                         Si rolling=True et window_size est fourni, il est utilisé directement.
                         Si rolling=True et window_size est None, il est déduit de target_freq 
                         (en supposant que df est en 1m et target_freq est convertible en minutes).
            
        Returns:
            DataFrame avec klines resamplées ou glissantes.
        """
        if df.empty:
            logger.warning("Input DataFrame for resampling is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
            
        timestamp_col_name = getattr(Kline, 'OHLCV_TIMESTAMP', 'timestamp') 
        current_df = df 

        if not isinstance(df.index, pd.DatetimeIndex):
            if timestamp_col_name in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col_name]):
                        current_df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name], errors='raise', unit='ms')
                    current_df = df.set_index(timestamp_col_name) 
                    logger.info(f"Set DataFrame index from '{timestamp_col_name}' column.")
                except Exception as e:
                    raise DataError(f"Index is not DatetimeIndex and '{timestamp_col_name}' column conversion failed: {e}")
            else: 
                try:
                    current_df = df.copy() 
                    current_df.index = pd.to_datetime(current_df.index, errors='raise') 
                    logger.info(f"Converted existing DataFrame index to DatetimeIndex.")
                except Exception as e:
                    raise DataError(
                        f"DataFrame index is not a DatetimeIndex, '{timestamp_col_name}' column not found. "
                        f"Failed to convert existing index: {e}"
                    )
        
        if not isinstance(current_df.index, pd.DatetimeIndex): 
            raise DataError("DataFrame index could not be ensured as DatetimeIndex.")

        if not current_df.index.is_monotonic_increasing:
            logger.debug("Input DataFrame index is not monotonic increasing. Sorting index.")
            current_df = current_df.sort_index()

        pandas_freq_target = self.interval_mapping.get(target_freq, target_freq)
        
        if rolling:
            actual_window_size = window_size
            if actual_window_size is None:
                if not pandas_freq_target:
                    raise DataError("target_freq is required to determine window_size for rolling klines when window_size is not explicitly provided.")
                try:
                    base_freq_minutes = 1 
                    if current_df.index.freq:
                        try:
                            base_freq_minutes = self._freq_to_minutes(current_df.index.freqstr)
                        except ValueError:
                            logger.warning(f"Could not parse base frequency '{current_df.index.freqstr}' of DataFrame. Assuming 1 minute for window calculation.")
                    
                    target_minutes = self._freq_to_minutes(pandas_freq_target)
                    actual_window_size = target_minutes // base_freq_minutes
                    if target_minutes % base_freq_minutes != 0:
                        logger.warning(f"Target frequency '{target_freq}' ({target_minutes}min) is not an exact multiple of base frequency ({base_freq_minutes}min). Window size rounded down to {actual_window_size}.")

                except ValueError as e:
                    raise DataError(f"Cannot determine window_size (number of periods) for rolling klines from target_freq '{target_freq}' (pandas_freq '{pandas_freq_target}'): {e}. Please provide an explicit integer window_size.")
            
            if not isinstance(actual_window_size, int) or actual_window_size <= 0:
                raise DataError(f"Window size for rolling klines must be a positive integer. Got: {actual_window_size}")
            
            return self._generate_rolling_klines(current_df, actual_window_size)
        else:
            return self._generate_standard_klines(current_df, pandas_freq_target)
            
    def _generate_standard_klines(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            required_cols_minimal = [
                Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, 
                Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME
            ]
            missing_cols = [col for col in required_cols_minimal if col not in df_copy.columns]
            if missing_cols:
                raise DataError(f"Missing required minimal columns for standard kline generation: {missing_cols}")
                
            current_aggregation_rules = {
                col: rule for col, rule in self.aggregation_rules.items() if col in df_copy.columns
            }
            if not current_aggregation_rules:
                raise DataError(
                    f"None of the columns specified in aggregation rules ({list(self.aggregation_rules.keys())}) "
                    f"were found in the input DataFrame. Cannot perform aggregation."
                )

            for col, rule in current_aggregation_rules.items():
                if rule in ["sum", "max", "min", "first", "last", "mean", "median", "std", "var"]: 
                    if not pd.api.types.is_numeric_dtype(df_copy[col]):
                        logger.debug(f"Standard klines: Column '{col}' is not numeric (dtype: {df_copy[col].dtype}). Attempting conversion for aggregation '{rule}'.")
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

            resampled = df_copy.resample(freq, label='left', closed='left').agg(current_aggregation_rules)
            resampled = resampled.dropna(subset=[Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE], how='all') 
            
            if not resampled.empty:
                if 'pair' in df_copy.columns and 'pair' not in resampled.columns:
                    if df_copy['pair'].nunique() == 1:
                         resampled['pair'] = df_copy['pair'].iloc[0]
                    else: 
                         logger.warning("'pair' column exists but varies or not handled by simple propagation. Taking first value per resampled period.")
                         resampled['pair'] = df_copy.resample(freq, label='left', closed='left')['pair'].first()
                
                is_closed_col = Kline.OHLCV_IS_KLINE_CLOSED
                resampled[is_closed_col] = True 

                close_time_col = Kline.OHLCV_KLINE_CLOSE_TIME
                try:
                    offset = pd.tseries.frequencies.to_offset(freq)
                    if offset:
                        resampled[close_time_col] = resampled.index + offset - pd.Timedelta(milliseconds=1)
                    else:
                        logger.warning(f"Could not determine offset from freq '{freq}' for {close_time_col}. Skipping {close_time_col} calculation.")
                        if close_time_col not in resampled.columns: resampled[close_time_col] = pd.NaT
                except Exception as ve: 
                    logger.warning(f"Could not create offset from freq '{freq}' for {close_time_col}: {ve}. Skipping {close_time_col} calculation.")
                    if close_time_col not in resampled.columns: resampled[close_time_col] = pd.NaT
            else: 
                expected_cols = list(current_aggregation_rules.keys())
                if 'pair' in df_copy.columns: expected_cols.append('pair')
                expected_cols.append(Kline.OHLCV_IS_KLINE_CLOSED)
                expected_cols.append(Kline.OHLCV_KLINE_CLOSE_TIME)
                resampled = pd.DataFrame(columns=list(dict.fromkeys(expected_cols)), index=resampled.index) 

            logger.debug(f"Generated {len(resampled)} standard klines at {freq} frequency")
            return resampled
            
        except Exception as e:
            logger.error(f"Error generating standard klines for frequency {freq}: {e}")
            raise DataError(f"Failed to generate standard klines: {e}", original_exception=getattr(e, 'original_exception', e))

    def _generate_rolling_klines(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        if not isinstance(window_size, int) or window_size <= 0:
            raise DataError(f"window_size must be a positive integer. Got: {window_size}")

        df_input_copy = df.copy()
        numeric_cols_to_ensure = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE,
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]
        for col in numeric_cols_to_ensure:
            if col in df_input_copy.columns:
                if not pd.api.types.is_numeric_dtype(df_input_copy[col]):
                    logger.debug(f"Rolling klines: Column '{col}' is not numeric (dtype: {df_input_copy[col].dtype}). Attempting conversion.")
                    df_input_copy[col] = pd.to_numeric(df_input_copy[col], errors='coerce')
                    if df_input_copy[col].isnull().all() and not df[col].isnull().all():
                        logger.warning(f"Conversion of column '{col}' to numeric resulted in all NaNs. Original had data.")
                    if not pd.api.types.is_numeric_dtype(df_input_copy[col]):
                        logger.error(f"Critical: Failed to convert column '{col}' to numeric for rolling calculation. Dtype remains {df_input_copy[col].dtype}.")
                
        expected_cols_output = []
        base_rolling_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME]
        for col in base_rolling_cols:
            if col in df_input_copy.columns: expected_cols_output.append(col)
        
        optional_sum_cols = [
            Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]
        for col in optional_sum_cols:
            if col in df_input_copy.columns and pd.api.types.is_numeric_dtype(df_input_copy[col]):
                expected_cols_output.append(col)
        
        expected_cols_output.append(Kline.OHLCV_KLINE_CLOSE_TIME)
        expected_cols_output.append(Kline.OHLCV_IS_KLINE_CLOSED)
        if 'pair' in df_input_copy.columns: expected_cols_output.append('pair')
        expected_cols_output = list(dict.fromkeys(expected_cols_output)) 

        if df_input_copy.empty or len(df_input_copy) < window_size:
            logger.warning(f"Input DataFrame is empty or shorter ({len(df_input_copy)}) than window_size ({window_size}) for rolling klines.")
            idx = pd.to_datetime([])
            if df_input_copy.index.tz:
                idx = idx.tz_localize(df_input_copy.index.tz)
            return pd.DataFrame(columns=expected_cols_output, index=idx)

        logger.debug(f"Generating rolling klines with window_size={window_size}")
        df_result = pd.DataFrame(index=df_input_copy.index)

        # --- Standard rolling aggregations (respect min_periods for non-NaN count) ---
        if Kline.OHLCV_HIGH in df_input_copy.columns:
            df_result[Kline.OHLCV_HIGH] = df_input_copy[Kline.OHLCV_HIGH].rolling(window=window_size, min_periods=window_size).max()
        if Kline.OHLCV_LOW in df_input_copy.columns:
            df_result[Kline.OHLCV_LOW] = df_input_copy[Kline.OHLCV_LOW].rolling(window=window_size, min_periods=window_size).min()
        if Kline.OHLCV_VOLUME in df_input_copy.columns:
            df_result[Kline.OHLCV_VOLUME] = df_input_copy[Kline.OHLCV_VOLUME].rolling(window=window_size, min_periods=window_size).sum()
        
        for col_name in optional_sum_cols:
            if col_name in df_input_copy.columns and pd.api.types.is_numeric_dtype(df_input_copy[col_name]):
                 if col_name not in df_result.columns: 
                    df_result[col_name] = df_input_copy[col_name].rolling(window=window_size, min_periods=window_size).sum()

        # --- Direct assignment for first/last type values with leading NaNs ---
        # These are independent of the min_periods behavior of other columns,
        # as long as a window of window_size rows exists.
        open_col = Kline.OHLCV_OPEN
        if open_col in df_input_copy.columns:
            s_open = pd.Series(np.nan, index=df_input_copy.index, dtype=df_input_copy[open_col].dtype)
            if len(df_input_copy) >= window_size:
                s_open.iloc[window_size-1:] = df_input_copy[open_col].iloc[0 : len(df_input_copy) - (window_size - 1)].values
            df_result[open_col] = s_open
        
        close_col = Kline.OHLCV_CLOSE
        if close_col in df_input_copy.columns:
            s_close = pd.Series(np.nan, index=df_input_copy.index, dtype=df_input_copy[close_col].dtype)
            if len(df_input_copy) >= window_size:
                s_close.iloc[window_size-1:] = df_input_copy[close_col].iloc[window_size-1 : len(df_input_copy)].values
            df_result[close_col] = s_close

        close_time_col = Kline.OHLCV_KLINE_CLOSE_TIME
        if close_time_col in df_input_copy.columns:
            s_close_time = pd.Series(pd.NaT, index=df_input_copy.index, dtype=df_input_copy[close_time_col].dtype)
            if len(df_input_copy) >= window_size:
                s_close_time.iloc[window_size-1:] = df_input_copy[close_time_col].iloc[window_size-1 : len(df_input_copy)] # Assign Series slice
            df_result[close_time_col] = s_close_time
        else:
            logger.warning(f"'{close_time_col}' not in input DataFrame. Cannot calculate rolling '{close_time_col}'.")
            if close_time_col in expected_cols_output and close_time_col not in df_result.columns:
                 df_result[close_time_col] = pd.NaT
        
        # --- Propagated columns (direct copy as per test) ---
        if 'pair' in df_input_copy.columns:
            df_result['pair'] = df_input_copy['pair'].copy() 
        
        # --- Propagated columns ---
        if 'pair' in df_input_copy.columns:
            # 'pair' is directly copied as per test_rolling_kline_propagated_and_static_columns's expectation
            df_result['pair'] = df_input_copy['pair'].copy()

        is_closed_col = Kline.OHLCV_IS_KLINE_CLOSED
        # OHLCV_IS_KLINE_CLOSED in the output rolling_df should indicate if a kline was actually computed for that row.
        # It should have pd.NA for the initial (window_size-1) rows where no complete window exists,
        # and True for subsequent rows where a rolling kline is formed.
        # This makes its NaN pattern consistent with 'open', 'close', and 'kline_close_time'
        # which also have leading NaNs due to how they are sliced.
        series_is_closed = pd.Series(pd.NA, index=df_input_copy.index, dtype='boolean')
        if len(df_input_copy) >= window_size:
            # Set to True for all rows from the first possible complete window onwards.
            # df_input_copy is guaranteed to be long enough here for iloc to be valid.
            series_is_closed.iloc[window_size-1:] = True
        df_result[is_closed_col] = series_is_closed


        for col in expected_cols_output:
            if col not in df_result.columns:
                logger.warning(f"Expected output column '{col}' was missing after processing. Filling with NaNs.")
                original_col_data = df_input_copy.get(col) 
                if original_col_data is not None and pd.api.types.is_datetime64_any_dtype(original_col_data): 
                    df_result[col] = pd.NaT
                elif original_col_data is not None and (pd.api.types.is_bool_dtype(original_col_data.dtype) or col == Kline.OHLCV_IS_KLINE_CLOSED) : 
                    df_result[col] = pd.NA 
                else: 
                    df_result[col] = np.nan
        
        final_cols_order = [col for col in expected_cols_output if col in df_result.columns]
        return df_result[final_cols_order]

    def _freq_to_minutes(self, freq_str: str) -> int:
        if freq_str is None:
            raise ValueError("Frequency string cannot be None.")
        if freq_str in self.interval_mapping:
            pandas_freq = self.interval_mapping[freq_str]
        else: 
            pandas_freq = freq_str
        try:
            offset = pd.tseries.frequencies.to_offset(pandas_freq)
            if hasattr(offset, 'delta'): 
                timedelta = offset.delta
            elif isinstance(offset, pd.offsets.Day):
                timedelta = pd.Timedelta(days=offset.n)
            elif isinstance(offset, pd.offsets.Hour):
                timedelta = pd.Timedelta(hours=offset.n)
            elif isinstance(offset, pd.offsets.Minute):
                timedelta = pd.Timedelta(minutes=offset.n)
            elif isinstance(offset, pd.offsets.Second):
                 timedelta = pd.Timedelta(seconds=offset.n)
            elif isinstance(offset, (pd.offsets.Week, pd.offsets.WeekOfMonth)): 
                timedelta = pd.Timedelta(days=7 * offset.n)
            elif isinstance(offset, (pd.offsets.MonthEnd, pd.offsets.MonthBegin, pd.offsets.BusinessMonthEnd, pd.offsets.BusinessMonthBegin)):
                logger.debug(f"Approximating month-based frequency '{pandas_freq}' as 30 days for minute conversion.")
                return offset.n * 30 * 1440 
            else: 
                timedelta = pd.Timedelta(pandas_freq) # Fallback
            
            return int(timedelta.total_seconds() / 60)

        except ValueError: 
            if pandas_freq.upper().endswith('M') and not pandas_freq.upper().endswith('MS'): 
                 num_part_match = re.match(r'(\d*)M', pandas_freq, re.IGNORECASE)
                 if num_part_match:
                     num_part = num_part_match.group(1)
                     number = int(num_part) if num_part else 1
                     logger.debug(f"Regex: Approximating month-based frequency '{pandas_freq}' as {number} * 30 days for minute conversion.")
                     return number * 30 * 1440 
                 raise ValueError(f"Cannot reliably convert monthly frequency '{pandas_freq}' to fixed minutes via regex.")
            if pandas_freq.upper().endswith('W'): 
                 num_part_match = re.match(r'(\d*)W', pandas_freq, re.IGNORECASE)
                 if num_part_match:
                     num_part = num_part_match.group(1)
                     number = int(num_part) if num_part else 1
                     return number * 7 * 1440
                 raise ValueError(f"Cannot reliably convert weekly frequency '{pandas_freq}' to fixed minutes via regex.")

            unit_mapping = {'T': 1, 'H': 60, 'D': 1440} 
            match = re.fullmatch(r'(\d*)([THD])S?', pandas_freq, re.IGNORECASE) 
            if match:
                number_str, unit_char = match.groups()
                number = int(number_str) if number_str else 1
                unit_char_upper = unit_char.upper()
                if unit_char_upper in unit_mapping:
                    return number * unit_mapping[unit_char_upper]
                else: 
                    raise ValueError(f"Unrecognized unit '{unit_char}' in frequency string '{pandas_freq}' via regex.")
            else:
                raise ValueError(f"Could not parse frequency string '{freq_str}' (interpreted as '{pandas_freq}') to minutes.")
            
    def validate_klines(
        self,
        df: pd.DataFrame,
        check_continuity: bool = True,
        expected_freq_minutes: Optional[int] = None 
    ) -> Tuple[bool, List[str]]:
        errors = []
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors 
        
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("DataFrame must have a DatetimeIndex")
            return False, errors 
        
        required_cols = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, 
            Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing essential kline columns: {missing_cols}")
            
        cols_to_check_nulls = [col for col in required_cols if col in df.columns]
        if cols_to_check_nulls: 
            null_counts = df[cols_to_check_nulls].isnull().sum()
            if null_counts.any():
                errors.append(f"Null values found in essential columns: {null_counts[null_counts > 0].to_dict()}")
                
        ohlc_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]
        if all(col in df.columns for col in ohlc_cols): 
            df_ohlc_check = df[ohlc_cols].dropna() 
            if not df_ohlc_check.empty:
                invalid_ohlc = df_ohlc_check[
                    (df_ohlc_check[Kline.OHLCV_HIGH] < df_ohlc_check[Kline.OHLCV_LOW]) |
                    (df_ohlc_check[Kline.OHLCV_HIGH] < df_ohlc_check[Kline.OHLCV_OPEN]) |
                    (df_ohlc_check[Kline.OHLCV_HIGH] < df_ohlc_check[Kline.OHLCV_CLOSE]) |
                    (df_ohlc_check[Kline.OHLCV_LOW] > df_ohlc_check[Kline.OHLCV_OPEN]) |
                    (df_ohlc_check[Kline.OHLCV_LOW] > df_ohlc_check[Kline.OHLCV_CLOSE])
                ]
                if not invalid_ohlc.empty:
                    errors.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows (after dropping NaNs from OHLC for this check)")
            
        volume_cols_to_check = [
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES
        ]
        for vol_col in volume_cols_to_check:
            if vol_col in df.columns and pd.api.types.is_numeric_dtype(df[vol_col]):
                if (df[vol_col].dropna() < 0).any(): 
                    errors.append(f"Negative values found in column '{vol_col}'")
            
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing")
        if df.index.has_duplicates:
            errors.append(f"Found {df.index.duplicated().sum()} duplicate timestamps in index")

        if check_continuity and len(df) > 1 and not df.index.has_duplicates and df.index.is_monotonic_increasing:
            time_diffs = df.index.to_series().diff().dropna()
            if not time_diffs.empty: 
                if expected_freq_minutes is not None:
                    expected_delta = pd.Timedelta(minutes=expected_freq_minutes)
                    tolerance = pd.Timedelta(seconds=max(1, int(expected_freq_minutes * 0.01 * 60))) 
                    
                    incorrect_intervals = time_diffs[
                        (time_diffs > expected_delta + tolerance) | (time_diffs < expected_delta - tolerance)
                    ]
                    if not incorrect_intervals.empty:
                        examples = incorrect_intervals.head(3).to_dict()
                        errors.append(
                            f"Found {len(incorrect_intervals)} time intervals not matching expected frequency "
                            f"of {expected_freq_minutes} min (expected delta: {expected_delta}, tolerance: {tolerance}). Examples: {examples}"
                        )
                else: 
                    inferred_freq_str = pd.infer_freq(df.index)
                    if inferred_freq_str:
                        try:
                            inferred_freq_minutes = self._freq_to_minutes(inferred_freq_str)
                            expected_delta_inferred = pd.Timedelta(minutes=inferred_freq_minutes)
                            tolerance_inferred = pd.Timedelta(seconds=max(1, int(inferred_freq_minutes * 0.01 * 60)))
                            
                            anomalous_gaps = time_diffs[time_diffs > expected_delta_inferred + tolerance_inferred]
                            if not anomalous_gaps.empty:
                                errors.append(f"Found {len(anomalous_gaps)} potential time gaps larger than inferred freq {inferred_freq_str} (delta: {expected_delta_inferred}, tolerance: {tolerance_inferred}).")
                        except ValueError:
                             errors.append(f"Could not create Timedelta or parse inferred_freq: {inferred_freq_str} for detailed continuity check.")
                    else: 
                        heuristic_base_freq_minutes = 1 
                        current_df_freq_str = getattr(df.index, 'freqstr', None)
                        if current_df_freq_str:
                            try: heuristic_base_freq_minutes = self._freq_to_minutes(current_df_freq_str)
                            except ValueError: pass
                        
                        if (time_diffs > pd.Timedelta(minutes=3 * heuristic_base_freq_minutes)).any():
                            errors.append(f"Could not infer frequency, and found some time gaps larger than 3x a heuristic base interval of {heuristic_base_freq_minutes} min.")
        
        is_valid = len(errors) == 0
        if not is_valid: 
            logger.debug(f"Kline validation summary: {len(errors)} error(s) found. Details: {errors}")
        return is_valid, errors
        
    def fill_missing_klines(
        self, df: pd.DataFrame, freq: str = '1T', 
        method: str = 'forward', limit_fill: Optional[int] = None
    ) -> pd.DataFrame:
        if df.empty: return df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex for fill_missing_klines.")
        
        pandas_freq = self.interval_mapping.get(freq, freq)
        df_sorted = df.sort_index()
        if df_sorted.empty: return df_sorted

        try:
            min_idx, max_idx = df_sorted.index.min(), df_sorted.index.max()
            if pd.isna(min_idx) or pd.isna(max_idx):
                 logger.warning("Cannot create date range due to NaT in min/max index for fill_missing_klines. Returning original df.")
                 return df
            full_index = pd.date_range(start=min_idx, end=max_idx, freq=pandas_freq)
        except Exception as e:
            logger.error(f"Error creating date range for fill_missing_klines (freq: {pandas_freq}): {e}")
            return df 

        df_reindexed = df_sorted.reindex(full_index)
        price_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]
        volume_cols = [
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]

        if method == 'forward':
            for col in price_cols:
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].ffill(limit=limit_fill)
            for col in volume_cols: 
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].fillna(0)
        elif method == 'interpolate': 
            for col in price_cols:
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].interpolate(method='linear', limit_area='inside', limit=limit_fill)
            for col in volume_cols:
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].fillna(0)
        elif method == 'zero_volume_carry_price': 
            close_col = Kline.OHLCV_CLOSE
            if close_col in df_reindexed.columns:
                if not pd.api.types.is_numeric_dtype(df_reindexed[close_col]):
                    df_reindexed[close_col] = pd.to_numeric(df_reindexed[close_col], errors='coerce')
                df_reindexed['close_price_ffill'] = df_reindexed[close_col].ffill(limit=limit_fill)
                for col in [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW]:
                    if col in df_reindexed.columns: 
                        df_reindexed[col] = df_reindexed[col].fillna(df_reindexed['close_price_ffill'])
                df_reindexed[close_col] = df_reindexed[close_col].fillna(df_reindexed['close_price_ffill']) 
                df_reindexed.drop(columns=['close_price_ffill'], inplace=True, errors='ignore')
            for col in volume_cols:
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].fillna(0)
        else:
            logger.warning(f"Unknown fill_missing_klines method: {method}. No filling performed beyond reindexing.")

        other_cols = [col for col in df_reindexed.columns if col not in price_cols + volume_cols]
        for col in other_cols: 
            if df_reindexed[col].isnull().any():
                if col == Kline.OHLCV_IS_KLINE_CLOSED: 
                    df_reindexed[col] = df_reindexed[col].fillna(True) 
                elif col == Kline.OHLCV_KLINE_CLOSE_TIME:
                    missing_close_time_mask = df_reindexed[col].isnull()
                    if missing_close_time_mask.any():
                        try:
                            offset = pd.tseries.frequencies.to_offset(pandas_freq)
                            df_reindexed.loc[missing_close_time_mask, col] = df_reindexed.index[missing_close_time_mask] + offset - pd.Timedelta(milliseconds=1)
                        except Exception: 
                            logger.debug(f"Could not calculate close time for filled klines with freq '{pandas_freq}'. Ffilling/Bfilling '{col}'.")
                            df_reindexed[col] = df_reindexed[col].ffill(limit=limit_fill).bfill(limit=limit_fill)
                else: 
                    df_reindexed[col] = df_reindexed[col].ffill(limit=limit_fill)
                    df_reindexed[col] = df_reindexed[col].bfill(limit=limit_fill) 
        return df_reindexed
        
    def merge_klines(self, df1: pd.DataFrame, df2: pd.DataFrame, prefer: str = 'df2') -> pd.DataFrame:
        if df1.empty and df2.empty: return pd.DataFrame(index=pd.DatetimeIndex([]))
        if df1.empty: return df2.sort_index()
        if df2.empty: return df1.sort_index()
        
        if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
            raise DataError("Both DataFrames must have a DatetimeIndex for merging.")
        
        if df1.index.tz != df2.index.tz:
            logger.warning(f"Timezone mismatch during merge: df1.tz={df1.index.tz}, df2.tz={df2.index.tz}. Attempting to align to df2's timezone or make naive if df2 is naive.")
            try:
                if df2.index.tz is not None: 
                    df1 = df1.tz_convert(df2.index.tz) if df1.index.tz is not None else df1.tz_localize(df2.index.tz)
                else: 
                    df1 = df1.tz_localize(None)
            except Exception as e:
                raise DataError(f"Failed to align timezones for merging: {e}")

        if prefer == 'df1':
            combined = pd.concat([df2, df1]) 
            combined = combined[~combined.index.duplicated(keep='last')] 
        elif prefer == 'df2': 
            combined = pd.concat([df1, df2])
            combined = combined[~combined.index.duplicated(keep='last')] 
        else:
            raise ValueError(f"Unsupported 'prefer' strategy: {prefer}. Use 'df1' or 'df2'.")
        
        combined.sort_index(inplace=True)
        return combined
        
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        if df.empty: return {"message": "DataFrame is empty", "count": 0}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex for statistics.")
        
        stats: Dict[str, any] = {'count': len(df), 'start_date': None, 'end_date': None, 'duration_days': None}
        if len(df) > 0:
            stats['start_date'] = df.index.min()
            stats['end_date'] = df.index.max()
            if pd.notna(stats['start_date']) and pd.notna(stats['end_date']):
                 stats['duration_days'] = (stats['end_date'] - stats['start_date']).days
        
        stat_configs = {
            'avg_price': (Kline.OHLCV_CLOSE, 'mean'), 'min_price': (Kline.OHLCV_LOW, 'min'), 
            'max_price': (Kline.OHLCV_HIGH, 'max'), 'price_volatility': (Kline.OHLCV_CLOSE, 'std'),
            'total_volume_base': (Kline.OHLCV_VOLUME, 'sum'), 'avg_volume_base': (Kline.OHLCV_VOLUME, 'mean'),
            'volume_base_volatility': (Kline.OHLCV_VOLUME, 'std'),
            'total_volume_quote': (Kline.OHLCV_QUOTE_ASSET_VOLUME, 'sum'),
            'avg_volume_quote': (Kline.OHLCV_QUOTE_ASSET_VOLUME, 'mean'),
            'volume_quote_volatility': (Kline.OHLCV_QUOTE_ASSET_VOLUME, 'std'),
            'total_trades': (Kline.OHLCV_NUMBER_OF_TRADES, 'sum'),
            'avg_trades': (Kline.OHLCV_NUMBER_OF_TRADES, 'mean'),
            'trades_volatility': (Kline.OHLCV_NUMBER_OF_TRADES, 'std'),
        }
        for stat_name, (col_name, func_name) in stat_configs.items():
            if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]) and not df[col_name].isnull().all():
                try:
                    if func_name == 'mean': stats[stat_name] = df[col_name].mean()
                    elif func_name == 'min': stats[stat_name] = df[col_name].min()
                    elif func_name == 'max': stats[stat_name] = df[col_name].max()
                    elif func_name == 'sum': stats[stat_name] = df[col_name].sum()
                    elif func_name == 'std': stats[stat_name] = df[col_name].std()
                except Exception as e: 
                    logger.warning(f"Could not calculate stat '{stat_name}' for col '{col_name}': {e}")
                    stats[stat_name] = np.nan
            else: stats[stat_name] = np.nan

        close_col = Kline.OHLCV_CLOSE
        if close_col in df.columns and pd.api.types.is_numeric_dtype(df[close_col]) and \
           len(df) > 1 and not df[close_col].isnull().all():
            valid_close_prices = df[close_col].dropna()
            if len(valid_close_prices) > 1:
                first_valid_price = valid_close_prices.iloc[0]
                if first_valid_price != 0: 
                    returns = valid_close_prices.pct_change().dropna()
                    if not returns.empty:
                        stats['avg_return'] = returns.mean()
                        stats['return_volatility'] = returns.std()
                        stats['total_return_pct'] = (valid_close_prices.iloc[-1] / first_valid_price - 1) * 100
                    else: stats.update({'avg_return': np.nan, 'return_volatility': np.nan, 'total_return_pct': np.nan})
                else: 
                    stats.update({'avg_return': np.nan, 'return_volatility': np.nan, 'total_return_pct': np.nan})
            else: stats.update({'avg_return': np.nan, 'return_volatility': np.nan, 'total_return_pct': np.nan})
        else: stats.update({'avg_return': np.nan, 'return_volatility': np.nan, 'total_return_pct': np.nan}) 
        return stats

def example_usage():
    import sys
    logger.remove() 
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}", level="DEBUG")

    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    periods = 100 
    index_name = Kline.OHLCV_TIMESTAMP 
    dates = pd.date_range(start=start_time, periods=periods, freq='1T', name=index_name)
    
    base_open = 40000.0 
    data = {
        Kline.OHLCV_OPEN: base_open + np.random.normal(0, 50, periods).cumsum(),
        Kline.OHLCV_VOLUME: np.random.uniform(0.1, 2.0, periods),
        Kline.OHLCV_NUMBER_OF_TRADES: np.random.randint(10, 200, periods).astype(float),
        'pair': 'BTCUSDT',
        Kline.OHLCV_KLINE_CLOSE_TIME: [d + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1) for d in dates]
    }
    df = pd.DataFrame(data, index=dates)
    
    df[Kline.OHLCV_LOW] = df[Kline.OHLCV_OPEN] - np.random.uniform(0, 100, periods)
    df[Kline.OHLCV_HIGH] = df[Kline.OHLCV_OPEN] + np.random.uniform(0, 100, periods)
    df[Kline.OHLCV_CLOSE] = df[Kline.OHLCV_OPEN] + np.random.normal(0, 20, periods)
    
    df[Kline.OHLCV_HIGH] = df[[Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_CLOSE]].max(axis=1)
    df[Kline.OHLCV_LOW] = df[[Kline.OHLCV_OPEN, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]].min(axis=1)
    df[Kline.OHLCV_HIGH] = np.maximum(df[Kline.OHLCV_HIGH], df[Kline.OHLCV_LOW])

    df[Kline.OHLCV_QUOTE_ASSET_VOLUME] = df[Kline.OHLCV_VOLUME] * ((df[Kline.OHLCV_OPEN] + df[Kline.OHLCV_CLOSE]) / 2) 
    df[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] = df[Kline.OHLCV_VOLUME] * np.random.uniform(0.4, 0.6, periods)
    df[Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME] = df[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] * ((df[Kline.OHLCV_OPEN] + df[Kline.OHLCV_CLOSE]) / 2)
    df[Kline.OHLCV_IS_KLINE_CLOSED] = True 

    for col in [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, 
                Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, 
                Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME,
                Kline.OHLCV_NUMBER_OF_TRADES]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if Kline.OHLCV_IS_KLINE_CLOSED in df.columns:
        df[Kline.OHLCV_IS_KLINE_CLOSED] = df[Kline.OHLCV_IS_KLINE_CLOSED].astype('boolean')


    logger.info(f"Sample 1m klines data created. Shape: {df.shape}")

    processor = KlineProcessor()
    
    try:
        logger.info("\nTest: Standard 5m klines...")
        klines_5m = processor.resample_klines(df.copy(), target_freq='5m', rolling=False)
        print(f"Generated {len(klines_5m)} standard 5m klines.")
        if not klines_5m.empty:
            is_valid, errors = processor.validate_klines(klines_5m, expected_freq_minutes=5)
        else:
            print("Standard 5m klines DataFrame is empty.")
    except DataError as e: 
        print(f"Error (5m standard): {e}")
        if hasattr(e, 'original_exception'): print(f"Original exception: {e.original_exception}")


    try:
        logger.info("\nTest: Rolling 10-period klines (based on 1m)...")
        klines_10p_rolling = processor.resample_klines(df.copy(), target_freq='10m', rolling=True, window_size=10) 
        print(f"Generated {len(klines_10p_rolling)} rolling 10-period klines.")
        if not klines_10p_rolling.empty:
            is_valid, errors = processor.validate_klines(klines_10p_rolling.dropna(subset=[Kline.OHLCV_OPEN]), check_continuity=False) 
        else:
            print("Rolling 10-period klines DataFrame is empty.")

    except DataError as e: 
        print(f"Error (10p rolling): {e}")
        if hasattr(e, 'original_exception'): print(f"Original exception: {e.original_exception}")

if __name__ == "__main__":
    example_usage()
