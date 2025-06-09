# src/data/kline_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone
from loguru import logger
import re

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
            '1d': '1D', '3d': '3D', '1w': '1W', 
            '1M': 'MS' 
        }
        
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
        """
        if df.empty:
            logger.warning("Input DataFrame for resampling is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
            
        current_df = df.copy() 

        if not isinstance(current_df.index, pd.DatetimeIndex):
            raise DataError(f"Input DataFrame must have a DatetimeIndex. Index type found: {type(current_df.index)}")
        if current_df.index.name != Kline.OHLCV_TIMESTAMP:
            logger.warning(f"Input DataFrame index name is '{current_df.index.name}', expected '{Kline.OHLCV_TIMESTAMP}'. Renaming.")
            current_df.index.name = Kline.OHLCV_TIMESTAMP
        if current_df.index.tz is None or current_df.index.tz != timezone.utc:
            logger.warning(f"Input DataFrame index timezone is '{current_df.index.tz}', expected UTC. Converting to UTC.")
            current_df.index = current_df.index.tz_localize('UTC') if current_df.index.tz is None else current_df.index.tz_convert('UTC')

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
                    base_freq_str = pd.infer_freq(current_df.index)
                    if not base_freq_str:
                        if len(current_df.index) >= 2:
                            diffs = current_df.index.to_series().diff().dropna()
                            if not diffs.empty:
                                base_freq_str = pd.tseries.frequencies.to_offset(diffs.mode()[0]).freqstr if not diffs.mode().empty else "1T" 
                            else: base_freq_str = "1T" 
                        else: base_freq_str = "1T" 
                        logger.warning(f"Could not reliably infer base frequency of DataFrame. Assuming '{base_freq_str}' for window calculation.")
                    
                    base_freq_minutes = self._freq_to_minutes(base_freq_str)
                    target_minutes = self._freq_to_minutes(pandas_freq_target) 
                    
                    if base_freq_minutes == 0:
                         raise DataError(f"Base frequency '{base_freq_str}' results in zero minutes, cannot calculate rolling window size relative to it.")

                    actual_window_size = target_minutes // base_freq_minutes
                    if target_minutes % base_freq_minutes != 0:
                        logger.warning(f"Target frequency '{target_freq}' ({target_minutes}min) is not an exact multiple of base frequency '{base_freq_str}' ({base_freq_minutes}min). Window size rounded down to {actual_window_size}.")
                except ValueError as e:
                    raise DataError(f"Cannot determine window_size (number of periods) for rolling klines from target_freq '{target_freq}': {e}. Please provide an explicit integer window_size.")
            
            if not isinstance(actual_window_size, int) or actual_window_size <= 0:
                raise DataError(f"Window size for rolling klines must be a positive integer. Got: {actual_window_size}")
            
            return self._generate_rolling_klines(current_df, actual_window_size)
        else:
            return self._generate_standard_klines(current_df, pandas_freq_target)
            
    def _generate_standard_klines(self, df: pd.DataFrame, pandas_freq: str) -> pd.DataFrame:
        """Génère des klines standard par rééchantillonnage."""
        try:
            required_cols_minimal = [
                Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, 
                Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME
            ]
            missing_cols = [col for col in required_cols_minimal if col not in df.columns]
            if missing_cols:
                raise DataError(f"Missing required minimal columns for standard kline generation: {missing_cols}")
                
            current_aggregation_rules = {
                col: rule for col, rule in self.aggregation_rules.items() if col in df.columns
            }
            if not current_aggregation_rules:
                raise DataError(
                    f"None of the columns specified in aggregation rules ({list(self.aggregation_rules.keys())}) "
                    f"were found in the input DataFrame. Cannot perform aggregation."
                )

            for col, rule in current_aggregation_rules.items():
                if rule in ["sum", "max", "min", "mean", "median", "std", "var"]: 
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.debug(f"Standard klines: Column '{col}' is not numeric (dtype: {df[col].dtype}). Attempting conversion for aggregation '{rule}'.")
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e_conv:
                            raise DataError(f"Failed to convert column '{col}' to numeric for aggregation: {e_conv}")

            resampled_df = df.resample(pandas_freq, label='left', closed='left').agg(current_aggregation_rules)
            resampled_df = resampled_df.dropna(subset=[Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE], how='all') 
            
            if not resampled_df.empty:
                if Kline.PAIR in df.columns:
                    if df[Kline.PAIR].nunique() == 1:
                        resampled_df[Kline.PAIR] = df[Kline.PAIR].iloc[0]
                    else: 
                        logger.warning(f"'{Kline.PAIR}' column has multiple unique values. Propagating first value per resampled period.")
                        resampled_df[Kline.PAIR] = df.resample(pandas_freq, label='left', closed='left')[Kline.PAIR].first()
                
                resampled_df[Kline.OHLCV_IS_KLINE_CLOSED] = True 

                try:
                    offset = pd.tseries.frequencies.to_offset(pandas_freq)
                    if offset:
                        resampled_df[Kline.OHLCV_KLINE_CLOSE_TIME] = resampled_df.index + offset - pd.Timedelta(milliseconds=1)
                    else:
                        raise ValueError(f"Could not create offset from pandas_freq '{pandas_freq}'")
                except Exception as e_offset: 
                    logger.warning(f"Could not calculate '{Kline.OHLCV_KLINE_CLOSE_TIME}' for resampled klines (freq '{pandas_freq}'): {e_offset}. Column will be NaT.")
                    resampled_df[Kline.OHLCV_KLINE_CLOSE_TIME] = pd.NaT
            else: 
                expected_cols = list(current_aggregation_rules.keys())
                if Kline.PAIR in df.columns: expected_cols.append(Kline.PAIR)
                expected_cols.append(Kline.OHLCV_IS_KLINE_CLOSED)
                expected_cols.append(Kline.OHLCV_KLINE_CLOSE_TIME)
                resampled_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols)), index=resampled_df.index.copy()) 

            logger.debug(f"Generated {len(resampled_df)} standard klines at {pandas_freq} frequency")
            return resampled_df
            
        except Exception as e:
            logger.error(f"Error generating standard klines for frequency {pandas_freq}: {e}", exc_info=True)
            raise DataError(f"Failed to generate standard klines: {e}", original_exception=getattr(e, 'original_exception', e))

    def _generate_rolling_klines(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """Génère des klines glissantes."""
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
                    try:
                        df_input_copy[col] = pd.to_numeric(df_input_copy[col], errors='coerce')
                    except Exception as e_conv_roll:
                         raise DataError(f"Failed to convert column '{col}' to numeric for rolling kline generation: {e_conv_roll}")

        expected_output_cols = []
        base_rolling_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME]
        for col in base_rolling_cols:
            if col in df_input_copy.columns: expected_output_cols.append(col)
        
        optional_sum_cols = [
            Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]
        for col in optional_sum_cols:
            if col in df_input_copy.columns and pd.api.types.is_numeric_dtype(df_input_copy[col]):
                expected_output_cols.append(col)
        
        expected_output_cols.extend([Kline.OHLCV_KLINE_CLOSE_TIME, Kline.OHLCV_IS_KLINE_CLOSED])
        if Kline.PAIR in df_input_copy.columns: expected_output_cols.append(Kline.PAIR)
        expected_output_cols = list(dict.fromkeys(expected_output_cols)) 

        if df_input_copy.empty or len(df_input_copy) < window_size:
            logger.warning(f"Input DataFrame is empty or shorter ({len(df_input_copy)}) than window_size ({window_size}) for rolling klines. Returning empty DataFrame with expected columns.")
            idx = pd.to_datetime([])
            if df_input_copy.index.tz: idx = idx.tz_localize(df_input_copy.index.tz)
            return pd.DataFrame(columns=expected_output_cols, index=idx)

        logger.debug(f"Generating rolling klines with window_size={window_size}")
        df_result = pd.DataFrame(index=df_input_copy.index)

        # Agrégations standard avec rolling
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

        if Kline.OHLCV_OPEN in df_input_copy.columns:
            df_result[Kline.OHLCV_OPEN] = df_input_copy[Kline.OHLCV_OPEN].rolling(window=window_size, min_periods=window_size).apply(lambda x: x.iloc[0], raw=True)
        if Kline.OHLCV_CLOSE in df_input_copy.columns:
            df_result[Kline.OHLCV_CLOSE] = df_input_copy[Kline.OHLCV_CLOSE].rolling(window=window_size, min_periods=window_size).apply(lambda x: x.iloc[-1], raw=True)
        if Kline.OHLCV_KLINE_CLOSE_TIME in df_input_copy.columns:
            df_result[Kline.OHLCV_KLINE_CLOSE_TIME] = df_input_copy[Kline.OHLCV_KLINE_CLOSE_TIME].rolling(window=window_size, min_periods=window_size).apply(lambda x: x.iloc[-1], raw=False) 
        
        if Kline.PAIR in df_input_copy.columns:
            df_result[Kline.PAIR] = df_input_copy[Kline.PAIR] 
        
        df_result[Kline.OHLCV_IS_KLINE_CLOSED] = False 
        if Kline.OHLCV_HIGH in df_result.columns: # Se baser sur une colonne qui utilise min_periods=window_size
             df_result.loc[df_result[Kline.OHLCV_HIGH].notna(), Kline.OHLCV_IS_KLINE_CLOSED] = True
        elif Kline.OHLCV_CLOSE in df_result.columns : 
             df_result.loc[df_result[Kline.OHLCV_CLOSE].notna(), Kline.OHLCV_IS_KLINE_CLOSED] = True

        for col in expected_output_cols:
            if col not in df_result.columns:
                logger.warning(f"Rolling klines: Expected output column '{col}' was missing after processing. Filling appropriately.")
                original_col_data = df_input_copy.get(col) 
                if original_col_data is not None and pd.api.types.is_datetime64_any_dtype(original_col_data): 
                    df_result[col] = pd.NaT
                elif original_col_data is not None and (pd.api.types.is_bool_dtype(original_col_data.dtype) or col == Kline.OHLCV_IS_KLINE_CLOSED) : 
                    df_result[col] = pd.NA 
                else: 
                    df_result[col] = np.nan
        
        # Retourner le DataFrame avec l'ordre des colonnes attendu et seulement après les NaNs initiaux du rolling
        # Le .iloc[window_size-1:] supprime les premières lignes où les fenêtres glissantes ne sont pas complètes.
        return df_result[expected_output_cols].iloc[window_size-1:]

    def _freq_to_minutes(self, freq_str: str) -> int:
        """Convertit une chaîne de fréquence (style Binance ou Pandas) en minutes entières."""
        if not isinstance(freq_str, str):
            raise ValueError(f"Frequency string must be a string, got {type(freq_str)}")

        pandas_freq = self.interval_mapping.get(freq_str, freq_str)
        
        try:
            timedelta_obj = pd.Timedelta(pandas_freq)
            if pd.isna(timedelta_obj): 
                raise ValueError(f"Cannot create fixed Timedelta for frequency '{pandas_freq}' (e.g., monthly).")
            total_seconds = timedelta_obj.total_seconds()
            if total_seconds == 0 and pandas_freq != '0S': # '0S' est valide mais 0 minute
                raise ValueError(f"Frequency '{pandas_freq}' results in zero duration.")
            return int(total_seconds / 60)
        except ValueError:
            unit = pandas_freq[-1].upper()
            num_str = pandas_freq[:-1]
            num = int(num_str) if num_str.isdigit() else 1

            if unit == 'M': 
                return num * 30 * 24 * 60 
            elif unit == 'W': 
                return num * 7 * 24 * 60
            else:
                raise ValueError(f"Could not parse frequency string '{freq_str}' (Pandas alias '{pandas_freq}') to minutes.")
            
    def validate_klines(
        self,
        df: pd.DataFrame,
        check_continuity: bool = True,
        expected_freq_minutes: Optional[int] = None 
    ) -> Tuple[bool, List[str]]:
        """Valide un DataFrame de klines."""
        errors: List[str] = []
        if not isinstance(df, pd.DataFrame):
            errors.append(f"Input is not a DataFrame (type: {type(df)}).")
            return False, errors 

        if df.empty:
            return True, errors 
        
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("DataFrame must have a DatetimeIndex.")
            return False, errors 
        
        if df.index.name != Kline.OHLCV_TIMESTAMP:
             errors.append(f"DataFrame index name is '{df.index.name}', expected '{Kline.OHLCV_TIMESTAMP}'.")

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
        if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in ohlc_cols): 
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
                    errors.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows (checked on non-NaN OHLC rows). First 3 examples at indices: {invalid_ohlc.head(3).index.tolist()}")
        else:
            errors.append(f"Skipping OHLC consistency check due to missing or non-numeric OHLC columns among: {ohlc_cols}")
            
        volume_cols_to_check = [
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES
        ]
        for vol_col in volume_cols_to_check:
            if vol_col in df.columns and pd.api.types.is_numeric_dtype(df[vol_col]):
                if (df[vol_col].dropna() < 0).any(): 
                    errors.append(f"Negative values found in column '{vol_col}'")
            
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing (data not sorted by time).")
        if df.index.has_duplicates:
            errors.append(f"Found {df.index.duplicated().sum()} duplicate timestamps in index.")

        if check_continuity and len(df.index.drop_duplicates()) > 1 : 
            # Ensure index is sorted before diffing, and use unique index values for diff
            sorted_unique_index = df.index.drop_duplicates().sort_values()
            if len(sorted_unique_index) > 1:
                time_diffs = sorted_unique_index.to_series().diff().dropna()
            
                if not time_diffs.empty: 
                    current_expected_freq_minutes = expected_freq_minutes
                    if current_expected_freq_minutes is None: # Try to infer if not provided
                        inferred_pandas_freq = pd.infer_freq(sorted_unique_index) # Infer from sorted unique index
                        if inferred_pandas_freq:
                            try:
                                current_expected_freq_minutes = self._freq_to_minutes(inferred_pandas_freq)
                            except ValueError:
                                logger.debug(f"Could not convert inferred frequency '{inferred_pandas_freq}' to minutes for continuity check.")
                        else: 
                            logger.debug("Could not infer frequency for continuity check (index may be irregular or too short).")

                    if current_expected_freq_minutes is not None and current_expected_freq_minutes > 0:
                        expected_delta = pd.Timedelta(minutes=current_expected_freq_minutes)
                        tolerance_seconds = max(1, int(current_expected_freq_minutes * 60 * 0.01))
                        tolerance = pd.Timedelta(seconds=tolerance_seconds)
                        
                        incorrect_intervals = time_diffs[
                            (time_diffs > expected_delta + tolerance) | (time_diffs < expected_delta - tolerance)
                        ]
                        if not incorrect_intervals.empty:
                            examples = {idx.isoformat(): val.isoformat() for idx, val in incorrect_intervals.head(3).items()} # Format for readability
                            errors.append(
                                f"Found {len(incorrect_intervals)} time intervals not matching expected frequency "
                                f"of {current_expected_freq_minutes} min (expected delta: {expected_delta}, tolerance: {tolerance}). Examples: {examples}"
                            )
                    elif expected_freq_minutes is None: 
                         if not time_diffs.empty:
                            median_diff_minutes = time_diffs.median().total_seconds() / 60
                            if median_diff_minutes > 0:
                                large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=median_diff_minutes * 5)] 
                                if not large_gaps.empty:
                                    errors.append(f"Found {len(large_gaps)} potential large time gaps (more than 5x median diff of ~{median_diff_minutes:.1f} min).")
        
        is_valid_result = len(errors) == 0
        if not is_valid_result: 
            logger.debug(f"Kline validation summary: {len(errors)} error(s) found. Details: {errors}")
        return is_valid_result, errors
        
    def fill_missing_klines(
        self, df: pd.DataFrame, freq: str = '1T', 
        method: str = 'forward', limit_fill: Optional[int] = None
    ) -> pd.DataFrame:
        """Remplit les klines manquantes dans un DataFrame."""
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
            if full_index.tz is None and df_sorted.index.tz is not None: # Ensure consistent timezone
                full_index = full_index.tz_localize(df_sorted.index.tz)

        except Exception as e:
            logger.error(f"Error creating date range for fill_missing_klines (freq: {pandas_freq}, min: {min_idx}, max: {max_idx}): {e}")
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
                
                ffilled_close = df_reindexed[close_col].ffill(limit=limit_fill)
                for col_price in [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]:
                    if col_price in df_reindexed.columns:
                        df_reindexed[col_price] = df_reindexed[col_price].fillna(ffilled_close)
            for col_vol in volume_cols:
                if col_vol in df_reindexed.columns: df_reindexed[col_vol] = df_reindexed[col_vol].fillna(0)
        else:
            logger.warning(f"Unknown fill_missing_klines method: {method}. No filling performed beyond reindexing.")

        other_cols = [col for col in df_reindexed.columns if col not in price_cols + volume_cols]
        for col in other_cols: 
            if df_reindexed[col].isnull().any(): 
                if col == Kline.OHLCV_IS_KLINE_CLOSED: 
                    df_reindexed[col] = df_reindexed[col].fillna(True).astype('boolean') 
                elif col == Kline.OHLCV_KLINE_CLOSE_TIME:
                    missing_close_time_mask = df_reindexed[col].isnull()
                    if missing_close_time_mask.any():
                        try:
                            offset = pd.tseries.frequencies.to_offset(pandas_freq)
                            df_reindexed.loc[missing_close_time_mask, col] = df_reindexed.index[missing_close_time_mask] + offset - pd.Timedelta(milliseconds=1)
                        except Exception as e_offset_fill: 
                            logger.debug(f"Could not calculate close time for filled klines with freq '{pandas_freq}': {e_offset_fill}. Ffilling/Bfilling '{col}'.")
                            df_reindexed[col] = df_reindexed[col].ffill(limit=limit_fill).bfill(limit=limit_fill)
                elif col == Kline.PAIR: 
                     df_reindexed[col] = df_reindexed[col].ffill(limit=limit_fill).bfill(limit=limit_fill)
                else: 
                    df_reindexed[col] = df_reindexed[col].ffill(limit=limit_fill)
                    df_reindexed[col] = df_reindexed[col].bfill(limit=limit_fill) 
        return df_reindexed
        
    def merge_klines(self, df1: pd.DataFrame, df2: pd.DataFrame, prefer: str = 'df2') -> pd.DataFrame:
        """Fusionne deux DataFrames de klines, en donnant la préférence à l'un en cas de doublons d'index."""
        if df1.empty and df2.empty: return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC')) # Ensure UTC index for empty
        if df1.empty: return df2.sort_index()
        if df2.empty: return df1.sort_index()
        
        if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
            raise DataError("Both DataFrames must have a DatetimeIndex for merging.")
        
        # Assurer la cohérence des timezones (convertir vers UTC si nécessaire)
        df1_processed = df1.copy()
        df2_processed = df2.copy()

        if df1_processed.index.tz is None: df1_processed.index = df1_processed.index.tz_localize('UTC')
        elif df1_processed.index.tz != timezone.utc: df1_processed.index = df1_processed.index.tz_convert('UTC')

        if df2_processed.index.tz is None: df2_processed.index = df2_processed.index.tz_localize('UTC')
        elif df2_processed.index.tz != timezone.utc: df2_processed.index = df2_processed.index.tz_convert('UTC')


        if prefer == 'df1': 
            combined = pd.concat([df2_processed, df1_processed]) 
            combined = combined[~combined.index.duplicated(keep='last')] 
        elif prefer == 'df2': 
            combined = pd.concat([df1_processed, df2_processed])
            combined = combined[~combined.index.duplicated(keep='last')] 
        else:
            raise ValueError(f"Unsupported 'prefer' strategy: {prefer}. Use 'df1' or 'df2'.")
        
        combined.sort_index(inplace=True)
        return combined
        
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calcule des statistiques descriptives sur un DataFrame de klines."""
        if df.empty: return {"message": "DataFrame is empty", "count": 0}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex for statistics.")
        
        stats: Dict[str, any] = {'count': len(df), 'start_date': None, 'end_date': None, 'duration_days': None}
        if len(df) > 0:
            min_date = df.index.min()
            max_date = df.index.max()
            stats['start_date'] = min_date.to_pydatetime() if pd.notna(min_date) else None
            stats['end_date'] = max_date.to_pydatetime() if pd.notna(max_date) else None
            if stats['start_date'] and stats['end_date']:
                stats['duration_days'] = (stats['end_date'] - stats['start_date']).days
        
        stat_configs = {
            'avg_price': (Kline.OHLCV_CLOSE, 'mean'), 'min_price': (Kline.OHLCV_LOW, 'min'), 
            'max_price': (Kline.OHLCV_HIGH, 'max'), 'price_volatility_std': (Kline.OHLCV_CLOSE, 'std'),
            'total_volume_base': (Kline.OHLCV_VOLUME, 'sum'), 'avg_volume_base': (Kline.OHLCV_VOLUME, 'mean'),
            'volume_base_volatility_std': (Kline.OHLCV_VOLUME, 'std'),
            'total_volume_quote': (Kline.OHLCV_QUOTE_ASSET_VOLUME, 'sum'),
            'avg_volume_quote': (Kline.OHLCV_QUOTE_ASSET_VOLUME, 'mean'),
            'volume_quote_volatility_std': (Kline.OHLCV_QUOTE_ASSET_VOLUME, 'std'),
            'total_trades': (Kline.OHLCV_NUMBER_OF_TRADES, 'sum'),
            'avg_trades': (Kline.OHLCV_NUMBER_OF_TRADES, 'mean'),
            'trades_volatility_std': (Kline.OHLCV_NUMBER_OF_TRADES, 'std'),
        }
        for stat_name, (col_name, func_name) in stat_configs.items():
            if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]) and not df[col_name].isnull().all():
                try:
                    stats[stat_name] = getattr(df[col_name], func_name)()
                except Exception as e_stat: 
                    logger.warning(f"Could not calculate stat '{stat_name}' for col '{col_name}': {e_stat}")
                    stats[stat_name] = np.nan
            else: 
                stats[stat_name] = np.nan 

        close_col = Kline.OHLCV_CLOSE
        if close_col in df.columns and pd.api.types.is_numeric_dtype(df[close_col]) and \
           len(df) > 1 and not df[close_col].isnull().all():
            valid_close_prices = df[close_col].dropna()
            if len(valid_close_prices) > 1:
                first_valid_price = valid_close_prices.iloc[0]
                if first_valid_price != 0: 
                    returns = valid_close_prices.pct_change().dropna()
                    if not returns.empty:
                        stats['avg_return_pct'] = returns.mean() * 100
                        stats['return_volatility_std_pct'] = returns.std() * 100
                        stats['total_return_pct'] = (valid_close_prices.iloc[-1] / first_valid_price - 1) * 100
                    else: stats.update({'avg_return_pct': np.nan, 'return_volatility_std_pct': np.nan, 'total_return_pct': np.nan})
                else: 
                    stats.update({'avg_return_pct': np.nan, 'return_volatility_std_pct': np.nan, 'total_return_pct': np.nan})
            else: stats.update({'avg_return_pct': np.nan, 'return_volatility_std_pct': np.nan, 'total_return_pct': np.nan})
        else: stats.update({'avg_return_pct': np.nan, 'return_volatility_std_pct': np.nan, 'total_return_pct': np.nan}) 
        return stats

# Example usage for manual testing (optional)
def _example_usage_kline_processor():
    import sys
    logger.remove() 
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}", level="DEBUG")

    start_time_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    periods_num = 100 
    dates_idx = pd.date_range(start=start_time_dt, periods=periods_num, freq='1T', name=Kline.OHLCV_TIMESTAMP) # Utiliser '1T' pour minutes
    
    base_open_val = 40000.0 
    data_dict = {
        Kline.OHLCV_OPEN: base_open_val + np.random.normal(0, 50, periods_num).cumsum(),
        Kline.OHLCV_VOLUME: np.random.uniform(0.1, 2.0, periods_num),
        Kline.OHLCV_NUMBER_OF_TRADES: np.random.randint(10, 200, periods_num).astype(float),
        Kline.PAIR: 'BTCUSDT', 
        Kline.OHLCV_KLINE_CLOSE_TIME: [d + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1) for d in dates_idx]
    }
    df_klines = pd.DataFrame(data_dict, index=dates_idx)
    
    df_klines[Kline.OHLCV_LOW] = df_klines[Kline.OHLCV_OPEN] - np.random.uniform(0, 100, periods_num)
    df_klines[Kline.OHLCV_HIGH] = df_klines[Kline.OHLCV_OPEN] + np.random.uniform(0, 100, periods_num)
    df_klines[Kline.OHLCV_CLOSE] = df_klines[Kline.OHLCV_OPEN] + np.random.normal(0, 20, periods_num)
    
    df_klines[Kline.OHLCV_HIGH] = df_klines[[Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_CLOSE]].max(axis=1)
    df_klines[Kline.OHLCV_LOW] = df_klines[[Kline.OHLCV_OPEN, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]].min(axis=1)
    df_klines[Kline.OHLCV_HIGH] = np.maximum(df_klines[Kline.OHLCV_HIGH], df_klines[Kline.OHLCV_LOW])

    df_klines[Kline.OHLCV_QUOTE_ASSET_VOLUME] = df_klines[Kline.OHLCV_VOLUME] * ((df_klines[Kline.OHLCV_OPEN] + df_klines[Kline.OHLCV_CLOSE]) / 2) 
    df_klines[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] = df_klines[Kline.OHLCV_VOLUME] * np.random.uniform(0.4, 0.6, periods_num)
    df_klines[Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME] = df_klines[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] * ((df_klines[Kline.OHLCV_OPEN] + df_klines[Kline.OHLCV_CLOSE]) / 2)
    df_klines[Kline.OHLCV_IS_KLINE_CLOSED] = True 

    numeric_cols_list = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, 
                     Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, 
                     Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME,
                     Kline.OHLCV_NUMBER_OF_TRADES]
    for col_item in numeric_cols_list:
        if col_item in df_klines.columns:
            df_klines[col_item] = pd.to_numeric(df_klines[col_item], errors='coerce')
    if Kline.OHLCV_IS_KLINE_CLOSED in df_klines.columns: 
        df_klines[Kline.OHLCV_IS_KLINE_CLOSED] = df_klines[Kline.OHLCV_IS_KLINE_CLOSED].astype('boolean')


    logger.info(f"Sample 1m klines data created. Shape: {df_klines.shape}. Columns: {df_klines.columns.tolist()}")
    logger.info(f"Sample data head:\n{df_klines.head()}")

    processor = KlineProcessor()
    
    is_valid_source, errors_source = processor.validate_klines(df_klines.copy(), expected_freq_minutes=1)
    logger.info(f"Source 1m data validation: Is Valid = {is_valid_source}, Errors = {errors_source}")

    try:
        logger.info("\nTest: Standard 5m klines...")
        klines_5m_std = processor.resample_klines(df_klines.copy(), target_freq='5m', rolling=False)
        logger.info(f"Generated {len(klines_5m_std)} standard 5m klines. Columns: {klines_5m_std.columns.tolist()}")
        if not klines_5m_std.empty:
            logger.info(f"Sample 5m standard klines:\n{klines_5m_std.head()}")
            is_valid_5m, errors_5m = processor.validate_klines(klines_5m_std, expected_freq_minutes=5)
            logger.info(f"5m standard klines validation: Is Valid = {is_valid_5m}, Errors = {errors_5m}")
        else:
            logger.warning("Standard 5m klines DataFrame is empty.")
    except DataError as e_5m: 
        logger.error(f"Error generating 5m standard klines: {e_5m}")
        if hasattr(e_5m, 'original_exception'): logger.error(f"Original exception: {e_5m.original_exception}")

    try:
        logger.info("\nTest: Rolling 10-period klines (based on 1m source)...")
        klines_10p_roll = processor.resample_klines(df_klines.copy(), target_freq='10m', rolling=True, window_size=None) 
        logger.info(f"Generated {len(klines_10p_roll)} rolling 10-period klines. Columns: {klines_10p_roll.columns.tolist()}")
        if not klines_10p_roll.empty:
            logger.info(f"Sample 10-period rolling klines:\n{klines_10p_roll.head()}")
            is_valid_10r, errors_10r = processor.validate_klines(klines_10p_roll.dropna(subset=[Kline.OHLCV_OPEN]), check_continuity=True, expected_freq_minutes=1) 
            logger.info(f"Rolling 10-period klines validation: Is Valid = {is_valid_10r}, Errors = {errors_10r}")
        else:
            logger.warning("Rolling 10-period klines DataFrame is empty.")

    except DataError as e_10r: 
        logger.error(f"Error generating 10-period rolling klines: {e_10r}")
        if hasattr(e_10r, 'original_exception'): logger.error(f"Original exception: {e_10r.original_exception}")

if __name__ == "__main__":
    _example_usage_kline_processor()
