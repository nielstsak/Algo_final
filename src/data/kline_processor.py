import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone # Ajout de timezone pour example_usage
from loguru import logger

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
            
        # Ensure DataFrame index is DatetimeIndex
        # The column name for open time (if not index) should come from constants if defined.
        # Assuming Kline.OHLCV_TIMESTAMP is the constant for the open time column if it's not the index.
        # If not defined, fallback to 'kline_open_time' for robustness with older data.
        timestamp_col_name = getattr(Kline, 'OHLCV_TIMESTAMP_COLUMN', 'kline_open_time') # Or a more specific constant if available

        if not isinstance(df.index, pd.DatetimeIndex):
            if timestamp_col_name in df.columns:
                try:
                    # Ensure the column is in datetime format before setting as index
                    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col_name]):
                         df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name], unit='ms', errors='raise')
                    current_df = df.set_index(timestamp_col_name) # Use a new var to avoid modifying original df if it's passed around
                    logger.info(f"Set DataFrame index from '{timestamp_col_name}' column.")
                except Exception as e:
                    raise DataError(f"Index is not DatetimeIndex and '{timestamp_col_name}' column conversion failed: {e}")
            else: # Fallback to trying to convert the existing index
                try:
                    current_df = df.copy() # Work on a copy
                    current_df.index = pd.to_datetime(current_df.index, errors='raise') 
                    logger.info(f"Converted existing DataFrame index to DatetimeIndex.")
                except Exception as e:
                     raise DataError(
                        f"DataFrame index is not a DatetimeIndex, '{timestamp_col_name}' column not found. "
                        f"Failed to convert existing index: {e}"
                    )
        else:
            current_df = df # Already has DatetimeIndex

        if not isinstance(current_df.index, pd.DatetimeIndex): # Final check
            raise DataError("DataFrame index could not be ensured as DatetimeIndex.")


        pandas_freq_target = self.interval_mapping.get(target_freq, target_freq)
        
        if rolling:
            actual_window_size = window_size
            if actual_window_size is None:
                if not pandas_freq_target:
                    raise DataError("target_freq is required to determine window_size for rolling klines when window_size is not explicitly provided.")
                try:
                    actual_window_size = self._freq_to_minutes(pandas_freq_target)
                except ValueError as e:
                    raise DataError(f"Cannot determine window_size (number of periods) for rolling klines from target_freq '{target_freq}' (pandas_freq '{pandas_freq_target}'): {e}. Please provide an explicit integer window_size.")
            
            if not isinstance(actual_window_size, int) or actual_window_size <= 0:
                raise DataError(f"Window size for rolling klines must be a positive integer. Got: {actual_window_size}")
            
            return self._generate_rolling_klines(current_df, actual_window_size)
        else:
            return self._generate_standard_klines(current_df, pandas_freq_target)
            
    def _generate_standard_klines(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Génère des klines standard (non glissantes) avec pandas resample.
        Les timestamps des klines agrégées sont alignés sur le début de la période
        (conformément à `label='left'`, `closed='left'`). Column names used are
        from src.core.constants.Kline.
        
        Args:
            df: DataFrame source avec un DatetimeIndex.
            freq: Fréquence pandas (ex: '5T', '1H') pour l'agrégation.
            
        Returns:
            DataFrame avec klines resamplées.
        """
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

            resampled = df_copy.resample(freq, label='left', closed='left').agg(current_aggregation_rules)
            resampled = resampled.dropna(subset=[Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE])
            
            if 'pair' in df_copy.columns and 'pair' not in resampled.columns:
                if not df_copy.empty and not resampled.empty : # Check if resampled is not empty before assigning
                     resampled['pair'] = df_copy['pair'].iloc[0] 
            
            is_closed_col = Kline.OHLCV_IS_KLINE_CLOSED
            if not resampled.empty: # Only assign if resampled is not empty
                if is_closed_col not in resampled.columns:
                    resampled[is_closed_col] = True 
                else:
                    resampled[is_closed_col] = True

            close_time_col = Kline.OHLCV_KLINE_CLOSE_TIME
            if not resampled.empty: # Only assign if resampled is not empty
                try:
                    offset = pd.Timedelta(freq)
                    resampled[close_time_col] = resampled.index + offset - pd.Timedelta(milliseconds=1)
                except ValueError as ve:
                    logger.warning(f"Could not create Timedelta from freq '{freq}' for {close_time_col}: {ve}. Skipping {close_time_col}.")

            logger.debug(f"Generated {len(resampled)} standard klines at {freq} frequency")
            return resampled
            
        except Exception as e:
            logger.error(f"Error generating standard klines for frequency {freq}: {e}")
            raise DataError(f"Failed to generate standard klines: {e}", original_exception=getattr(e, 'original_exception', e))

    def _generate_rolling_klines(self, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """
        Génère des klines glissantes (rolling windows) en utilisant DataFrame.rolling().
        Le timestamp de la kline glissante correspond à la fin de la fenêtre.
        Les `window_size - 1` premières lignes auront des NaNs pour les valeurs calculées.
        Les noms de colonnes utilisés sont ceux de src.core.constants.Kline.

        Args:
            df: DataFrame source avec des klines (ex: 1m) et un DatetimeIndex.
                Les colonnes doivent correspondre aux noms définis dans src.core.constants.Kline.
            window_size: Taille de la fenêtre glissante en nombre de périodes de base du df.

        Returns:
            DataFrame avec les klines glissantes.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise DataError(f"window_size must be a positive integer. Got: {window_size}")

        df_input_copy = df.copy()

        # --- Explicitly convert relevant columns to numeric to avoid DataError ---
        numeric_cols_to_ensure = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE,
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]
        for col in numeric_cols_to_ensure:
            if col in df_input_copy.columns:
                if not pd.api.types.is_numeric_dtype(df_input_copy[col]):
                    logger.debug(f"Column '{col}' in _generate_rolling_klines input is not numeric (dtype: {df_input_copy[col].dtype}). Attempting conversion.")
                    df_input_copy[col] = pd.to_numeric(df_input_copy[col], errors='coerce')
                    if not pd.api.types.is_numeric_dtype(df_input_copy[col]):
                         logger.error(f"Critical: Failed to convert column '{col}' to numeric for rolling calculation. Dtype remains {df_input_copy[col].dtype}.")
                         # Depending on strictness, could raise DataError here
                    else:
                         logger.debug(f"Column '{col}' successfully converted to numeric (new dtype: {df_input_copy[col].dtype}).")

        # Define expected output columns based on input and aggregation logic
        expected_cols_output = []
        base_rolling_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME]
        for col in base_rolling_cols:
            if col in df_input_copy.columns: expected_cols_output.append(col)
        
        optional_sum_cols = [
            Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]
        for col in optional_sum_cols:
            if col in df_input_copy.columns and self.aggregation_rules.get(col) == 'sum':
                expected_cols_output.append(col)
        
        expected_cols_output.append(Kline.OHLCV_KLINE_CLOSE_TIME)
        expected_cols_output.append(Kline.OHLCV_IS_KLINE_CLOSED)
        if 'pair' in df_input_copy.columns: expected_cols_output.append('pair')
        expected_cols_output = list(dict.fromkeys(expected_cols_output)) # Deduplicate

        if df_input_copy.empty or len(df_input_copy) < window_size:
            logger.warning(f"Input DataFrame is empty or shorter ({len(df_input_copy)}) than window_size ({window_size}) for rolling klines.")
            return pd.DataFrame(columns=expected_cols_output, index=pd.to_datetime([]).tz_localize(df_input_copy.index.tz if df_input_copy.index.tz else None))


        logger.debug(f"Generating rolling klines with window_size={window_size}")
        df_result = pd.DataFrame(index=df_input_copy.index)

        # --- Apply rolling calculations ---
        if Kline.OHLCV_OPEN in df_input_copy.columns:
            df_result[Kline.OHLCV_OPEN] = df_input_copy[Kline.OHLCV_OPEN].rolling(window=window_size, min_periods=window_size).apply(lambda x: x[0] if len(x) > 0 else np.nan, raw=True)
        if Kline.OHLCV_HIGH in df_input_copy.columns:
            df_result[Kline.OHLCV_HIGH] = df_input_copy[Kline.OHLCV_HIGH].rolling(window=window_size, min_periods=window_size).max()
        if Kline.OHLCV_LOW in df_input_copy.columns:
            df_result[Kline.OHLCV_LOW] = df_input_copy[Kline.OHLCV_LOW].rolling(window=window_size, min_periods=window_size).min()
        if Kline.OHLCV_CLOSE in df_input_copy.columns:
            df_result[Kline.OHLCV_CLOSE] = df_input_copy[Kline.OHLCV_CLOSE].rolling(window=window_size, min_periods=window_size).apply(lambda x: x[-1] if len(x) > 0 else np.nan, raw=True)
        if Kline.OHLCV_VOLUME in df_input_copy.columns:
            df_result[Kline.OHLCV_VOLUME] = df_input_copy[Kline.OHLCV_VOLUME].rolling(window=window_size, min_periods=window_size).sum()
        
        for col_name in optional_sum_cols:
            if col_name in df_input_copy.columns and self.aggregation_rules.get(col_name) == 'sum':
                df_result[col_name] = df_input_copy[col_name].rolling(window=window_size, min_periods=window_size).sum()
        
        close_time_col = Kline.OHLCV_KLINE_CLOSE_TIME
        if close_time_col in df_input_copy.columns:
            df_result[close_time_col] = df_input_copy[close_time_col].rolling(window=window_size, min_periods=window_size).apply(lambda x: x[-1] if len(x) > 0 else pd.NaT, raw=False) # raw=False for datetime
        else:
            logger.warning(f"'{close_time_col}' not in input DataFrame. Cannot calculate rolling '{close_time_col}'.")
            df_result[close_time_col] = pd.NaT

        if 'pair' in df_input_copy.columns:
            df_result['pair'] = df_input_copy['pair'] # Direct copy, NaNs will align with rolling NaNs if source has them at those spots

        is_closed_col = Kline.OHLCV_IS_KLINE_CLOSED
        df_result[is_closed_col] = True 
        
        # Ensure all expected columns are present, fill with NaNs if somehow missed (should not happen)
        for col in expected_cols_output:
            if col not in df_result.columns:
                 df_result[col] = np.nan # Or pd.NaT for datetime like columns
                 if col == Kline.OHLCV_KLINE_CLOSE_TIME : df_result[col] = pd.NaT

        return df_result[expected_cols_output]

    def _freq_to_minutes(self, freq: str) -> int:
        pandas_freq = self.interval_mapping.get(freq, freq)
        unit_mapping = {'T': 1, 'H': 60, 'D': 1440, 'W': 10080}
        import re
        match = re.fullmatch(r'(\d*)([THDW])S?', pandas_freq, re.IGNORECASE)
        if match:
            number_str, unit_char = match.groups()
            number = int(number_str) if number_str else 1
            unit_char_upper = unit_char.upper()
            if unit_char_upper in unit_mapping:
                return number * unit_mapping[unit_char_upper]
            else:
                raise ValueError(f"Unrecognized unit '{unit_char}' in frequency string '{pandas_freq}'")
        elif pandas_freq.upper() in ['M', 'MS']:
             raise ValueError(f"Frequency '{pandas_freq}' (month) has variable length and cannot be directly converted to fixed minutes.")
        else:
            raise ValueError(f"Could not parse frequency string '{pandas_freq}' to minutes. Expected format like '5T', '1H'.")
            
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
                    errors.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows")
            
        volume_cols_to_check = [
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES
        ]
        for vol_col in volume_cols_to_check:
            if vol_col in df.columns and pd.api.types.is_numeric_dtype(df[vol_col]): # Check if numeric before < 0
                if (df[vol_col].dropna() < 0).any(): # Dropna before comparison to avoid TypeError with pd.NA
                    errors.append(f"Negative values found in column '{vol_col}'")
            
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing")
        if df.index.has_duplicates:
            errors.append(f"Found {df.index.duplicated().sum()} duplicate timestamps in index")

        if check_continuity and len(df) > 1 and not df.index.has_duplicates and df.index.is_monotonic_increasing:
            time_diffs = df.index.to_series().diff().dropna()
            if expected_freq_minutes is not None:
                expected_delta = pd.Timedelta(minutes=expected_freq_minutes)
                tolerance = pd.Timedelta(seconds=1) 
                incorrect_intervals = time_diffs[
                    (time_diffs > expected_delta + tolerance) | (time_diffs < expected_delta - tolerance)
                ]
                if not incorrect_intervals.empty:
                    examples = incorrect_intervals.head(3).to_dict()
                    errors.append(
                        f"Found {len(incorrect_intervals)} time intervals not matching expected frequency "
                        f"of {expected_freq_minutes} min. Examples: {examples}"
                    )
            else:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq:
                    expected_delta_inferred = pd.Timedelta(inferred_freq)
                    tolerance = pd.Timedelta(seconds=1)
                    anomalous_gaps = time_diffs[time_diffs > expected_delta_inferred + tolerance]
                    if not anomalous_gaps.empty:
                         errors.append(f"Found {len(anomalous_gaps)} potential time gaps larger than inferred freq {inferred_freq}.")
                else: 
                    if (time_diffs > pd.Timedelta(hours=1)).any(): 
                        errors.append("Found time gaps larger than 1 hour.")
        is_valid = len(errors) == 0
        if not is_valid: logger.warning(f"Kline validation failed: {errors}")
        return is_valid, errors
        
    def fill_missing_klines(
        self, df: pd.DataFrame, freq: str = '1T', 
        method: str = 'forward', limit_fill: Optional[int] = None
    ) -> pd.DataFrame:
        if df.empty: return df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex for fill_missing_klines.")
        df_sorted = df.sort_index()
        if df_sorted.empty: return df_sorted
        full_index = pd.date_range(start=df_sorted.index.min(), end=df_sorted.index.max(), freq=freq)
        df_reindexed = df_sorted.reindex(full_index)
        
        price_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]
        volume_cols = [
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]

        if method == 'forward':
            for col in price_cols:
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].fillna(method='ffill', limit=limit_fill)
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
                df_reindexed['close_price_ffill'] = df_reindexed[close_col].fillna(method='ffill', limit=limit_fill)
                for col in [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW]:
                    if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].fillna(df_reindexed['close_price_ffill'])
                df_reindexed[close_col] = df_reindexed['close_price_ffill']
                df_reindexed.drop(columns=['close_price_ffill'], inplace=True, errors='ignore')
            for col in volume_cols:
                if col in df_reindexed.columns: df_reindexed[col] = df_reindexed[col].fillna(0)
        else:
            logger.warning(f"Unknown fill_missing_klines method: {method}.")

        other_cols = [col for col in df_reindexed.columns if col not in price_cols + volume_cols]
        for col in other_cols: 
            if df_reindexed[col].isnull().any():
                 df_reindexed[col] = df_reindexed[col].fillna(method='ffill', limit=limit_fill)
                 df_reindexed[col] = df_reindexed[col].fillna(method='bfill', limit=limit_fill)
        return df_reindexed
        
    def merge_klines(self, df1: pd.DataFrame, df2: pd.DataFrame, prefer: str = 'df2') -> pd.DataFrame:
        if df1.empty and df2.empty: return pd.DataFrame()
        if df1.empty: return df2.sort_index()
        if df2.empty: return df1.sort_index()
        if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
            raise DataError("Both DataFrames must have a DatetimeIndex for merging.")
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
        if df.empty: return {"message": "DataFrame is empty"}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame must have a DatetimeIndex for statistics.")
        stats: Dict[str, any] = {'count': len(df), 'start_date': None, 'end_date': None, 'duration_days': None}
        if len(df) > 0:
            stats['start_date'] = df.index.min()
            stats['end_date'] = df.index.max()
            if stats['end_date'] and stats['start_date']:
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
                if func_name == 'mean': stats[stat_name] = df[col_name].mean()
                elif func_name == 'min': stats[stat_name] = df[col_name].min()
                elif func_name == 'max': stats[stat_name] = df[col_name].max()
                elif func_name == 'sum': stats[stat_name] = df[col_name].sum()
                elif func_name == 'std': stats[stat_name] = df[col_name].std()
            else: stats[stat_name] = np.nan

        close_col = Kline.OHLCV_CLOSE
        if close_col in df.columns and pd.api.types.is_numeric_dtype(df[close_col]) and \
           len(df) > 1 and not df[close_col].isnull().all():
            valid_close_prices = df[close_col].dropna()
            if len(valid_close_prices) > 1 and valid_close_prices.iloc[0] != 0:
                returns = valid_close_prices.pct_change().dropna()
                if not returns.empty:
                    stats['avg_return'] = returns.mean()
                    stats['return_volatility'] = returns.std()
                    stats['total_return_pct'] = (valid_close_prices.iloc[-1] / valid_close_prices.iloc[0] - 1) * 100
                else: stats.update({'avg_return': np.nan, 'return_volatility': np.nan, 'total_return_pct': np.nan})
            else: stats.update({'avg_return': np.nan, 'return_volatility': np.nan, 'total_return_pct': np.nan})
        else: stats.update({'avg_return': None, 'return_volatility': None, 'total_return_pct': None})
        return stats

def example_usage():
    import sys
    logger.remove() 
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO")

    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    periods = 100 
    # Using Kline.OHLCV_TIMESTAMP for the index name directly if it's the standard
    index_name = Kline.OHLCV_TIMESTAMP 
    dates = pd.date_range(start=start_time, periods=periods, freq='1T', name=index_name)
    
    base_open = 40000.0 # Use float for prices
    data = {
        Kline.OHLCV_OPEN: base_open + np.random.normal(0, 50, periods).cumsum(),
        Kline.OHLCV_VOLUME: np.random.uniform(0.1, 2.0, periods),
        Kline.OHLCV_NUMBER_OF_TRADES: np.random.randint(10, 200, periods).astype(float), # Keep numeric
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
    df[Kline.OHLCV_QUOTE_ASSET_VOLUME] = df[Kline.OHLCV_VOLUME] * df[Kline.OHLCV_OPEN]
    df[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] = df[Kline.OHLCV_VOLUME] * np.random.uniform(0.4, 0.6, periods)
    df[Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME] = df[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] * df[Kline.OHLCV_OPEN]
    df[Kline.OHLCV_IS_KLINE_CLOSED] = True

    # Ensure all relevant columns are numeric after creation
    for col in [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, 
                Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, 
                Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME,
                Kline.OHLCV_NUMBER_OF_TRADES]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')


    logger.info(f"Sample 1m klines data created. Shape: {df.shape}")
    # print(df.head().to_string())
    # print(df.info())


    processor = KlineProcessor()
    
    try:
        logger.info("Test: Standard 5m klines...")
        klines_5m = processor.resample_klines(df.copy(), '5m', rolling=False)
        print(f"\nGenerated {len(klines_5m)} standard 5m klines:")
        # print(klines_5m.head().to_string())
    except DataError as e: print(f"Error (5m standard): {e}")

    try:
        logger.info("Test: Rolling 10-period klines (based on 1m)...")
        klines_10p_rolling = processor.resample_klines(df.copy(), target_freq='10m', rolling=True, window_size=10)
        print(f"\nGenerated {len(klines_10p_rolling)} rolling 10-period klines:")
        # print(klines_10p_rolling.head(15).to_string())
        if not klines_10p_rolling.empty:
             is_valid, errors = processor.validate_klines(klines_10p_rolling, check_continuity=False)
             # print(f"Validation rolling 10p: {'Valid' if is_valid else 'Invalid'}. Errors: {errors if errors else 'None'}")
    except DataError as e: print(f"Error (10p rolling): {e}")

if __name__ == "__main__":
    example_usage()
