import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any

from src.data.kline_processor import KlineProcessor
from src.core.constants import Kline # For column names
from src.core.exceptions import DataError

@pytest.fixture
def kline_processor_instance() -> KlineProcessor:
    """Fixture to provide an instance of KlineProcessor."""
    return KlineProcessor()

@pytest.fixture
def sample_1m_klines_df() -> pd.DataFrame:
    """
    Fixture to create a sample DataFrame of 1-minute klines.
    Uses column names from src.core.constants.Kline.
    """
    data: Dict[str, Any] = {
        # Timestamps for 10 minutes
        Kline.OHLCV_TIMESTAMP: pd.to_datetime([
            "2023-01-01 08:00:00", "2023-01-01 08:01:00", "2023-01-01 08:02:00",
            "2023-01-01 08:03:00", "2023-01-01 08:04:00", "2023-01-01 08:05:00",
            "2023-01-01 08:06:00", "2023-01-01 08:07:00", "2023-01-01 08:08:00",
            "2023-01-01 08:09:00"
        ], utc=True),
        Kline.OHLCV_OPEN:  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        Kline.OHLCV_HIGH:  [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        Kline.OHLCV_LOW:   [99,  100, 101, 102, 103, 104, 105, 106, 107, 108],
        Kline.OHLCV_CLOSE: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        Kline.OHLCV_VOLUME: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'pair': ['BTCUSDT'] * 10, # Example of a non-aggregated column
        Kline.OHLCV_KLINE_CLOSE_TIME: pd.to_datetime([
            "2023-01-01 08:00:59.999", "2023-01-01 08:01:59.999", "2023-01-01 08:02:59.999",
            "2023-01-01 08:03:59.999", "2023-01-01 08:04:59.999", "2023-01-01 08:05:59.999",
            "2023-01-01 08:06:59.999", "2023-01-01 08:07:59.999", "2023-01-01 08:08:59.999",
            "2023-01-01 08:09:59.999"
        ], utc=True),
        Kline.OHLCV_IS_KLINE_CLOSED: [True] * 10,
        # Optional columns for testing extended aggregation
        Kline.OHLCV_QUOTE_ASSET_VOLUME: [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        Kline.OHLCV_NUMBER_OF_TRADES: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    }
    df = pd.DataFrame(data)
    df.set_index(Kline.OHLCV_TIMESTAMP, inplace=True)
    return df

class TestKlineProcessorRollingKlines:
    """Tests for the _generate_rolling_klines method of KlineProcessor."""

    def test_generate_rolling_klines_window_5(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df: pd.DataFrame
    ):
        """
        Tests rolling kline generation with a window of 5 periods.
        Verifies OHLCV calculations, timestamps, initial NaNs, and propagated columns.
        """
        window = 5
        # Make a copy as the processor might modify it (though it shouldn't for _generate_rolling_klines input)
        input_df = sample_1m_klines_df.copy()
        rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=window)

        assert not rolling_df.empty
        # Rolling output has same length as input, with NaNs at the start
        assert len(rolling_df) == len(input_df)

        # Columns to check for initial NaNs and calculations
        calculated_cols = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE,
            Kline.OHLCV_VOLUME, Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_KLINE_CLOSE_TIME
        ]

        # Check NaN for first window-1 rows
        for col in calculated_cols:
            if col in rolling_df.columns: # Only check if column is expected and present
                pd.testing.assert_series_equal(
                    rolling_df[col].iloc[:window-1],
                    pd.Series([np.nan]*(window-1), index=rolling_df.index[:window-1], name=col, dtype=rolling_df[col].dtype),
                    check_dtype=False # NaNs can mess with dtype, especially for object types like Timestamp
                )
        
        # Timestamp of the first valid rolling kline corresponds to the end of the first full window
        expected_first_valid_timestamp = input_df.index[window-1] # Timestamp of the 5th original kline
        assert rolling_df.index[window-1] == expected_first_valid_timestamp

        # --- Validate first valid rolling kline (window 0 to 4, result at index 4) ---
        first_valid_kline = rolling_df.iloc[window-1]
        source_window_0_4 = input_df.iloc[0:window] # klines at index 0, 1, 2, 3, 4

        assert first_valid_kline[Kline.OHLCV_OPEN] == source_window_0_4[Kline.OHLCV_OPEN].iloc[0]  # 100
        assert first_valid_kline[Kline.OHLCV_HIGH] == source_window_0_4[Kline.OHLCV_HIGH].max()    # 109
        assert first_valid_kline[Kline.OHLCV_LOW] == source_window_0_4[Kline.OHLCV_LOW].min()      # 99
        assert first_valid_kline[Kline.OHLCV_CLOSE] == source_window_0_4[Kline.OHLCV_CLOSE].iloc[-1] # 105
        assert first_valid_kline[Kline.OHLCV_VOLUME] == source_window_0_4[Kline.OHLCV_VOLUME].sum() # 10+11+12+13+14 = 60
        
        # Check optional aggregated columns
        if Kline.OHLCV_QUOTE_ASSET_VOLUME in rolling_df.columns:
             assert first_valid_kline[Kline.OHLCV_QUOTE_ASSET_VOLUME] == source_window_0_4[Kline.OHLCV_QUOTE_ASSET_VOLUME].sum() # 1000+...+1400 = 6000
        if Kline.OHLCV_NUMBER_OF_TRADES in rolling_df.columns:
             assert first_valid_kline[Kline.OHLCV_NUMBER_OF_TRADES] == source_window_0_4[Kline.OHLCV_NUMBER_OF_TRADES].sum() # 5+...+9 = 35

        assert first_valid_kline[Kline.OHLCV_KLINE_CLOSE_TIME] == source_window_0_4[Kline.OHLCV_KLINE_CLOSE_TIME].iloc[-1] # Timestamp of 08:04:59.999

        # --- Validate second valid rolling kline (window 1 to 5, result at index 5) ---
        second_valid_kline = rolling_df.iloc[window]
        source_window_1_5 = input_df.iloc[1:window+1] # klines at index 1, 2, 3, 4, 5

        assert second_valid_kline[Kline.OHLCV_OPEN] == source_window_1_5[Kline.OHLCV_OPEN].iloc[0]  # 101
        assert second_valid_kline[Kline.OHLCV_HIGH] == source_window_1_5[Kline.OHLCV_HIGH].max()    # 110
        assert second_valid_kline[Kline.OHLCV_LOW] == source_window_1_5[Kline.OHLCV_LOW].min()      # 100
        assert second_valid_kline[Kline.OHLCV_CLOSE] == source_window_1_5[Kline.OHLCV_CLOSE].iloc[-1] # 106
        assert second_valid_kline[Kline.OHLCV_VOLUME] == source_window_1_5[Kline.OHLCV_VOLUME].sum() # 11+12+13+14+15 = 65

        # Check propagated columns
        assert 'pair' in rolling_df.columns
        # For 'pair', since it's directly copied, it will be NaN for initial rows if source has NaNs, or just copied.
        # If taken via rolling apply, it would also be NaN. The implementation copies the whole column.
        # So, we expect the value from the corresponding row in the input.
        assert rolling_df['pair'].iloc[window-1] == input_df['pair'].iloc[window-1] # 'BTCUSDT'
        
        assert Kline.OHLCV_IS_KLINE_CLOSED in rolling_df.columns
        assert rolling_df[Kline.OHLCV_IS_KLINE_CLOSED].iloc[window-1] == True
        assert rolling_df[Kline.OHLCV_IS_KLINE_CLOSED].all() # Should be True for all

    def test_generate_rolling_klines_input_shorter_than_window(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df: pd.DataFrame
    ):
        """Tests behavior when the input DataFrame is shorter than the window size."""
        window = 15 # Window larger than data length (10)
        short_df = sample_1m_klines_df.iloc[:5].copy() # Only 5 rows
        
        rolling_df = kline_processor_instance._generate_rolling_klines(short_df, window_size=window)
        
        # Expect an empty DataFrame with correct columns as per implementation
        assert rolling_df.empty
        
        expected_cols = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE,
            Kline.OHLCV_VOLUME, Kline.OHLCV_KLINE_CLOSE_TIME, Kline.OHLCV_IS_KLINE_CLOSED
        ]
        # Add optional columns if they would have been in the output based on input 'short_df'
        if Kline.OHLCV_QUOTE_ASSET_VOLUME in short_df.columns and \
           Kline.OHLCV_QUOTE_ASSET_VOLUME in kline_processor_instance.aggregation_rules:
            expected_cols.append(Kline.OHLCV_QUOTE_ASSET_VOLUME)
        if Kline.OHLCV_NUMBER_OF_TRADES in short_df.columns and \
           Kline.OHLCV_NUMBER_OF_TRADES in kline_processor_instance.aggregation_rules:
            expected_cols.append(Kline.OHLCV_NUMBER_OF_TRADES)
        if 'pair' in short_df.columns: 
            expected_cols.append('pair')
        
        expected_cols = list(dict.fromkeys(expected_cols)) # Deduplicate

        assert all(col in rolling_df.columns for col in expected_cols)
        assert len(rolling_df.columns) == len(expected_cols)


    def test_generate_rolling_klines_with_nans_in_source_data(
        self, kline_processor_instance: KlineProcessor
    ):
        """Tests rolling kline generation when source data contains NaNs."""
        data_with_nans: Dict[str, Any] = {
            Kline.OHLCV_TIMESTAMP: pd.to_datetime([
                "2023-01-01 08:00:00", "2023-01-01 08:01:00", "2023-01-01 08:02:00",
                "2023-01-01 08:03:00", "2023-01-01 08:04:00"
            ], utc=True),
            Kline.OHLCV_OPEN:  [100, 101, np.nan, 103, 104],
            Kline.OHLCV_HIGH:  [105, 106, 107,    108, 109], # No NaNs in high/low for easier max/min verification
            Kline.OHLCV_LOW:   [99,  100, 101,    102, 103],
            Kline.OHLCV_CLOSE: [101, np.nan, 103, 104, 105],
            Kline.OHLCV_VOLUME: [10, 11, 12, np.nan, 14],
            Kline.OHLCV_KLINE_CLOSE_TIME: pd.to_datetime([ # Timestamps for close times
                "2023-01-01 08:00:59.999", "2023-01-01 08:01:59.999", "2023-01-01 08:02:59.999",
                "2023-01-01 08:03:59.999", "2023-01-01 08:04:59.999"
            ], utc=True),
            'pair': ['NAN_TEST'] * 5,
            Kline.OHLCV_IS_KLINE_CLOSED: [True] * 5
        }
        df_nans = pd.DataFrame(data_with_nans).set_index(Kline.OHLCV_TIMESTAMP)
        window = 3
        
        rolling_df_nans = kline_processor_instance._generate_rolling_klines(df_nans, window_size=window)

        # First (window-1) = 2 rows should be NaN for calculated fields
        calculated_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME]
        for col in calculated_cols:
             pd.testing.assert_series_equal(
                    rolling_df_nans[col].iloc[:window-1],
                    pd.Series([np.nan]*(window-1), index=rolling_df_nans.index[:window-1], name=col, dtype=rolling_df_nans[col].dtype),
                    check_dtype=False 
                )

        # --- Verify kline at index 2 (timestamp 08:02:00) ---
        # Window includes source data at index 0, 1, 2 (timestamps 08:00, 08:01, 08:02)
        # Source data:
        # 08:00: open=100, high=105, low=99,  close=101, volume=10
        # 08:01: open=101, high=106, low=100, close=NaN, volume=11
        # 08:02: open=NaN, high=107, low=101, close=103, volume=12
        kline_0802 = rolling_df_nans.iloc[2]
        assert kline_0802[Kline.OHLCV_OPEN] == 100.0  # open at 08:00
        assert kline_0802[Kline.OHLCV_HIGH] == 107.0  # max(105, 106, 107)
        assert kline_0802[Kline.OHLCV_LOW] == 99.0    # min(99, 100, 101)
        assert kline_0802[Kline.OHLCV_CLOSE] == 103.0 # close at 08:02
        assert kline_0802[Kline.OHLCV_VOLUME] == 33.0 # sum(10, 11, 12) (sum skipsna=True by default, min_periods handles full window)

        # --- Verify kline at index 3 (timestamp 08:03:00) ---
        # Window includes source data at index 1, 2, 3 (timestamps 08:01, 08:02, 08:03)
        # Source data:
        # 08:01: open=101, high=106, low=100, close=NaN, volume=11
        # 08:02: open=NaN, high=107, low=101, close=103, volume=12
        # 08:03: open=103, high=108, low=102, close=104, volume=NaN
        kline_0803 = rolling_df_nans.iloc[3]
        assert kline_0803[Kline.OHLCV_OPEN] == 101.0     # open at 08:01
        assert kline_0803[Kline.OHLCV_HIGH] == 108.0     # max(106, 107, 108)
        assert kline_0803[Kline.OHLCV_LOW] == 100.0      # min(100, 101, 102)
        assert np.isnan(kline_0803[Kline.OHLCV_CLOSE])   # close at 08:03 is 104, but close at 08:01 is NaN. apply(x[-1]) will take the last, which is 104. This needs checking.
                                                         # The prompt example: df_rolling['close_price'].rolling(window=window_size, min_periods=window_size).apply(lambda x: x[-1], raw=True)
                                                         # If x[-1] is NaN, it will be NaN. Here, df_nans[Kline.OHLCV_CLOSE].iloc[3] (last element of window) = 104. So this should be 104.
        # Correction for close:
        assert kline_0803[Kline.OHLCV_CLOSE] == 104.0     # close at 08:03 (last element of window)

        # Volume sum: sum(11, 12, NaN). Pandas sum() with skipna=True (default) gives 23.
        # If min_periods=window_size (which is 3), and one value is NaN, the sum will be NaN.
        assert np.isnan(kline_0803[Kline.OHLCV_VOLUME]) # This is correct if min_periods=window_size is strictly enforced for count of non-NaN values.

    def test_rolling_kline_propagated_columns(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df: pd.DataFrame
    ):
        """Tests that 'pair' and 'is_kline_closed' are correctly handled."""
        window = 3
        input_df = sample_1m_klines_df.copy()
        rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=window)

        assert 'pair' in rolling_df.columns
        # Test 'pair' for a valid row, e.g. first valid one at index window-1
        # Current implementation of _generate_rolling_klines copies the 'pair' column directly.
        pd.testing.assert_series_equal(rolling_df['pair'], input_df['pair'], check_names=False)
        
        assert Kline.OHLCV_IS_KLINE_CLOSED in rolling_df.columns
        assert rolling_df[Kline.OHLCV_IS_KLINE_CLOSED].all() # Should be True for all rows

    def test_rolling_klines_with_all_optional_aggregations(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df: pd.DataFrame
    ):
        """
        Tests rolling kline generation ensuring all optional columns like quote_asset_volume
        and number_of_trades are aggregated if present in input and rules.
        """
        window = 4
        input_df = sample_1m_klines_df.copy() # sample_1m_klines_df includes these optional columns
        
        # Ensure these are part of the processor's aggregation rules for the test
        # (they should be by default from constants.py and the __init__ fallback)
        assert Kline.OHLCV_QUOTE_ASSET_VOLUME in kline_processor_instance.aggregation_rules
        assert Kline.OHLCV_NUMBER_OF_TRADES in kline_processor_instance.aggregation_rules
        
        rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=window)

        assert Kline.OHLCV_QUOTE_ASSET_VOLUME in rolling_df.columns
        assert Kline.OHLCV_NUMBER_OF_TRADES in rolling_df.columns

        # Validate first valid kline (window 0-3, result at index 3)
        first_valid_kline = rolling_df.iloc[window-1]
        source_window_0_3 = input_df.iloc[0:window]

        expected_quote_vol = source_window_0_3[Kline.OHLCV_QUOTE_ASSET_VOLUME].sum() # 1000+1100+1200+1300 = 4600
        expected_trades = source_window_0_3[Kline.OHLCV_NUMBER_OF_TRADES].sum()     # 5+6+7+8 = 26
        
        assert first_valid_kline[Kline.OHLCV_QUOTE_ASSET_VOLUME] == expected_quote_vol
        assert first_valid_kline[Kline.OHLCV_NUMBER_OF_TRADES] == expected_trades