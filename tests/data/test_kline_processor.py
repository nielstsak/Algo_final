import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from src.data.kline_processor import KlineProcessor
from src.core.constants import Kline # For column names
from src.core.exceptions import DataError

@pytest.fixture
def kline_processor_instance() -> KlineProcessor:
    """Fixture to provide an instance of KlineProcessor."""
    return KlineProcessor()

@pytest.fixture
def sample_1m_klines_df_numeric() -> pd.DataFrame:
    """
    Fixture to create a sample DataFrame of 1-minute klines with explicitly numeric types.
    Uses column names from src.core.constants.Kline.
    """
    base_time = datetime(2023, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    data: Dict[str, Any] = {
        Kline.OHLCV_TIMESTAMP: [base_time + timedelta(minutes=i) for i in range(10)],
        Kline.OHLCV_OPEN:  np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.float64),
        Kline.OHLCV_HIGH:  np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114], dtype=np.float64),
        Kline.OHLCV_LOW:   np.array([99,  100, 101, 102, 103, 104, 105, 106, 107, 108], dtype=np.float64),
        Kline.OHLCV_CLOSE: np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=np.float64),
        Kline.OHLCV_VOLUME: np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float64),
        'pair': ['BTCUSDT'] * 10,
        Kline.OHLCV_KLINE_CLOSE_TIME: [base_time + timedelta(minutes=i, seconds=59, milliseconds=999) for i in range(10)],
        Kline.OHLCV_IS_KLINE_CLOSED: [True] * 10,
        Kline.OHLCV_QUOTE_ASSET_VOLUME: np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900], dtype=np.float64),
        Kline.OHLCV_NUMBER_OF_TRADES: np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.float64),
        Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: np.array([5,5.5,6,6.5,7,7.5,8,8.5,9,9.5], dtype=np.float64),
        Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: np.array([500,555,612,670,728,787.5,848,910,972,1035.5], dtype=np.float64),
    }
    df = pd.DataFrame(data)
    df.set_index(Kline.OHLCV_TIMESTAMP, inplace=True)
    # Ensure boolean column has boolean dtype
    df[Kline.OHLCV_IS_KLINE_CLOSED] = df[Kline.OHLCV_IS_KLINE_CLOSED].astype('boolean')
    return df

class TestKlineProcessorRollingKlines:
    """Tests for the _generate_rolling_klines method of KlineProcessor."""

    def test_generate_rolling_klines_window_5(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df_numeric: pd.DataFrame
    ):
        window = 5
        input_df = sample_1m_klines_df_numeric.copy()
        rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=window)

        assert not rolling_df.empty, "Resulting DataFrame should not be empty"
        assert len(rolling_df) == len(input_df), "Rolling output should have same length as input"

        # Columns that are calculated by rolling and should have leading NaNs
        # kline_close_time, open, close are handled by direct slicing, so their NaN pattern is different.
        rolling_agg_cols = [
            Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_VOLUME,
            Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME
        ]
        
        # For open, close, kline_close_time (direct assignment)
        direct_assign_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE, Kline.OHLCV_KLINE_CLOSE_TIME]

        for col in rolling_agg_cols:
            if col in rolling_df.columns:
                assert rolling_df[col].iloc[:window-1].isnull().all(), f"Initial NaNs missing for rolling agg col {col}"
        
        for col in direct_assign_cols:
             if col in rolling_df.columns:
                assert rolling_df[col].iloc[:window-1].isnull().all(), f"Initial NaNs missing for direct assign col {col}"


        expected_first_valid_timestamp = input_df.index[window-1]
        assert rolling_df.index[window-1] == expected_first_valid_timestamp

        first_valid_kline = rolling_df.iloc[window-1]
        source_window_0_4 = input_df.iloc[0:window]

        assert first_valid_kline[Kline.OHLCV_OPEN] == pytest.approx(source_window_0_4[Kline.OHLCV_OPEN].iloc[0])
        assert first_valid_kline[Kline.OHLCV_HIGH] == pytest.approx(source_window_0_4[Kline.OHLCV_HIGH].max())
        assert first_valid_kline[Kline.OHLCV_LOW] == pytest.approx(source_window_0_4[Kline.OHLCV_LOW].min())
        assert first_valid_kline[Kline.OHLCV_CLOSE] == pytest.approx(source_window_0_4[Kline.OHLCV_CLOSE].iloc[-1])
        assert first_valid_kline[Kline.OHLCV_VOLUME] == pytest.approx(source_window_0_4[Kline.OHLCV_VOLUME].sum())
        if Kline.OHLCV_QUOTE_ASSET_VOLUME in rolling_df.columns:
             assert first_valid_kline[Kline.OHLCV_QUOTE_ASSET_VOLUME] == pytest.approx(source_window_0_4[Kline.OHLCV_QUOTE_ASSET_VOLUME].sum())
        if Kline.OHLCV_NUMBER_OF_TRADES in rolling_df.columns:
             assert first_valid_kline[Kline.OHLCV_NUMBER_OF_TRADES] == pytest.approx(source_window_0_4[Kline.OHLCV_NUMBER_OF_TRADES].sum())
        assert first_valid_kline[Kline.OHLCV_KLINE_CLOSE_TIME] == source_window_0_4[Kline.OHLCV_KLINE_CLOSE_TIME].iloc[-1]

        second_valid_kline = rolling_df.iloc[window]
        source_window_1_5 = input_df.iloc[1:window+1]
        assert second_valid_kline[Kline.OHLCV_OPEN] == pytest.approx(source_window_1_5[Kline.OHLCV_OPEN].iloc[0])
        assert second_valid_kline[Kline.OHLCV_HIGH] == pytest.approx(source_window_1_5[Kline.OHLCV_HIGH].max())
        assert second_valid_kline[Kline.OHLCV_LOW] == pytest.approx(source_window_1_5[Kline.OHLCV_LOW].min())
        assert second_valid_kline[Kline.OHLCV_CLOSE] == pytest.approx(source_window_1_5[Kline.OHLCV_CLOSE].iloc[-1])
        assert second_valid_kline[Kline.OHLCV_VOLUME] == pytest.approx(source_window_1_5[Kline.OHLCV_VOLUME].sum())

        assert 'pair' in rolling_df.columns
        assert rolling_df['pair'].iloc[window-1] == 'BTCUSDT'
        assert Kline.OHLCV_IS_KLINE_CLOSED in rolling_df.columns
        assert rolling_df[Kline.OHLCV_IS_KLINE_CLOSED].iloc[window-1] == True # pd.NA or True pd.NA or True
        # Check all valid rows for IS_KLINE_CLOSED (where open is not NaN)
        assert rolling_df[Kline.OHLCV_IS_KLINE_CLOSED][rolling_df[Kline.OHLCV_OPEN].notna()].all()


    def test_generate_rolling_klines_input_shorter_than_window(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df_numeric: pd.DataFrame
    ):
        window = 15 
        short_df = sample_1m_klines_df_numeric.iloc[:5].copy()
        rolling_df = kline_processor_instance._generate_rolling_klines(short_df, window_size=window)
        assert rolling_df.empty

        expected_cols = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE,
            Kline.OHLCV_VOLUME, Kline.OHLCV_KLINE_CLOSE_TIME, Kline.OHLCV_IS_KLINE_CLOSED
        ]
        if Kline.OHLCV_QUOTE_ASSET_VOLUME in short_df.columns: expected_cols.append(Kline.OHLCV_QUOTE_ASSET_VOLUME)
        if Kline.OHLCV_NUMBER_OF_TRADES in short_df.columns: expected_cols.append(Kline.OHLCV_NUMBER_OF_TRADES)
        if Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME in short_df.columns: expected_cols.append(Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME)
        if Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME in short_df.columns: expected_cols.append(Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME)
        if 'pair' in short_df.columns : expected_cols.append('pair')
        
        expected_cols = list(dict.fromkeys(expected_cols))
        assert all(col in rolling_df.columns for col in expected_cols)
        assert len(rolling_df.columns) == len(expected_cols)


    def test_generate_rolling_klines_with_nans_in_source_data(
        self, kline_processor_instance: KlineProcessor
    ):
        base_time = datetime(2023, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        data_with_nans: Dict[str, Any] = {
            Kline.OHLCV_TIMESTAMP: [base_time + timedelta(minutes=i) for i in range(5)],
            Kline.OHLCV_OPEN:  np.array([100, 101, np.nan, 103, 104], dtype=np.float64),
            Kline.OHLCV_HIGH:  np.array([105, 106, 107, 108, 109], dtype=np.float64),
            Kline.OHLCV_LOW:   np.array([99,  100, 101, 102, 103], dtype=np.float64),
            Kline.OHLCV_CLOSE: np.array([101, np.nan, 103, 104, 105], dtype=np.float64),
            Kline.OHLCV_VOLUME: np.array([10, 11, 12, np.nan, 14], dtype=np.float64),
            Kline.OHLCV_KLINE_CLOSE_TIME: [base_time + timedelta(minutes=i, seconds=59, milliseconds=999) for i in range(5)],
            'pair': ['NAN_TEST'] * 5,
            Kline.OHLCV_IS_KLINE_CLOSED: [True] * 5 #astype boolean
        }
        df_nans = pd.DataFrame(data_with_nans).set_index(Kline.OHLCV_TIMESTAMP)
        df_nans[Kline.OHLCV_IS_KLINE_CLOSED] = df_nans[Kline.OHLCV_IS_KLINE_CLOSED].astype('boolean')

        window = 3
        rolling_df_nans = kline_processor_instance._generate_rolling_klines(df_nans, window_size=window)

        # Check NaNs for first (window-1) rows
        direct_assign_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE, Kline.OHLCV_KLINE_CLOSE_TIME]
        rolling_agg_cols = [Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_VOLUME]

        for col in direct_assign_cols + rolling_agg_cols:
            if col in rolling_df_nans.columns:
                assert rolling_df_nans[col].iloc[:window-1].isnull().all(), f"Initial NaNs for {col}"
        
        # Kline at index 2 (08:02:00)
        # Window: 08:00 (O:100 H:105 L:99 C:101 V:10), 08:01 (O:101 H:106 L:100 C:NaN V:11), 08:02 (O:NaN H:107 L:101 C:103 V:12)
        kline_0802 = rolling_df_nans.iloc[2]
        assert kline_0802[Kline.OHLCV_OPEN] == 100.0  
        assert kline_0802[Kline.OHLCV_HIGH] == 107.0  
        assert kline_0802[Kline.OHLCV_LOW] == 99.0    
        assert kline_0802[Kline.OHLCV_CLOSE] == 103.0 
        assert kline_0802[Kline.OHLCV_VOLUME] == 33.0 # sum(10,11,12)
        assert kline_0802[Kline.OHLCV_KLINE_CLOSE_TIME] == df_nans[Kline.OHLCV_KLINE_CLOSE_TIME].iloc[2]

        # Kline at index 3 (08:03:00)
        # Window: 08:01 (O:101 H:106 L:100 C:NaN V:11), 08:02 (O:NaN H:107 L:101 C:103 V:12), 08:03 (O:103 H:108 L:102 C:104 V:NaN)
        kline_0803 = rolling_df_nans.iloc[3]
        assert kline_0803[Kline.OHLCV_OPEN] == 101.0 
        assert kline_0803[Kline.OHLCV_HIGH] == 108.0 
        assert kline_0803[Kline.OHLCV_LOW] == 100.0  
        assert kline_0803[Kline.OHLCV_CLOSE] == 104.0 
        # Volume sum(11, 12, NaN) with min_periods=3 should be NaN because only 2 non-NaN values.
        assert np.isnan(kline_0803[Kline.OHLCV_VOLUME]) 
        assert kline_0803[Kline.OHLCV_KLINE_CLOSE_TIME] == df_nans[Kline.OHLCV_KLINE_CLOSE_TIME].iloc[3]

    def test_rolling_kline_propagated_and_static_columns(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df_numeric: pd.DataFrame
    ):
        window = 3
        input_df = sample_1m_klines_df_numeric.copy()
        rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=window)

        assert 'pair' in rolling_df.columns
        # Direct copy means NaNs in 'pair' will align with where other calculations produce NaNs initially
        pd.testing.assert_series_equal(rolling_df['pair'], input_df['pair'], check_names=False)
        
        assert Kline.OHLCV_IS_KLINE_CLOSED in rolling_df.columns
        # IS_KLINE_CLOSED should be True for rows where a rolling kline is computed, NaN otherwise
        expected_is_closed = pd.Series(pd.NA, index=input_df.index, dtype='boolean')
        expected_is_closed.iloc[window-1:] = True
        pd.testing.assert_series_equal(rolling_df[Kline.OHLCV_IS_KLINE_CLOSED], expected_is_closed, check_names=False)

    def test_rolling_klines_all_aggregations_present(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df_numeric: pd.DataFrame
    ):
        window = 4
        input_df = sample_1m_klines_df_numeric.copy()
        rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=window)

        # Check all expected aggregated columns are present
        assert Kline.OHLCV_QUOTE_ASSET_VOLUME in rolling_df.columns
        assert Kline.OHLCV_NUMBER_OF_TRADES in rolling_df.columns
        assert Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME in rolling_df.columns
        assert Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME in rolling_df.columns

        first_valid_kline = rolling_df.iloc[window-1]
        source_window = input_df.iloc[0:window]

        assert first_valid_kline[Kline.OHLCV_QUOTE_ASSET_VOLUME] == pytest.approx(source_window[Kline.OHLCV_QUOTE_ASSET_VOLUME].sum())
        assert first_valid_kline[Kline.OHLCV_NUMBER_OF_TRADES] == pytest.approx(source_window[Kline.OHLCV_NUMBER_OF_TRADES].sum())
        assert first_valid_kline[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME] == pytest.approx(source_window[Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME].sum())
        assert first_valid_kline[Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME] == pytest.approx(source_window[Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME].sum())

    def test_empty_input_df_rolling(self, kline_processor_instance: KlineProcessor):
        """Test _generate_rolling_klines with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=[
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME,
            Kline.OHLCV_KLINE_CLOSE_TIME, Kline.OHLCV_IS_KLINE_CLOSED, 'pair'
        ]).set_index(pd.to_datetime([]))
        empty_df[Kline.OHLCV_IS_KLINE_CLOSED] = empty_df[Kline.OHLCV_IS_KLINE_CLOSED].astype('boolean')


        rolling_df = kline_processor_instance._generate_rolling_klines(empty_df, window_size=5)
        assert rolling_df.empty
        assert Kline.OHLCV_OPEN in rolling_df.columns # Check if columns are preserved

    def test_resample_klines_calling_rolling(
        self, kline_processor_instance: KlineProcessor, sample_1m_klines_df_numeric: pd.DataFrame
    ):
        """Test that resample_klines correctly calls _generate_rolling_klines."""
        input_df = sample_1m_klines_df_numeric.copy()
        # target_freq '5m' with base 1m data should result in window_size=5 if not overridden
        rolling_df = kline_processor_instance.resample_klines(input_df, target_freq='5m', rolling=True)
        
        # Basic check: compare with direct call to _generate_rolling_klines with window_size=5
        expected_rolling_df = kline_processor_instance._generate_rolling_klines(input_df, window_size=5)
        
        pd.testing.assert_frame_equal(rolling_df, expected_rolling_df)

        # Test with explicit window_size override
        rolling_df_override = kline_processor_instance.resample_klines(input_df, target_freq='3m', rolling=True, window_size=3)
        expected_rolling_df_override = kline_processor_instance._generate_rolling_klines(input_df, window_size=3)
        pd.testing.assert_frame_equal(rolling_df_override, expected_rolling_df_override)
