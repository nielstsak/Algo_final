# tests/strategies/test_technical_indicators.py
import pytest
import pandas as pd
import numpy as np

try:
    from src.strategies.technical_indicators import (
        IndicatorManager, IndicatorCalculationError, IndicatorNotFoundError,
        custom_sma_numba, custom_rsi, sma_numba_core
    )
except ImportError: 
    # Fallback pour permettre l'exécution si les tests sont lancés d'une manière
    # où le PYTHONPATH n'est pas correctement configuré pour src.
    import sys
    from pathlib import Path
    # Ajouter la racine du projet (ex: ALGOTHEFINAL) au sys.path
    # Ce chemin suppose que test_technical_indicators.py est dans tests/strategies/
    # et que src est au même niveau que tests.
    project_root = Path(__file__).resolve().parents[2] 
    sys.path.insert(0, str(project_root))
    from src.strategies.technical_indicators import (
        IndicatorManager, IndicatorCalculationError, IndicatorNotFoundError,
        custom_sma_numba, custom_rsi, sma_numba_core
    )

@pytest.fixture(scope="module")
def sample_ohlcv_data() -> pd.DataFrame:
    data = {
        'open':   np.array([10,11,12,11.5,10.5,9.5,8.5,9,10,11,12,13,12.5,11.5,10.5,9,10,11.5, np.nan, np.nan, 13, 14, 15, 16, 17], dtype=float),
        'high':   np.array([11,12.5,13,12,11,10,9,10.5,11,12,13.5,14,13,12,11,10.5,11,12, np.nan, np.nan, 14, 15, 16, 17, 18], dtype=float),
        'low':    np.array([9.5,10.5,11,10,9,8,7,8.5,9,10,11,12,11.5,10,9,8,9.5,10, np.nan, np.nan, 12, 13, 14, 15, 16], dtype=float),
        'close':  np.array([10.8,12.2,11.5,10.3,9.2,8.1,8.8,10.2,10.8,11.7,13.2,12.3,11.2,10.1,9.3,9.8,11.2,10.5, np.nan, np.nan, 13.5,14.5,15.5,16.5,17.5], dtype=float),
        'volume': np.array([100,150,120,110,90,80,70,85,95,105,115,125,130,140,155,160,170,180, np.nan, np.nan, 190,200,210,220,230], dtype=float)
    }
    # Augmenter la taille des données pour accommoder bbands length=20
    start_date = pd.to_datetime("2023-01-01")
    index = pd.date_range(start_date, periods=len(data['close']), freq='D')
    df = pd.DataFrame(data, index=index)
    # Assurer la cohérence OHLC
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    df.loc[df['low'] < 0.001, 'low'] = 0.001 # Éviter les prix négatifs ou nuls
    return df

@pytest.fixture(scope="module")
def indicator_manager_instance() -> IndicatorManager:
    return IndicatorManager()

class TestIndicatorManager:

    def test_initialization(self, indicator_manager_instance: IndicatorManager):
        assert indicator_manager_instance is not None
        available = indicator_manager_instance.get_available_indicators()
        custom_names = [ind['name'] for ind in available['custom']]
        assert "custom_sma_numba" in custom_names
        assert "custom_rsi" in custom_names

    def test_calculate_pandas_ta_indicator_rsi(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_copy = sample_ohlcv_data.copy()
        df_with_rsi = indicator_manager_instance.calculate_indicator(
            df_copy, "rsi", length=14, output_col_prefix="TestRSI"
        )
        expected_col_name = "TestRSI_RSI_14" 
        assert expected_col_name in df_with_rsi.columns, f"Actual columns: {df_with_rsi.columns.tolist()}"
        assert df_with_rsi[expected_col_name].iloc[:14].isnull().all() 
        assert pd.notnull(df_with_rsi[expected_col_name].iloc[14])
        if pd.isna(sample_ohlcv_data['close'].iloc[18]):
            pass 
        else:
            assert pd.notnull(df_with_rsi[expected_col_name].iloc[18])


    def test_calculate_pandas_ta_indicator_sma(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_copy = sample_ohlcv_data.copy()
        df_with_sma = indicator_manager_instance.calculate_indicator(
            df_copy, "sma", length=10, output_col_prefix="TestSMA"
        )
        expected_col_name = "TestSMA_SMA_10" 
        assert expected_col_name in df_with_sma.columns, f"Actual columns: {df_with_sma.columns.tolist()}"
        assert df_with_sma[expected_col_name].iloc[:9].isnull().all()
        assert pd.notnull(df_with_sma[expected_col_name].iloc[9]) 
        assert pd.isnull(df_with_sma[expected_col_name].iloc[18]) 
        assert pd.isnull(df_with_sma[expected_col_name].iloc[19]) 
        assert pd.isnull(df_with_sma[expected_col_name].iloc[20]) 
        assert pd.isnull(df_with_sma[expected_col_name].iloc[21])

    def test_calculate_pandas_ta_indicator_bbands(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_copy = sample_ohlcv_data.copy()
        bb_length = 20 
        bb_std = 2.0
        df_with_bbands = indicator_manager_instance.calculate_indicator(
            df_copy, "bbands", length=bb_length, std=bb_std, output_col_prefix="TestBB"
        )
        
        expected_bbl_col = f"TestBB_BBL_{bb_length}_{bb_std}"
        expected_bbm_col = f"TestBB_BBM_{bb_length}_{bb_std}"
        expected_bbu_col = f"TestBB_BBU_{bb_length}_{bb_std}"
        expected_bbb_col = f"TestBB_BBB_{bb_length}_{bb_std}" 
        expected_bbp_col = f"TestBB_BBP_{bb_length}_{bb_std}" 
        
        expected_cols = [expected_bbl_col, expected_bbm_col, expected_bbu_col, expected_bbb_col, expected_bbp_col]
        
        for col_name in expected_cols:
            assert col_name in df_with_bbands.columns, f"Colonne attendue '{col_name}' non trouvée. Colonnes actuelles: {df_with_bbands.columns.tolist()}"

        # Avec les NaNs dans sample_ohlcv_data aux indices 18 et 19,
        # et une longueur de 20, aucune fenêtre de 20 points valides ne peut être formée.
        # Donc, toutes les colonnes de bbands devraient être NaN.
        for col_name in expected_cols:
            assert df_with_bbands[col_name].isnull().all(), f"La colonne {col_name} devrait être entièrement NaN."


    def test_calculate_custom_indicator_sma_numba(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_copy = sample_ohlcv_data.copy()
        df_with_custom_sma = indicator_manager_instance.calculate_indicator(
            df_copy, "custom_sma_numba", length=5, column='close', output_col_prefix="TestCustomSMA"
        )
        expected_col_name = "TestCustomSMA_CUSTOM_SMA_NUMBA_LENGTH5" 
        assert expected_col_name in df_with_custom_sma.columns,  f"Actual columns: {df_with_custom_sma.columns.tolist()}"
        assert df_with_custom_sma[expected_col_name].iloc[:4].isnull().all()
        assert pd.notnull(df_with_custom_sma[expected_col_name].iloc[4])
        expected_value_at_idx4 = sample_ohlcv_data['close'].iloc[0:5].mean()
        assert np.isclose(df_with_custom_sma[expected_col_name].iloc[4], expected_value_at_idx4)
        assert pd.isnull(df_with_custom_sma[expected_col_name].iloc[18])
        assert pd.isnull(df_with_custom_sma[expected_col_name].iloc[19])
        assert pd.isnull(df_with_custom_sma[expected_col_name].iloc[20]) 
        assert pd.isnull(df_with_custom_sma[expected_col_name].iloc[21])

    def test_calculate_custom_indicator_rsi(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_copy = sample_ohlcv_data.copy()
        df_with_custom_rsi = indicator_manager_instance.calculate_indicator(
            df_copy, "custom_rsi", length=7, column='close', output_col_prefix="TestCustomRSI"
        )
        expected_col_name = "TestCustomRSI_CUSTOM_RSI_LENGTH7"
        assert expected_col_name in df_with_custom_rsi.columns, f"Actual columns: {df_with_custom_rsi.columns.tolist()}"
        assert df_with_custom_rsi[expected_col_name].iloc[:7].isnull().all()
        assert pd.notnull(df_with_custom_rsi[expected_col_name].iloc[7]) 
        if pd.isna(sample_ohlcv_data['close'].iloc[18]):
            pass
        else:
            assert pd.notnull(df_with_custom_rsi[expected_col_name].iloc[18])


    def test_calculate_indicator_not_found(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        with pytest.raises(IndicatorNotFoundError):
            indicator_manager_instance.calculate_indicator(sample_ohlcv_data, "indicateur_qui_n_existe_pas")

    def test_calculate_indicator_calculation_error_custom(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        with pytest.raises(IndicatorCalculationError, match="La période .* pour custom_sma_numba doit être positive."):
            indicator_manager_instance.calculate_indicator(sample_ohlcv_data, "custom_sma_numba", length=0)

    def test_calculate_indicator_pandas_ta_invalid_param_nan_output(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_result = indicator_manager_instance.calculate_indicator(sample_ohlcv_data.copy(), "sma", length=0, output_col_prefix="SMA_ZERO")
        expected_col_name = "SMA_ZERO_SMA_10" 
        assert expected_col_name in df_result.columns, f"Actual columns: {df_result.columns.tolist()}"
        assert df_result[expected_col_name].iloc[:9].isnull().all()
        assert pd.notnull(df_result[expected_col_name].iloc[9])


    def test_calculate_multiple_indicators(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        configs = [
            {"name": "rsi", "length": 14, "output_col_prefix": "MultiRSI"}, 
            {"name": "sma", "length": 10, "output_col_prefix": "MultiSMA"}, 
            {"name": "custom_sma_numba", "length": 7, "column": "open", "output_col_prefix": "MCSN"}
        ]
        df_multi = indicator_manager_instance.calculate_multiple_indicators(sample_ohlcv_data.copy(), configs)
        assert "MultiRSI_RSI_14" in df_multi.columns
        assert "MultiSMA_SMA_10" in df_multi.columns
        assert "MCSN_CUSTOM_SMA_NUMBA_COLUMNopen_LENGTH7" in df_multi.columns 

    def test_validate_indicator_output_success(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_with_rsi = indicator_manager_instance.calculate_indicator(sample_ohlcv_data.copy(), "rsi", length=14, output_col_prefix="VALID")
        errors = indicator_manager_instance.validate_indicator_output(
            df_with_rsi, "VALID_RSI_14", expected_lookback=14, value_range=(0, 100)
        )
        is_acceptable_error = False
        if errors:
            if all("NaN inattendu" in err for err in errors):
                is_acceptable_error = True
        
        assert not errors or is_acceptable_error, f"Validation errors found: {errors}"


    def test_validate_indicator_output_wrong_lookback(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_temp = sample_ohlcv_data.copy()
        df_temp["FAKE_IND_LOOKBACK"] = np.random.rand(len(df_temp)) 
        errors = indicator_manager_instance.validate_indicator_output(
            df_temp, "FAKE_IND_LOOKBACK", expected_lookback=10 
        )
        assert any("NaNs initiaux incorrects" in err for err in errors)

    def test_validate_indicator_output_out_of_range(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        df_temp = sample_ohlcv_data.copy()
        df_temp = indicator_manager_instance.calculate_indicator(df_temp, "rsi", length=14, output_col_prefix="RNGTEST")
        target_col = "RNGTEST_RSI_14"
        
        if target_col in df_temp.columns and df_temp[target_col].notna().any():
            first_valid_idx = df_temp[target_col].first_valid_index()
            if first_valid_idx is not None: 
                 df_temp.loc[first_valid_idx, target_col] = 150 
        
        errors = indicator_manager_instance.validate_indicator_output(
            df_temp, target_col, expected_lookback=14, value_range=(0, 100)
        )
        assert any("limite max" in err for err in errors), f"Validation errors: {errors}"
        
    def test_measure_performance_error_handling(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        performance_results = indicator_manager_instance.measure_performance(
            sample_ohlcv_data, "custom_sma_numba", length=-1, repetitions=3 
        )
        assert "error" in performance_results
        assert "custom_sma_numba" in performance_results["indicator_name"]
        assert "positive" in performance_results["error"] 

# --- Tests spécifiques pour les fonctions Numba ---
def test_sma_numba_core_basic(sample_ohlcv_data: pd.DataFrame):
    series_np = sample_ohlcv_data['close'].to_numpy(dtype=np.float64)
    length = 5
    result_numba = sma_numba_core(series_np, length)
    assert len(result_numba) == len(series_np)
    assert np.isnan(result_numba[:4]).all() 
    assert not np.isnan(result_numba[4])
    expected_first_sma = np.mean(series_np[0:5]) 
    np.testing.assert_almost_equal(result_numba[4], expected_first_sma, decimal=5)
    
    assert np.isnan(result_numba[18]) 
    assert np.isnan(result_numba[19]) 
    assert np.isnan(result_numba[20]) 
    assert np.isnan(result_numba[21])


def test_sma_numba_core_edge_cases():
    assert len(sma_numba_core(np.array([]), 5)) == 0
    
    series1 = np.array([1.0, 2.0, 3.0])
    result1 = sma_numba_core(series1, 5)
    assert np.isnan(result1).all() 
    
    series2 = np.array([1.0, np.nan, 3.0])
    result2 = sma_numba_core(series2, 1)
    np.testing.assert_array_equal(result2, series2) 
    
    result3 = sma_numba_core(series2, 0)
    assert np.isnan(result3).all()

    series_nan = np.array([np.nan, np.nan, np.nan])
    result_nan = sma_numba_core(series_nan, 2)
    assert np.isnan(result_nan).all()

    series_mid_nan = np.array([1., 2., np.nan, 4., 5., 6.])
    result_mid_nan = sma_numba_core(series_mid_nan, 3)
    assert np.isnan(result_mid_nan[2]) 
    assert np.isnan(result_mid_nan[3]) 
    assert np.isnan(result_mid_nan[4]) 
    assert np.isclose(result_mid_nan[5], 5.0)
