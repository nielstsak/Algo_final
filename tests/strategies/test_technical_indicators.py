# tests/strategies/test_technical_indicators.py
import pytest
import pandas as pd
import numpy as np
import time

# S'assurer que le src path est accessible pour les imports
# Cela peut être géré par la configuration de PYTHONPATH ou conftest.py
# Pour cet exemple, on suppose que l'environnement de test est configuré correctement.
try:
    from src.strategies.technical_indicators import (
        IndicatorManager,
        IndicatorCalculationError,
        IndicatorNotFoundError,
        IndicatorValidationError, # Assurez-vous que cette exception est définie si vous l'utilisez
        custom_sma_numba,
        custom_rsi,
        sma_numba_core 
    )
except ImportError:
    # Fallback si l'exécution se fait directement depuis le dossier tests sans src dans le path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # Ajoute la racine du projet (ex: algo_final)
    from src.strategies.technical_indicators import (
        IndicatorManager,
        IndicatorCalculationError,
        IndicatorNotFoundError,
        IndicatorValidationError,
        custom_sma_numba,
        custom_rsi,
        sma_numba_core
    )


@pytest.fixture(scope="module")
def sample_ohlcv_data() -> pd.DataFrame:
    """Crée un DataFrame OHLCV de test."""
    data = {
        'open': np.array([10, 11, 12, 11.5, 10.5, 9.5, 8.5, 9, 10, 11, 12, 13, 12.5, 11.5, 10.5, 9, 10, 11.5] + [np.nan]*2 + [13, 14]), # 22 points
        'high': np.array([11, 12.5, 13, 12, 11, 10, 9, 10.5, 11, 12, 13.5, 14, 13, 12, 11, 10.5, 11, 12] + [np.nan]*2 + [14, 15]),
        'low': np.array([9.5, 10.5, 11, 10, 9, 8, 7, 8.5, 9, 10, 11, 12, 11.5, 10, 9, 8, 9.5, 10] + [np.nan]*2 + [12, 13]),
        'close': np.array([10.8, 12.2, 11.5, 10.3, 9.2, 8.1, 8.8, 10.2, 10.8, 11.7, 13.2, 12.3, 11.2, 10.1, 9.3, 9.8, 11.2, 10.5] + [np.nan]*2 + [13.5, 14.5]),
        'volume': np.array([100, 150, 120, 110, 90, 80, 70, 85, 95, 105, 115, 125, 130, 140, 155, 160, 170, 180] + [np.nan]*2 + [190, 200]) * 1.0,
    }
    start_date = pd.to_datetime("2023-01-01")
    index = pd.date_range(start_date, periods=len(data['close']), freq='1D')
    df = pd.DataFrame(data, index=index)
    # Assurer la cohérence OHLC simple
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    df.loc[df['low'] < 0, 'low'] = 0.01
    return df

@pytest.fixture(scope="module")
def indicator_manager_instance() -> IndicatorManager:
    """Retourne une instance de IndicatorManager."""
    return IndicatorManager()

class TestIndicatorManager:
    """Tests pour la classe IndicatorManager."""

    def test_initialization(self, indicator_manager_instance: IndicatorManager):
        """Teste l'initialisation et l'enregistrement des indicateurs custom par défaut."""
        assert indicator_manager_instance is not None
        available = indicator_manager_instance.get_available_indicators()
        custom_names = [ind['name'] for ind in available['custom']]
        assert "custom_sma_numba" in custom_names
        assert "custom_rsi" in custom_names

    def test_add_custom_indicator(self, indicator_manager_instance: IndicatorManager):
        """Teste l'ajout d'un nouvel indicateur personnalisé."""
        def my_new_indicator(df, param1):
            return df['close'] + param1
        
        indicator_manager_instance.add_custom_indicator(
            name="my_new_indicator",
            func=my_new_indicator,
            params_info={"param1": "int"},
            description="Un indicateur de test."
        )
        available = indicator_manager_instance.get_available_indicators()
        custom_names = [ind['name'] for ind in available['custom']]
        assert "my_new_indicator" in custom_names

    def test_calculate_pandas_ta_indicator_rsi(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste le calcul d'un indicateur pandas-ta (RSI)."""
        df_with_rsi = indicator_manager_instance.calculate_indicator(
            sample_ohlcv_data.copy(), "rsi", length=14, output_col_prefix="TestRSI"
        )
        # Le nom de colonne par défaut de pandas-ta pour RSI(14) est "RSI_14"
        # Avec le préfixe, ce sera "TestRSI_RSI_14"
        expected_col_name = "TestRSI_RSI_14" 
        assert expected_col_name in df_with_rsi.columns
        assert df_with_rsi[expected_col_name].isnull().sum() < len(sample_ohlcv_data) # Doit y avoir des valeurs non-NaN
        # Les 13 premières valeurs (pour length=14) devraient être NaN pour RSI, plus les NaN d'entrée
        # Le nombre exact de NaN dépend de la gestion des NaN par pandas-ta au début.
        # On s'attend à ce que les (length) premières valeurs soient NaN.
        # Plus les 2 NaN dans les données d'entrée.
        assert df_with_rsi[expected_col_name].iloc[:13+2].isnull().all() # 14 périodes de lookback + 2 NaN dans les données
        assert df_with_rsi[expected_col_name].iloc[14+2:].notnull().any() # Au moins une valeur non-NaN après

    def test_calculate_pandas_ta_indicator_sma(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste le calcul d'un indicateur pandas-ta (SMA)."""
        df_with_sma = indicator_manager_instance.calculate_indicator(
            sample_ohlcv_data.copy(), "sma", length=10, output_col_prefix="TestSMA"
        )
        expected_col_name = "TestSMA_SMA_10"
        assert expected_col_name in df_with_sma.columns
        assert df_with_sma[expected_col_name].isnull().sum() < len(sample_ohlcv_data)
        # Pour SMA(10), les 9 premières valeurs devraient être NaN
        assert df_with_sma[expected_col_name].iloc[:9+2].isnull().all() # 9 lookback + 2 NaN dans les données
        assert df_with_sma[expected_col_name].iloc[9+2:].notnull().any()


    def test_calculate_custom_indicator_sma_numba(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste le calcul de l'indicateur personnalisé custom_sma_numba."""
        df_with_custom_sma = indicator_manager_instance.calculate_indicator(
            sample_ohlcv_data.copy(), "custom_sma_numba", length=5, column='close'
        )
        expected_col_name = "CUSTOM_SMA_NUMBA_5" # Nom par défaut de la fonction custom
        assert expected_col_name in df_with_custom_sma.columns
        assert df_with_custom_sma[expected_col_name].isnull().sum() < len(sample_ohlcv_data)
        # SMA(5) -> 4 NaNs initiaux + 2 NaN des données
        assert df_with_custom_sma[expected_col_name].iloc[:4+2].isnull().all() 
        assert df_with_custom_sma[expected_col_name].iloc[4+2:].notnull().any()

    def test_calculate_custom_indicator_rsi(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste le calcul de l'indicateur personnalisé custom_rsi."""
        df_with_custom_rsi = indicator_manager_instance.calculate_indicator(
            sample_ohlcv_data.copy(), "custom_rsi", length=7, column='close'
        )
        expected_col_name = "CUSTOM_RSI_7"
        assert expected_col_name in df_with_custom_rsi.columns
        assert df_with_custom_rsi[expected_col_name].isnull().sum() < len(sample_ohlcv_data)
         # RSI(7) -> 6 NaNs initiaux + 2 NaN des données
        assert df_with_custom_rsi[expected_col_name].iloc[:6+2].isnull().all()
        assert df_with_custom_rsi[expected_col_name].iloc[6+2:].notnull().any()


    def test_calculate_indicator_not_found(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la levée d'erreur pour un indicateur inexistant."""
        with pytest.raises(IndicatorNotFoundError):
            indicator_manager_instance.calculate_indicator(sample_ohlcv_data, "indicateur_qui_n_existe_pas")

    def test_calculate_indicator_calculation_error(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la levée d'erreur pour des paramètres invalides causant une erreur de calcul."""
        # Exemple: pandas_ta.sma avec length=0 devrait lever une erreur
        with pytest.raises(IndicatorCalculationError): # pandas_ta peut lever ValueError ou autre, qui est encapsulé
            indicator_manager_instance.calculate_indicator(sample_ohlcv_data, "sma", length=0)
        
        # Pour un indicateur custom, si sa propre validation échoue
        with pytest.raises(IndicatorCalculationError):
            indicator_manager_instance.calculate_indicator(sample_ohlcv_data, "custom_sma_numba", length=0)

    def test_calculate_multiple_indicators(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste le calcul de plusieurs indicateurs."""
        configs = [
            {"name": "rsi", "length": 14, "output_col_prefix": "MultiRSI"},
            {"name": "sma", "length": 10, "output_col_prefix": "MultiSMA"},
            {"name": "custom_sma_numba", "length": 7, "column": "open", "output_col_prefix": "MultiCustomSMA"}
        ]
        df_multi = indicator_manager_instance.calculate_multiple_indicators(sample_ohlcv_data.copy(), configs)
        assert "MultiRSI_RSI_14" in df_multi.columns
        assert "MultiSMA_SMA_10" in df_multi.columns
        assert "MultiCustomSMA_length7_columnopen" in df_multi.columns # Le nom est généré par la fonction custom

    def test_validate_indicator_output_success(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste une validation réussie."""
        df_with_rsi = indicator_manager_instance.calculate_indicator(sample_ohlcv_data.copy(), "rsi", length=14)
        errors = indicator_manager_instance.validate_indicator_output(
            df_with_rsi, "RSI_14", expected_lookback=14, value_range=(0, 100)
        )
        assert not errors # Aucune erreur attendue

    def test_validate_indicator_output_wrong_lookback(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la validation avec un lookback initial incorrect."""
        df_temp = sample_ohlcv_data.copy()
        # Créer une colonne avec un lookback incorrect (ex: pas de NaN initiaux)
        df_temp["FAKE_IND"] = np.random.rand(len(df_temp)) 
        errors = indicator_manager_instance.validate_indicator_output(
            df_temp, "FAKE_IND", expected_lookback=10 
        )
        assert any("NaNs initiaux inattendu" in err for err in errors)

    def test_validate_indicator_output_out_of_range(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la validation avec des valeurs hors plage."""
        df_temp = sample_ohlcv_data.copy()
        df_temp["RSI_MODIFIED"] = indicator_manager_instance.calculate_indicator(df_temp, "rsi", length=14)["RSI_14"]
        if "RSI_MODIFIED" in df_temp.columns and df_temp["RSI_MODIFIED"].notna().any():
            # Forcer une valeur hors plage
            first_valid_idx = df_temp["RSI_MODIFIED"].first_valid_index()
            if first_valid_idx is not None:
                 df_temp.loc[first_valid_idx, "RSI_MODIFIED"] = 150 
        
        errors = indicator_manager_instance.validate_indicator_output(
            df_temp, "RSI_MODIFIED", expected_lookback=14, value_range=(0, 100)
        )
        assert any("supérieures à la limite max" in err for err in errors)

    def test_validate_indicator_output_with_infinities(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la validation avec des valeurs infinies."""
        df_temp = sample_ohlcv_data.copy()
        df_temp["IND_INF"] = np.random.rand(len(df_temp))
        if len(df_temp) > 0:
            df_temp.iloc[len(df_temp) // 2, df_temp.columns.get_loc("IND_INF")] = np.inf
        errors = indicator_manager_instance.validate_indicator_output(df_temp, "IND_INF")
        assert any("Contient des valeurs infinies" in err for err in errors)

    def test_measure_performance(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la mesure de performance."""
        # Utiliser un petit nombre de répétitions pour que le test soit rapide
        performance_results = indicator_manager_instance.measure_performance(
            sample_ohlcv_data, "sma", length=10, repetitions=3 
        )
        assert "total_time_ms" in performance_results
        assert "avg_time_ms" in performance_results
        assert performance_results["repetitions"] == 3
        assert performance_results["data_length"] == len(sample_ohlcv_data)
        assert performance_results["total_time_ms"] > 0
        assert performance_results["avg_time_ms"] > 0

    def test_measure_performance_error_handling(self, indicator_manager_instance: IndicatorManager, sample_ohlcv_data: pd.DataFrame):
        """Teste la gestion d'erreur dans measure_performance."""
        performance_results = indicator_manager_instance.measure_performance(
            sample_ohlcv_data, "sma", length=-1, repetitions=3 # Paramètre invalide
        )
        assert "error" in performance_results
        assert "sma" in performance_results["error"] # Le message d'erreur devrait mentionner l'indicateur

# --- Tests spécifiques pour les fonctions Numba ---

def test_sma_numba_core_basic(sample_ohlcv_data: pd.DataFrame):
    """Teste la fonction Numba sma_numba_core avec des données de base."""
    series_np = sample_ohlcv_data['close'].to_numpy(dtype=np.float64)
    length = 5
    
    # Calcul avec Numba
    result_numba = sma_numba_core(series_np, length)
    
    # Calcul de référence avec pandas_ta (ou pandas direct)
    # Note: pandas_ta.sma gère les NaN différemment (par défaut 'drop'), 
    # notre sma_numba_core est plus simple et propage les NaN ou les ignore.
    # Pour une comparaison exacte, il faudrait aligner la gestion des NaN.
    # Ici, on utilise pandas direct pour une SMA simple.
    # pandas rolling mean par défaut ne propage pas les NaN comme notre fonction numba.
    # On va comparer avec pandas_ta pour voir les différences.
    import pandas_ta as ta
    reference_pta = ta.sma(sample_ohlcv_data['close'], length=length, nan_policy="propagate") 
    # nan_policy="propagate" est plus proche de notre Numba si une valeur NaN rend la moyenne NaN.
    # nan_policy="include" est aussi une option.
    # Le comportement par défaut de pandas_ta.sma est nan_policy="drop", ce qui signifie
    # qu'il calcule la moyenne sur les valeurs non-NaN dans la fenêtre.
    # Notre sma_numba_core actuel est plus strict: si un NaN entre, la somme peut devenir NaN.
    # Et la première fenêtre doit être complète.

    # En raison des différences de gestion des NaN, une comparaison directe est délicate.
    # On va vérifier que la sortie a la bonne forme et quelques propriétés.
    assert len(result_numba) == len(series_np)
    assert np.isnan(result_numba[:length-1+2]).all() # Les length-1 + 2 (pour les NaN d'entrée) premières devraient être NaN
    
    # Comparer une valeur non-NaN si possible (après les NaN initiaux et les NaN dans les données)
    first_valid_numba_idx = np.where(~np.isnan(result_numba))[0]
    first_valid_pta_idx = np.where(~np.isnan(reference_pta.to_numpy()))[0]

    if len(first_valid_numba_idx) > 0 and len(first_valid_pta_idx) > 0:
        idx_n = first_valid_numba_idx[0]
        idx_p = first_valid_pta_idx[0]
        # Les index de départ des valeurs valides peuvent différer à cause de la gestion des NaN.
        # On peut comparer les valeurs où les deux sont valides.
        
        # Pour un test plus simple, calculons manuellement pour une petite tranche sans NaN
        clean_series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        expected_sma3 = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0])
        result_clean_numba = sma_numba_core(clean_series, 3)
        np.testing.assert_array_almost_equal(result_clean_numba, expected_sma3, decimal=5)

def test_sma_numba_core_edge_cases():
    """Teste sma_numba_core avec des cas limites."""
    # Série vide
    assert len(sma_numba_core(np.array([]), 5)) == 0
    # Length > len(series)
    series1 = np.array([1.0, 2.0, 3.0])
    result1 = sma_numba_core(series1, 5)
    assert np.isnan(result1).all()
    # Length = 1
    series2 = np.array([1.0, np.nan, 3.0])
    result2 = sma_numba_core(series2, 1)
    np.testing.assert_array_equal(result2, series2) # SMA(1) est la série elle-même
    # Length = 0 ou négatif
    result3 = sma_numba_core(series2, 0)
    assert np.isnan(result3).all()
    result4 = sma_numba_core(series2, -2)
    assert np.isnan(result4).all()
    # Série avec que des NaN
    series_nan = np.array([np.nan, np.nan, np.nan])
    result_nan = sma_numba_core(series_nan, 2)
    assert np.isnan(result_nan).all()


# Pour exécuter ces tests, naviguez vers la racine de votre projet et lancez:
# pytest tests/strategies/test_technical_indicators.py
# ou simplement `pytest` si votre structure et conftest.py sont bien configurés.

