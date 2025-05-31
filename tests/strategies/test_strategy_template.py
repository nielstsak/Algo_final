# tests/strategies/test_strategy_template.py
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

# Importations depuis le projet
from src.strategies.implementations.strategy_template import StrategyTemplate
from src.core.exceptions import InvalidStrategyParamsError

# Chemin vers le template de stratégie à modifier si différent
# Assurez-vous que le template est bien dans src/strategies/implementations/
# et que les imports sont corrects par rapport à la racine de votre projet.

@pytest.fixture
def default_params() -> Dict[str, Any]:
    """Paramètres par défaut pour StrategyTemplate."""
    return StrategyTemplate.default_params.copy()

@pytest.fixture
def sample_klines_data_1m(default_params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Crée un exemple de DataFrame de klines 1m avec des colonnes OHLCV
    et potentiellement les colonnes d'indicateurs attendues par le template
    (si elles étaient définies statiquement, sinon elles seront ajoutées comme NaN).
    """
    dates = pd.to_datetime([
        "2023-01-01 00:00:00", "2023-01-01 00:01:00", "2023-01-01 00:02:00",
        "2023-01-01 00:03:00", "2023-01-01 00:04:00", "2023-01-01 00:05:00",
        "2023-01-01 00:06:00", "2023-01-01 00:07:00", "2023-01-01 00:08:00",
        "2023-01-01 00:09:00"
    ], utc=True)
    data = {
        "open": np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float),
        "high": np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114], dtype=float),
        "low": np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104], dtype=float),
        "close": np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float),
        "volume": np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=float),
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="timestamp"))

    # Si le template attendait des colonnes d'indicateurs spécifiques pré-calculées,
    # on pourrait les ajouter ici. Par exemple :
    # atr_freq = default_params.get('indicateur_frequence')
    # atr_p = default_params.get('atr_period_sl_tp')
    # atr_col_name = f"ATR_{atr_freq}_p{atr_p}"
    # df[atr_col_name] = np.random.rand(len(df)) * 2 + 0.5 # Exemple de valeurs ATR

    return {"1m": df}


class TestStrategyTemplate:
    """Tests pour la classe StrategyTemplate."""

    def test_initialization_default_params(self, default_params: Dict[str, Any]):
        """Teste l'initialisation avec les paramètres par défaut."""
        strategy = StrategyTemplate(params=default_params.copy())
        assert strategy.name == "StrategyTemplate"
        assert strategy.params["param_exemple_1"] == default_params["param_exemple_1"]
        assert strategy.strategy_name_log_prefix == f"[{StrategyTemplate.name}][DEFAULT_PAIR]"

    def test_initialization_custom_params(self, default_params: Dict[str, Any]):
        """Teste l'initialisation avec des paramètres personnalisés."""
        custom_p = default_params.copy()
        custom_p["param_exemple_1"] = 20
        custom_p["indicateur_frequence"] = "5m"
        
        strategy = StrategyTemplate(params=custom_p, pair_symbol="BTCUSDC")
        assert strategy.params["param_exemple_1"] == 20
        assert strategy.get_param("indicateur_frequence") == "5m"
        assert strategy.strategy_name_log_prefix == f"[{StrategyTemplate.name}][BTCUSDC]"

    def test_validate_params_valid(self, default_params: Dict[str, Any]):
        """Teste une validation de paramètres réussie."""
        try:
            StrategyTemplate(params=default_params.copy())
        except InvalidStrategyParamsError:
            pytest.fail("validate_params a levé une exception pour des paramètres valides.")

    def test_validate_params_invalid_param1(self, default_params: Dict[str, Any]):
        """Teste une validation de paramètres échouée (param_exemple_1 invalide)."""
        invalid_p = default_params.copy()
        invalid_p["param_exemple_1"] = 0 # Doit être > 0
        with pytest.raises(InvalidStrategyParamsError, match="Doit être un entier positif"):
            StrategyTemplate(params=invalid_p)

    def test_validate_params_missing_required(self, default_params: Dict[str, Any]):
        """Teste une validation avec un paramètre requis manquant (si on en ajoute)."""
        # Note: default_params de StrategyTemplate les a tous.
        # Pour un test plus poussé, il faudrait un param requis non dans default_params.
        # Pour l'instant, cette validation est basique dans le template.
        pass

    def test_calculate_indicators_structure(self, default_params: Dict[str, Any], sample_klines_data_1m: Dict[str, pd.DataFrame]):
        """
        Teste que calculate_indicators retourne la structure attendue (dict de DataFrames)
        et que les colonnes OHLCV de base et les colonnes d'indicateurs attendues
        (même si NaN) sont présentes.
        """
        strategy = StrategyTemplate(params=default_params.copy())
        main_tf = strategy.required_timeframes[0]
        
        # Copier pour éviter les modifications par effet de bord
        klines_copy = {main_tf: sample_klines_data_1m[main_tf].copy()}
        
        indicators_result = strategy.calculate_indicators(klines_copy)

        assert isinstance(indicators_result, dict)
        assert main_tf in indicators_result # Ou strategy.get_param("indicateur_frequence") si indicateurs sur autre freq
        result_df = indicators_result[main_tf]
        assert isinstance(result_df, pd.DataFrame)

        # Vérifier que les colonnes OHLCV de base sont toujours là
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result_df.columns

        # Vérifier que les colonnes d'indicateurs "attendues" par le template
        # (si elles étaient définies dans __init__ comme self.rsi_col_strat) y sont, même si NaN.
        # Le template actuel ne définit pas de self.xxx_col_strat pour des indicateurs spécifiques
        # car il attend qu'ils soient calculés ou qu'ils soient passés.
        # Si on avait :
        # atr_freq = strategy.get_param('indicateur_frequence')
        # atr_p = strategy.get_param('atr_period_sl_tp')
        # atr_col_expected = f"ATR_{atr_freq}_p{atr_p}"
        # assert atr_col_expected in result_df.columns
        # Dans le template actuel, calculate_indicators ne fait que vérifier OHLCV.
        # On peut tester que le DataFrame retourné est essentiellement le même que l'entrée
        # pour le template de base (car il ne calcule rien activement).
        pd.testing.assert_frame_equal(result_df, sample_klines_data_1m[main_tf], check_dtype=False)


    def test_generate_signals_empty_indicators(self, default_params: Dict[str, Any]):
        """Teste la génération de signaux avec un DataFrame d'indicateurs vide."""
        strategy = StrategyTemplate(params=default_params.copy())
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) # Simuler un DF vide
        # Le template actuel calcule les indicateurs sur le main_tf[0] des klines
        main_tf = strategy.required_timeframes[0]
        indicators_data = {main_tf: empty_df}
        
        signals_df = strategy.generate_signals(indicators_data)

        assert isinstance(signals_df, pd.DataFrame)
        expected_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']
        for col in expected_cols:
            assert col in signals_df.columns
        assert signals_df.empty # Devrait être vide car indicateurs vides

    def test_generate_signals_structure_with_data(self, default_params: Dict[str, Any], sample_klines_data_1m: Dict[str, pd.DataFrame]):
        """
        Teste que generate_signals retourne un DataFrame avec la structure correcte
        quand il y a des données. Le template actuel retourne des signaux neutres.
        """
        strategy = StrategyTemplate(params=default_params.copy())
        main_tf = strategy.required_timeframes[0]

        # Simuler que calculate_indicators a été appelé
        # Dans le template, calculate_indicators retourne essentiellement le klines_df enrichi
        # ou vérifié.
        indicators_data = strategy.calculate_indicators(sample_klines_data_1m)
        
        signals_df = strategy.generate_signals(indicators_data)

        assert isinstance(signals_df, pd.DataFrame)
        expected_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']
        for col in expected_cols:
            assert col in signals_df.columns
            if col in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
                assert signals_df[col].dtype == bool
                assert not signals_df[col].any() # Le template génère des signaux neutres
            else: # sl, tp
                assert signals_df[col].isnull().all() # Le template ne calcule pas sl/tp

        assert len(signals_df) == len(sample_klines_data_1m[main_tf])


    def test_generate_order_request_no_position_no_signal(
        self, default_params: Dict[str, Any], sample_klines_data_1m: Dict[str, pd.DataFrame]
    ):
        """
        Teste generate_order_request quand il n'y a pas de position et pas de signal d'entrée.
        Le template actuel ne génère pas de signaux d'entrée, donc il devrait retourner None.
        """
        strategy = StrategyTemplate(params=default_params.copy())
        main_tf = strategy.required_timeframes[0]
        
        # Simuler un état d'indicateurs (le template ne les utilise pas activement pour les ordres)
        # Dans un vrai test, on s'assurerait que les indicateurs sont calculés.
        # Pour le template, calculate_indicators retourne le df tel quel après vérifications.
        indicators_for_live = strategy.calculate_indicators(sample_klines_data_1m)

        # Informations pour l'ordre
        symbol = "BTCUSDC"
        current_position = 0 # Pas de position
        available_capital = 10000.0
        symbol_info = { # Infos minimales pour les filtres de précision
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001"}
            ]
        }
        
        order_request = strategy.generate_order_request(
            data_dict=indicators_for_live, # C'est le dict d'indicateurs
            symbol=symbol,
            current_position=current_position,
            available_capital=available_capital,
            symbol_info=symbol_info
        )
        assert order_request is None, "Le template ne devrait pas générer d'ordre sans logique de signal"

    def test_generate_order_request_with_position(
        self, default_params: Dict[str, Any], sample_klines_data_1m: Dict[str, pd.DataFrame]
    ):
        """
        Teste generate_order_request quand une position est déjà ouverte.
        Le template actuel ne gère pas les sorties explicites, donc il devrait retourner None.
        """
        strategy = StrategyTemplate(params=default_params.copy())
        main_tf = strategy.required_timeframes[0]
        indicators_for_live = strategy.calculate_indicators(sample_klines_data_1m)
        
        symbol = "BTCUSDC"
        current_position = 1 # Position longue
        available_capital = 10000.0
        symbol_info = {
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001"}
            ]
        }
        
        order_request = strategy.generate_order_request(
            data_dict=indicators_for_live,
            symbol=symbol,
            current_position=current_position,
            available_capital=available_capital,
            symbol_info=symbol_info
        )
        assert order_request is None, "Le template ne gère pas les sorties d'ordres pour l'instant"

    # Plus de tests seraient nécessaires pour :
    # - Différents scénarios de signaux dans generate_signals (si le template était plus complexe)
    # - Différents scénarios d'ordres dans generate_order_request (si le template était plus complexe)
    # - La logique de _calculate_quantity et _build_entry_params_formatted (en les mockant
    #   ou en fournissant une BaseStrategy de test avec des implémentations simples).
    # - Test avec des indicateurs manquants ou entièrement NaN.