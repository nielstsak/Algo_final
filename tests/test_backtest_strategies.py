import pytest
import pandas as pd
import numpy as np

# Importer la stratégie à tester
from src.strategies.implementations.bbands_volume_rsi_strategy_impl import (
    BbandsVolumeRsiStrategy,
)


@pytest.fixture
def bbands_breakout_strategy_params():
    """
    Fournit un jeu de paramètres par défaut et valides pour la stratégie
    BbandsVolumeRsiStrategy version "breakout".
    """
    return {
        "bbands_period": 5,
        "bbands_std_dev": 2.0,
        "volume_ma_period": 5,
        "rsi_period": 5,
        "rsi_buy_breakout_threshold": 60,
        "rsi_sell_breakout_threshold": 40,
        "atr_period": 5,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
    }


def test_bbands_volume_rsi_breakout_long_signal(bbands_breakout_strategy_params):
    """
    Vérifie que la stratégie génère correctement un signal d'achat (long)
    lorsque toutes les conditions de cassure haussière sont remplies pour la première fois.
    Valide également le calcul du SL/TP.
    """
    # Création de données de test spécifiques pour déclencher un signal long
    data = {
        "open": [100, 101, 102, 103, 104, 105],
        "high": [101, 102, 103, 104, 110, 106],  # High à 110 pour la bougie du signal
        "low": [99, 100, 101, 102, 103, 104],
        "close": [101, 102, 103, 104, 108, 105],  # Close à 108 pour la bougie du signal
        "volume": [100, 110, 120, 130, 200, 150],  # Volume à 200 pour la bougie du signal
    }
    test_df = pd.DataFrame(data)

    # Instanciation de la stratégie avec les paramètres de test
    strategy = BbandsVolumeRsiStrategy(params=bbands_breakout_strategy_params)

    # 1. Calculer les indicateurs réels
    # Pour ce test, nous allons mocker les indicateurs pour garantir des conditions précises.
    indicators_df = test_df.copy()
    indicators_df["BBU"] = [102, 103, 104, 105, 106, 107]  # close (108) > bb_upper (106)
    indicators_df["BBM"] = [100, 101, 102, 103, 104, 105]
    indicators_df["BBL"] = [98, 99, 100, 101, 102, 103]
    indicators_df["volume_ma"] = [
        105,
        115,
        125,
        135,
        140,
        155,
    ]  # volume (200) > volume_ma (140)
    indicators_df["rsi"] = [50, 55, 58, 59, 65, 61]  # rsi (65) > threshold (60)
    indicators_df["atr"] = [
        1.0,
        1.2,
        1.1,
        1.3,
        2.0,
        1.8,
    ]  # atr = 2.0 pour le calcul SL/TP

    # 2. Génération des signaux à partir des indicateurs (réels ou mockés)
    signals = strategy.generate_signals(indicators_df)

    # --- Assertions ---
    # Un seul signal d'achat doit être généré
    assert signals["entry_long"].sum() == 1
    # Le signal doit être à l'index 4
    assert signals.loc[4, "entry_long"] == True

    # Aucun autre signal ne doit être actif
    assert signals["entry_short"].sum() == 0
    assert signals["exit_long"].sum() == 0
    assert signals["exit_short"].sum() == 0

    # Vérification du calcul SL/TP
    expected_sl = 108 - (2.0 * bbands_breakout_strategy_params["sl_atr_mult"])
    expected_tp = 108 + (2.0 * bbands_breakout_strategy_params["tp_atr_mult"])
    assert np.isclose(signals.loc[4, "sl"], expected_sl)
    assert np.isclose(signals.loc[4, "tp"], expected_tp)

    # Les autres lignes ne doivent pas avoir de SL/TP
    assert signals["sl"].drop(index=4).isnull().all()
    assert signals["tp"].drop(index=4).isnull().all()


def test_bbands_volume_rsi_breakout_short_signal(bbands_breakout_strategy_params):
    """
    Vérifie que la stratégie génère correctement un signal de vente (short)
    lorsque toutes les conditions de cassure baissière sont remplies pour la première fois.
    """
    data = {
        "open": [105, 104, 103, 102, 101, 100],
        "high": [106, 105, 104, 103, 102, 101],
        "low": [104, 103, 102, 101, 95, 99],  # Low à 95 pour la bougie du signal
        "close": [104, 103, 102, 101, 97, 100],  # Close à 97 pour la bougie du signal
        "volume": [100, 110, 120, 130, 200, 150],  # Volume à 200 pour la bougie du signal
    }
    test_df = pd.DataFrame(data)

    strategy = BbandsVolumeRsiStrategy(params=bbands_breakout_strategy_params)

    # Mocker les indicateurs pour déclencher le signal short
    indicators_df = test_df.copy()
    indicators_df["BBU"] = [106, 105, 104, 103, 102, 101]
    indicators_df["BBM"] = [104, 103, 102, 101, 100, 100]
    indicators_df["BBL"] = [102, 101, 100, 99, 98, 99]  # close (97) < bb_lower (98)
    indicators_df["volume_ma"] = [
        105,
        115,
        125,
        135,
        140,
        155,
    ]  # volume (200) > volume_ma (140)
    indicators_df["rsi"] = [50, 45, 42, 41, 35, 39]  # rsi (35) < threshold (40)
    indicators_df["atr"] = [1.0, 1.2, 1.1, 1.3, 2.0, 1.8]

    signals = strategy.generate_signals(indicators_df)

    assert signals["entry_short"].sum() == 1
    assert signals.loc[4, "entry_short"] == True

    assert signals["entry_long"].sum() == 0

    expected_sl = 97 + (2.0 * bbands_breakout_strategy_params["sl_atr_mult"])
    expected_tp = 97 - (2.0 * bbands_breakout_strategy_params["tp_atr_mult"])
    assert np.isclose(signals.loc[4, "sl"], expected_sl)
    assert np.isclose(signals.loc[4, "tp"], expected_tp)


def test_bbands_volume_rsi_no_signal(bbands_breakout_strategy_params):
    """
    Vérifie que la stratégie ne génère aucun signal si les conditions
    ne sont jamais toutes remplies simultanément.
    """
    data = {
        "open": [100, 101, 102, 103, 104, 105],
        "high": [101, 102, 103, 104, 105, 106],
        "low": [99, 100, 101, 102, 103, 104],
        "close": [101, 102, 103, 104, 105, 105],
        "volume": [100, 110, 120, 130, 140, 150],
    }
    test_df = pd.DataFrame(data)

    strategy = BbandsVolumeRsiStrategy(params=bbands_breakout_strategy_params)

    # Utiliser le calcul réel des indicateurs, qui ne devrait pas produire de signal ici
    indicators_df = strategy.calculate_indicators(test_df)
    signals = strategy.generate_signals(indicators_df)

    assert signals["entry_long"].sum() == 0
    assert signals["entry_short"].sum() == 0
    assert signals["sl"].isnull().all()
    assert signals["tp"].isnull().all()
