import pytest
import pandas as pd
import numpy as np

# Importer la stratégie à tester
from src.strategies.implementations.bbands_volume_rsi_strategy_impl import BbandsVolumeRsiStrategy

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
        'open':  [100, 101, 102, 103, 104, 105],
        'high':  [101, 102, 103, 104, 110, 106], # High à 110 pour la bougie du signal
        'low':   [99,  100, 101, 102, 103, 104],
        'close': [101, 102, 103, 104, 108, 105], # Close à 108 pour la bougie du signal
        'volume':[100, 110, 120, 130, 200, 150], # Volume à 200 pour la bougie du signal
    }
    test_df = pd.DataFrame(data)

    # Instanciation de la stratégie avec les paramètres de test
    strategy = BbandsVolumeRsiStrategy(bbands_breakout_strategy_params)
    
    # Remplacer les indicateurs calculés par des valeurs contrôlées pour le test
    # On force les conditions à être vraies uniquement à l'index 4
    strategy.indicators_df = test_df.copy()
    strategy.indicators_df['bb_upper'] =    [102, 103, 104, 105, 106, 107] # close (108) > bb_upper (106)
    strategy.indicators_df['bb_lower'] =    [98,  99,  100, 101, 102, 103]
    strategy.indicators_df['bb_middle'] =   [100, 101, 102, 103, 104, 105]
    strategy.indicators_df['volume_ma'] =   [105, 115, 125, 135, 140, 155] # volume (200) > volume_ma (140)
    strategy.indicators_df['rsi'] =         [50,  55,  58,  59,  65,  61] # rsi (65) > threshold (60)
    strategy.indicators_df['atr'] =         [1.0, 1.2, 1.1, 1.3, 2.0, 1.8] # atr = 2.0 pour le calcul SL/TP

    # Génération des signaux
    strategy.generate_signals(test_df)
    signals = strategy.get_signals()
    
    # --- Assertions ---
    # 1. Vérifier qu'un seul signal d'entrée long a été généré à l'index 4
    assert signals['entry_long'].sum() == 1
    assert signals.loc[4, 'entry_long'] is True
    
    # 2. Vérifier qu'aucun autre type de signal n'a été généré
    assert signals['entry_short'].sum() == 0
    assert signals['exit_long'].sum() == 0
    assert signals['exit_short'].sum() == 0

    # 3. Vérifier le calcul du SL et TP à l'index 4
    expected_sl = 108 - (2.0 * 1.5)  # close - (atr * sl_mult)
    expected_tp = 108 + (2.0 * 3.0)  # close + (atr * tp_mult)
    assert np.isclose(signals.loc[4, 'sl'], expected_sl)
    assert np.isclose(signals.loc[4, 'tp'], expected_tp)
    
    # 4. Vérifier que SL et TP sont NaN partout ailleurs
    assert signals['sl'].drop(index=4).isnull().all()
    assert signals['tp'].drop(index=4).isnull().all()


def test_bbands_volume_rsi_breakout_short_signal(bbands_breakout_strategy_params):
    """
    Vérifie que la stratégie génère correctement un signal de vente (short)
    lorsque toutes les conditions de cassure baissière sont remplies pour la première fois.
    Valide également le calcul du SL/TP.
    """
    # Création de données de test spécifiques pour déclencher un signal short
    data = {
        'open':  [105, 104, 103, 102, 101, 100],
        'high':  [106, 105, 104, 103, 102, 101],
        'low':   [104, 103, 102, 101, 95,  99], # Low à 95 pour la bougie du signal
        'close': [104, 103, 102, 101, 97,  100], # Close à 97 pour la bougie du signal
        'volume':[100, 110, 120, 130, 200, 150], # Volume à 200 pour la bougie du signal
    }
    test_df = pd.DataFrame(data)

    strategy = BbandsVolumeRsiStrategy(bbands_breakout_strategy_params)
    
    # Remplacer les indicateurs par des valeurs contrôlées
    # Conditions vraies uniquement à l'index 4
    strategy.indicators_df = test_df.copy()
    strategy.indicators_df['bb_upper'] =    [106, 105, 104, 103, 102, 101]
    strategy.indicators_df['bb_lower'] =    [102, 101, 100, 99,  98,  99] # close (97) < bb_lower (98)
    strategy.indicators_df['bb_middle'] =   [104, 103, 102, 101, 100, 100]
    strategy.indicators_df['volume_ma'] =   [105, 115, 125, 135, 140, 155] # volume (200) > volume_ma (140)
    strategy.indicators_df['rsi'] =         [50,  45,  42,  41,  35,  39] # rsi (35) < threshold (40)
    strategy.indicators_df['atr'] =         [1.0, 1.2, 1.1, 1.3, 2.0, 1.8] # atr = 2.0

    # Génération des signaux
    strategy.generate_signals(test_df)
    signals = strategy.get_signals()
    
    # --- Assertions ---
    # 1. Un seul signal d'entrée short à l'index 4
    assert signals['entry_short'].sum() == 1
    assert signals.loc[4, 'entry_short'] is True
    
    # 2. Aucun autre signal
    assert signals['entry_long'].sum() == 0
    assert signals['exit_long'].sum() == 0
    assert signals['exit_short'].sum() == 0

    # 3. Calcul correct du SL/TP pour une position short
    expected_sl = 97 + (2.0 * 1.5)  # close + (atr * sl_mult)
    expected_tp = 97 - (2.0 * 3.0)  # close - (atr * tp_mult)
    assert np.isclose(signals.loc[4, 'sl'], expected_sl)
    assert np.isclose(signals.loc[4, 'tp'], expected_tp)
    
    # 4. SL/TP sont NaN partout ailleurs
    assert signals['sl'].drop(index=4).isnull().all()
    assert signals['tp'].drop(index=4).isnull().all()


def test_bbands_volume_rsi_no_signal(bbands_breakout_strategy_params):
    """
    Vérifie que la stratégie ne génère aucun signal si les conditions
    ne sont jamais toutes remplies simultanément.
    """
    data = {
        'open':  [100, 101, 102, 103, 104, 105],
        'high':  [101, 102, 103, 104, 105, 106],
        'low':   [99,  100, 101, 102, 103, 104],
        'close': [101, 102, 103, 104, 105, 105],
        'volume':[100, 110, 120, 130, 140, 150],
    }
    test_df = pd.DataFrame(data)

    strategy = BbandsVolumeRsiStrategy(bbands_breakout_strategy_params)
    strategy.generate_signals(test_df)
    signals = strategy.get_signals()

    # Assertion : Aucun signal d'entrée ne doit être généré
    assert signals['entry_long'].sum() == 0
    assert signals['entry_short'].sum() == 0
