# tests/test_backtest_strategies.py
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.strategies.implementations.bbands_volume_rsi_strategy_impl import BbandsVolumeRsiStrategy
from src.backtesting.vectorbt_engine import VectorBTEngine
from src.strategies.strategy_loader import StrategyLoader

@pytest.fixture(scope="module")
def backtest_engine() -> VectorBTEngine:
    """Fixture to create a VectorBTEngine instance for tests."""
    return VectorBTEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        freq='1h' # La fréquence du backtest
    )

@pytest.fixture(scope="module")
def bbands_strategy_instance() -> BbandsVolumeRsiStrategy:
    """Fixture to create an instance of the BbandsVolumeRsiStrategy."""
    loader = StrategyLoader()
    params = {
        'indicator_frequency': '1h',
        'rsi_buy_breakout_threshold': 60.0,
        'bbands_period': 20,
        'bbands_std_dev': 2.0,
        'volume_ma_period': 20,
        'rsi_period': 14,
        'atr_period_sl_tp': 14,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.0,
    }
    return loader.create_strategy("BbandsVolumeRsiStrategy", params=params, pair_symbol="TESTBTCUSDC")


def create_test_data_for_bbands_long_signal() -> pd.DataFrame:
    """
    Crée un DataFrame avec des données conçues pour déclencher un signal d'achat
    UNIQUEMENT sur l'avant-dernière ligne, en s'assurant que le trade n'est pas immédiatement clôturé.
    """
    periods = 51
    dates = pd.date_range(start="2023-01-01", periods=periods, freq='h', tz='UTC')

    # Données de base "inertes" qui ne déclencheront aucun signal
    base_price = 100.0
    data = {
        'open': np.full(periods, base_price),
        'high': np.full(periods, base_price + 2.0),
        'low': np.full(periods, base_price - 2.0),
        'close': np.full(periods, base_price),
        'volume': np.full(periods, 80.0),
        'BB_UPPER_1h_p20_sd2.0': np.full(periods, base_price + 5.0),
        'BB_LOWER_1h_p20_sd2.0': np.full(periods, base_price - 5.0),
        'RSI_1h_p14': np.full(periods, 50.0),
        'Volume_MA_1h_p20': np.full(periods, 100.0),
        'ATR_1h_p14': np.full(periods, 1.0)
    }
    df = pd.DataFrame(data, index=dates)

    # --- Étape 1: Assurer que la bougie AVANT le signal est neutre ---
    # Les données de base s'en chargent déjà, mais on peut être explicite.
    prev_signal_bar_idx = df.index[-3]
    df.loc[prev_signal_bar_idx, ['close', 'RSI_1h_p14', 'volume']] = [base_price, 50.0, 80.0]


    # --- Étape 2: Créer chirurgicalement la condition de signal à l'avant-dernière bougie ---
    signal_bar_idx = df.index[-2]
    # Forcer les 3 conditions à être vraies
    df.loc[signal_bar_idx, 'close'] = 106.0  # close (106) > BB_UPPER (105)
    df.loc[signal_bar_idx, 'volume'] = 110.0  # volume (110) > Volume_MA (100)
    df.loc[signal_bar_idx, 'RSI_1h_p14'] = 65.0   # RSI (65) > seuil (60)

    # --- Étape 3: Configurer la bougie d'entrée (la dernière) pour que le trade ne soit pas clôturé ---
    entry_bar_idx = df.index[-1]
    
    # Calculer le SL/TP basé sur les données de la bougie de signal
    entry_price_ref = df.loc[signal_bar_idx, 'close']  # 106.0
    atr_val = df.loc[signal_bar_idx, 'ATR_1h_p14']      # 1.0
    sl_atr_mult = 1.5
    tp_atr_mult = 2.0
    stop_loss_price = entry_price_ref - (sl_atr_mult * atr_val)    # 106.0 - 1.5 = 104.5
    take_profit_price = entry_price_ref + (tp_atr_mult * atr_val)  # 106.0 + 2.0 = 108.0

    # L'entrée se fait au 'open' de la bougie suivante.
    # On s'assure que le 'high' et le 'low' de cette bougie sont dans les limites SL/TP.
    df.loc[entry_bar_idx, 'open'] = 106.1  # Prix d'entrée
    df.loc[entry_bar_idx, 'high'] = take_profit_price - 0.1  # 107.9 (inférieur au TP)
    df.loc[entry_bar_idx, 'low'] = stop_loss_price + 0.1    # 104.6 (supérieur au SL)
    df.loc[entry_bar_idx, 'close'] = 107.0 # La clôture peut être n'importe où entre low et high

    return df


def test_signal_generation_and_backtest_execution(
    backtest_engine: VectorBTEngine, 
    bbands_strategy_instance: BbandsVolumeRsiStrategy
):
    """
    Vérifie que la stratégie génère un signal et que le backtest exécute un trade.
    """
    # 1. Préparer les données de test
    test_data_df = create_test_data_for_bbands_long_signal()
    
    # 2. Générer les signaux
    indicators_dict = {'1h': test_data_df}
    signals_df = bbands_strategy_instance.generate_signals(indicators_dict)

    # 3. Vérifier que le signal d'achat est généré sur l'avant-dernière ligne
    assert signals_df is not None
    assert not signals_df.empty
    assert signals_df['entry_long'].iloc[-2] == True, "Un signal d'achat (entry_long) était attendu sur l'avant-dernière bougie."
    assert signals_df['entry_long'].sum() == 1, "Un seul signal d'achat était attendu dans ce jeu de test."
    assert pd.notna(signals_df['sl'].iloc[-2]), "Le stop loss devrait être défini pour le signal d'entrée."
    assert pd.notna(signals_df['tp'].iloc[-2]), "Le take profit devrait être défini pour le signal d'entrée."

    # 4. Exécuter le backtest
    portfolio = backtest_engine.run_backtest(
        data=test_data_df, 
        signals=signals_df,
        symbol="TESTBTCUSDC"
    )

    # 5. Vérifier le résultat du backtest
    assert portfolio is not None, "Le portefeuille retourné par le backtest ne devrait pas être None."
    
    num_trades = portfolio.trades.count()
    assert num_trades > 0, f"Le backtest n'a exécuté aucun trade, mais un signal a été généré. Trades exécutés: {num_trades}"
    assert num_trades == 1, "Exactement un trade aurait dû être exécuté."
    
    print("\nTest de workflow de backtest réussi : signal généré et trade exécuté.")
    print(f"Stats du trade: {portfolio.trades.records_readable}")

