# tests/utils/test_exchange_utils.py
import pytest
from src.utils.exchange_utils import normalize_pair_symbol, normalize_pair_list

@pytest.mark.parametrize(
    "input_symbol, expected_output",
    [
        ("BTC/USDT", "BTCUSDT"),
        ("eth/usdc", "ETHUSDC"),
        ("ADAUSDT", "ADAUSDT"),
        ("  Sol/USDT  ", "SOLUSDT"),
        ("", ""),
        ("BTC", "BTC"), # Cas limite, pourrait être invalide ailleurs mais la normalisation le traite
    ]
)
def test_normalize_pair_symbol_valid_strings(input_symbol, expected_output):
    """Teste la normalisation des symboles de paires valides."""
    assert normalize_pair_symbol(input_symbol) == expected_output

def test_normalize_pair_symbol_non_string(caplog):
    """Teste la normalisation avec une entrée non-string."""
    assert normalize_pair_symbol(123) == 123
    assert "Attempted to normalize non-string pair symbol: 123" in caplog.text

@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        (["BTC/USDT", "eth/usdc", "ADAUSDT"], ["BTCUSDT", "ETHUSDC", "ADAUSDT"]),
        (["  Sol/USDT  ", "XRPUSDT"], ["SOLUSDT", "XRPUSDT"]),
        ([], []),
        ([""], [""]), # Liste avec une chaîne vide
        (["BTC/USDT", "BTC/USDT"], ["BTCUSDT", "BTCUSDT"]), # Doublons
    ]
)
def test_normalize_pair_list_valid(input_list, expected_output):
    """Teste la normalisation d'une liste de paires valides."""
    assert normalize_pair_list(input_list) == expected_output

def test_normalize_pair_list_with_non_string_item(caplog):
    """Teste la normalisation d'une liste contenant un non-string."""
    input_list = ["BTC/USDT", 123, "ETH/USDC"]
    expected_output = ["BTCUSDT", 123, "ETHUSDC"] # La fonction retourne l'élément non-string tel quel
    assert normalize_pair_list(input_list) == expected_output
    assert "Attempted to normalize non-string pair symbol: 123" in caplog.text

def test_normalize_pair_list_non_list_input(caplog):
    """Teste la normalisation avec une entrée qui n'est pas une liste."""
    assert normalize_pair_list("not_a_list") == []
    assert "Attempted to normalize non-list: not_a_list for pair list." in caplog.text

    assert normalize_pair_list(None) == []
    assert "Attempted to normalize non-list: None for pair list." in caplog.text

    assert normalize_pair_list(123) == []
    assert "Attempted to normalize non-list: 123 for pair list." in caplog.text