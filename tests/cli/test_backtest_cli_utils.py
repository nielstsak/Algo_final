# tests/cli/test_backtest_cli_utils.py
import pytest
import click
from src.cli.commands.backtest import CommaSeparatedPairs

@pytest.fixture
def comma_separated_pairs_type():
    return CommaSeparatedPairs()

@pytest.fixture
def mock_click_context():
    return click.Context(click.Command('test_cmd'))

@pytest.mark.parametrize("input_value, expected_output", [
    ("BTC/USDT,ETH/USDC,ADA/USDT", ["BTCUSDT", "ETHUSDC", "ADAUSDT"]),
    ("SOLUSDT", ["SOLUSDT"]),
    ("  XRP/USDT , dot/usdc  ", ["XRPUSDT", "DOTUSDC"]),
    ("", []), # Une chaîne vide devrait résulter en une liste vide après split et strip
    (None, []),
    (["BTCUSDT", "ETHUSDC"], ["BTCUSDT", "ETHUSDC"]), # Déjà une liste
])
def test_comma_separated_pairs_convert_valid(comma_separated_pairs_type, mock_click_context, input_value, expected_output):
    """Teste la conversion de CommaSeparatedPairs avec des entrées valides."""
    result = comma_separated_pairs_type.convert(input_value, None, mock_click_context)
    assert result == expected_output

@pytest.mark.parametrize("invalid_input", [
    123, # Non-string, non-list
    {"pair": "BTC/USDT"}, # Dict
])
def test_comma_separated_pairs_convert_invalid_type(comma_separated_pairs_type, mock_click_context, invalid_input):
    """Teste la conversion avec des types d'entrée invalides."""
    with pytest.raises(click.exceptions.BadParameter) as excinfo:
        comma_separated_pairs_type.convert(invalid_input, None, mock_click_context)
    assert "Expected a string for pairs" in str(excinfo.value) or "is not a valid comma-separated list" in str(excinfo.value)