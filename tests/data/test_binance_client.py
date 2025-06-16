# tests/data/test_binance_client.py
import pytest
from decimal import Decimal
from src.data.binance_client import BinanceDataClient # _validate_kline_data est une méthode d'instance
from src.core.exceptions import KlineValidationError

@pytest.fixture
def data_client_instance():
    """Crée une instance de BinanceDataClient pour les tests."""
    # Pas besoin de clés API pour tester _validate_kline_data
    return BinanceDataClient(api_key="test", api_secret="test")

def test_validate_kline_data_valid(data_client_instance: BinanceDataClient):
    """Teste _validate_kline_data avec une kline valide."""
    raw_kline = [
        1672531200000,  # Open time
        "100.0",        # Open
        "110.0",        # High
        "90.0",         # Low
        "105.0",        # Close
        "1000.0",       # Volume
        1672531259999,  # Close time
        "105000.0",     # Quote asset volume
        100,            # Number of trades
        "500.0",        # Taker buy base asset volume
        "52500.0",      # Taker buy quote asset volume
        "0"             # Ignore
    ]
    symbol = "BTCUSDT"
    validated = data_client_instance._validate_kline_data(raw_kline, symbol)

    assert validated['open_time'] == 1672531200000
    assert validated['open'] == 100.0
    assert validated['high'] == 110.0
    assert validated['low'] == 90.0
    assert validated['close'] == 105.0
    assert validated['volume'] == 1000.0
    assert validated['number_of_trades'] == 100

def test_validate_kline_data_high_less_than_low(data_client_instance: BinanceDataClient, caplog):
    """Teste _validate_kline_data où high < low."""
    raw_kline = [1672531200000, "100", "90", "110", "105", "1000", 1672531259999, "105000", 100, "500", "52500", "0"]
    symbol = "BTCUSDT"
    # La logique actuelle "corrige" cela en prenant max/min de O, H, L, C.
    # Si la nouvelle logique lève une exception, ce test devra être adapté.
    # Avec la correction du diff précédent, cela devrait lever une ValueError dans _validate_kline_data
    # qui est ensuite encapsulée dans KlineValidationError par fetch_klines.
    # Ici, nous testons _validate_kline_data directement.
    # La logique de "correction" a été retirée, donc une erreur est attendue.
    with pytest.raises(KlineValidationError) as excinfo:
        data_client_instance._validate_kline_data(raw_kline, symbol)
    assert "High price 90 is less than low price 110" in str(excinfo.value.original_exception)


@pytest.mark.parametrize("field_to_invalidate, bad_value, expected_error_msg_part", [
    (1, "not_a_number", "could not convert string to Decimal"), # Invalid open
    (5, "-100", "Base asset volume -100 is negative"), # Negative volume
    (0, "not_an_int", "invalid literal for int()"), # Invalid open_time
    (6, 1672531100000, "Close time 1672531100000 is before or same as open time 1672531200000"), # close_time < open_time
])
def test_validate_kline_data_invalid_types_or_values(data_client_instance: BinanceDataClient, field_to_invalidate, bad_value, expected_error_msg_part):
    """Teste _validate_kline_data avec divers types ou valeurs invalides."""
    raw_kline = [1672531200000, "100", "110", "90", "105", "1000", 1672531259999, "105000", 100, "500", "52500", "0"]
    raw_kline[field_to_invalidate] = bad_value
    symbol = "BTCUSDT"

    with pytest.raises(KlineValidationError) as excinfo:
        data_client_instance._validate_kline_data(raw_kline, symbol)
    
    assert expected_error_msg_part.lower() in str(excinfo.value.original_exception).lower()

def test_validate_kline_data_incomplete(data_client_instance: BinanceDataClient):
    """Teste _validate_kline_data avec des données kline incomplètes."""
    raw_kline = [1672531200000, "100", "110"] # Seulement 3 champs
    symbol = "BTCUSDT"
    with pytest.raises(KlineValidationError) as excinfo:
        data_client_instance._validate_kline_data(raw_kline, symbol)
    assert "Kline data is incomplete" in str(excinfo.value.original_exception)

@pytest.mark.parametrize("open_val, high_val, low_val, close_val, expected_error_msg_part", [
    ("115", "110", "90", "105", "Open price 115 is greater than high price 110"), # open > high
    ("85", "110", "90", "105", "Open price 85 is less than low price 90"),    # open < low
    ("100", "110", "90", "115", "Close price 115 is greater than high price 110"),# close > high
    ("100", "110", "90", "85", "Close price 85 is less than low price 90"),   # close < low
])
def test_validate_kline_data_ohlc_logic_violations(data_client_instance: BinanceDataClient, open_val, high_val, low_val, close_val, expected_error_msg_part):
    """Teste les violations de logique OHLC où O ou C sont en dehors de H-L."""
    raw_kline = [
        1672531200000, open_val, high_val, low_val, close_val,
        "1000", 1672531259999, "105000", 100, "500", "52500", "0"
    ]
    symbol = "TESTOHLC"
    with pytest.raises(KlineValidationError) as excinfo:
        data_client_instance._validate_kline_data(raw_kline, symbol)
    assert expected_error_msg_part in str(excinfo.value.original_exception)