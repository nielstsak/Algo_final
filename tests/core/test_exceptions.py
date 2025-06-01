# tests/core/test_exceptions.py
import pytest
from typing import Type, Optional, Dict, Any

from src.core.exceptions import (
    AlgoBotException,
    ConfigurationError,
    InvalidConfigurationValueError,
    MissingConfigurationError,
    APIError,
    BinanceAPIError,
    BinanceRateLimitError,
    BinanceInvalidSymbolError,
    BinanceInsufficientFundsError,
    DataError,
    DataDownloadError,
    KlineValidationError,
    UnsupportedDataFormatError,
    StorageError,
    ParquetStorageError,
    PostgresStorageError,
    CacheError,
    StrategyError,
    InvalidStrategyParamsError,
    SignalGenerationError,
    IndicatorCalculationError,
    StrategyLoadError,
    BacktestError,
    BacktestSetupError,
    InsufficientDataForBacktestError,
    OptimizationError,
    OptunaStudyError,
    WFOSplitError,
    LiveTradingError,
    OrderPlacementError,
    OrderExecutionError,
    RiskManagementViolationError,
    PositionManagementError,
    WebSocketConnectionError,
    SerializationError,
    InitializationError
)

# Helper to identify kwargs that are expected to be in `context` vs. direct attributes
# This mapping might need adjustment based on exact __init__ signatures of each exception
CONTEXT_ONLY_KWARGS_BY_CLASS: Dict[Type[AlgoBotException], list[str]] = {
    AlgoBotException: ["custom_param"],
    ConfigurationError: ["config_file"],
    # InvalidConfigurationValueError has 'parameter', 'value' as direct attributes
    # MissingConfigurationError has 'item' as direct attribute
    # APIError has 'api_name', 'status_code' as direct attributes
    # BinanceAPIError has 'binance_code' as direct attribute
    # BinanceInvalidSymbolError has 'symbol' as direct attribute
    # DataDownloadError has 'source' as direct attribute
    # KlineValidationError has 'symbol', 'timestamp' as direct attributes
    # UnsupportedDataFormatError has 'format_name' as direct attribute
    ParquetStorageError: ["path"], # Not defined as direct attr in ParquetStorageError __init__
    PostgresStorageError: ["query"],# Not defined as direct attr in PostgresStorageError __init__
    # StrategyError has 'strategy_name' as direct attribute
    # InvalidStrategyParamsError has 'strategy_name', 'parameter_name', 'details'
    # IndicatorCalculationError has 'indicator_name'
    BacktestSetupError: ["reason"], # Not defined as direct attr in BacktestSetupError __init__
    # InsufficientDataForBacktestError has 'required_period', 'available_period'
    # OptunaStudyError has 'study_name'
    # OrderPlacementError has 'symbol', 'order_details'
    # OrderExecutionError has 'order_id', 'symbol'
    # RiskManagementViolationError has 'rule_violated'
    WebSocketConnectionError: ["stream_url"], # Not defined as direct attr
    # SerializationError has 'data_type'
    # InitializationError has 'component_name'
}


# Liste de toutes les exceptions personnalisées à tester
# Chaque tuple contient: (ClasseException, args_pour_init_constructeur, kwargs_pour_init_complet)
# Pour InvalidStrategyParamsError, args_pour_init_constructeur = (strategy_name, parameter_name, details)
# Pour les autres, args_pour_init_constructeur = (message,)
ALL_CUSTOM_EXCEPTIONS_TO_TEST = [
    (AlgoBotException, ("Base message",), {"original_exception": ValueError("Original error"), "custom_param": "value"}),
    (ConfigurationError, ("Config message",), {"original_exception": TypeError("Original type error"), "config_file": "conf.yaml"}),
    (InvalidConfigurationValueError, ("Invalid value for db.host",), {"parameter": "db.host", "value": 123, "original_exception": ValueError()}),
    (MissingConfigurationError, ("Missing item api_key",), {"item": "api_key", "original_exception": KeyError()}),
    (APIError, ("API generic error",), {"api_name": "ExchangeX", "status_code": 500, "original_exception": ConnectionError()}),
    (BinanceAPIError, ("Binance API error",), {"status_code": 400, "binance_code": -1121, "original_exception": Exception("Binance detail")}),
    (BinanceRateLimitError, ("Rate limit hit",), {"status_code": 429, "binance_code": -1003}),
    (BinanceInvalidSymbolError, ("Invalid symbol XYZ",), {"symbol": "XYZ", "binance_code": -1121}),
    (BinanceInsufficientFundsError, ("Not enough funds",), {"binance_code": -2010}),
    (DataError, ("Generic data error",), {}),
    (DataDownloadError, ("Download failed",), {"source": "Binance", "original_exception": IOError()}),
    (KlineValidationError, ("Invalid kline data",), {"symbol": "BTCUSDT", "timestamp": 1234567890000}),
    (UnsupportedDataFormatError, ("Format X not supported",), {"format_name": "X"}),
    (StorageError, ("Storage access failed",), {}),
    (ParquetStorageError, ("Parquet write error",), {"path": "/data/file.parquet"}),
    (PostgresStorageError, ("Postgres query failed",), {"query": "SELECT *"}),
    (CacheError, ("Cache connection lost",), {}),
    (StrategyError, ("Strategy execution problem",), {"strategy_name": "MyStrategy"}),
    (InvalidStrategyParamsError, ("MyStrategy", "period", "must be > 0"), {"original_exception": TypeError("bad type"), "attempted_value": "abc"}), # constructor_args: strategy_name, parameter_name, details
    (SignalGenerationError, ("Could not generate signal",), {"strategy_name": "MyStrategy"}),
    (IndicatorCalculationError, ("RSI calculation failed",), {"indicator_name": "RSI", "strategy_name": "MyStrategy"}),
    (StrategyLoadError, ("Failed to load MyStrategy",), {"strategy_name": "MyStrategy"}),
    (BacktestError, ("Backtest failed",), {}),
    (BacktestSetupError, ("Backtest setup issue",), {"reason": "Data misaligned"}),
    (InsufficientDataForBacktestError, ("Not enough data",), {"required_period": "1yr", "available_period": "6mo"}),
    (OptimizationError, ("Optimization process error",), {}),
    (OptunaStudyError, ("Optuna study 'my_study' error",), {"study_name": "my_study"}),
    (WFOSplitError, ("WFO split generation failed",), {}),
    (LiveTradingError, ("Live trading system error",), {}),
    (OrderPlacementError, ("Failed to place order",), {"symbol": "BTCUSDT", "order_details": {"side": "BUY", "type": "LIMIT"}}),
    (OrderExecutionError, ("Order execution failed",), {"order_id": "12345", "symbol": "BTCUSDT"}),
    (RiskManagementViolationError, ("Max drawdown exceeded",), {"rule_violated": "max_drawdown"}),
    (PositionManagementError, ("Failed to update position",), {}),
    (WebSocketConnectionError, ("WebSocket disconnected",), {"stream_url": "wss://..."}),
    (SerializationError, ("JSON serialization failed",), {"data_type": "dict"}),
    (InitializationError, ("Component X init failed",), {"component_name": "ComponentX"})
]

@pytest.mark.parametrize("exc_class, constructor_args, full_kwargs", ALL_CUSTOM_EXCEPTIONS_TO_TEST)
def test_custom_exception_raising_and_attributes(
    exc_class: Type[AlgoBotException],
    constructor_args: tuple,
    full_kwargs: Dict[str, Any]
):
    """
    Tests that each custom exception can be raised, caught,
    and that its message and specific attributes/context are correctly set.
    """
    original_exc_instance = full_kwargs.get("original_exception")
    
    # Prepare arguments and kwargs for raising the exception
    init_args = constructor_args
    init_kwargs = full_kwargs.copy()
    if 'original_exception' in init_kwargs: # original_exception is always a kwarg to AlgoBotException
        pass
    
    expected_message_start: str

    if exc_class is InvalidStrategyParamsError:
        strategy_name, parameter_name, details = constructor_args
        expected_message_start = f"Invalid parameter '{parameter_name}' for strategy '{strategy_name}': {details}"
        # For InvalidStrategyParamsError, constructor_args are its specific args
        # The kwargs in full_kwargs (like original_exception, attempted_value) are additional
        current_init_kwargs = {
            "strategy_name": strategy_name,
            "parameter_name": parameter_name,
            "details": details,
        }
        # Merge with remaining full_kwargs (original_exception, other context)
        for k, v in full_kwargs.items():
            if k not in current_init_kwargs:
                current_init_kwargs[k] = v
        init_kwargs = current_init_kwargs
        init_args = () # No positional args, all are kwargs for this specific exception
    else:
        # For other exceptions, the first element of constructor_args is the message
        expected_message_start = constructor_args[0]
        # init_args are already set, init_kwargs are full_kwargs (message is positional)
        # No, AlgoBotException expects (message, original_exception=None, **kwargs)
        # So, the message is positional, others are kwargs.
        init_args = (expected_message_start,)
        init_kwargs.pop('message', None) # Message is now positional

    # 1. Test de la levée et de la capture de l'exception
    with pytest.raises(exc_class) as exc_info:
        raise exc_class(*init_args, **init_kwargs)

    # 2. Vérifier le message et l'exception originale
    raised_exception = exc_info.value
    assert isinstance(raised_exception, exc_class)
    assert isinstance(raised_exception, AlgoBotException)
    
    assert str(raised_exception).startswith(expected_message_start)

    if original_exc_instance:
        assert raised_exception.original_exception is original_exc_instance
        assert f"Caused by: {type(original_exc_instance).__name__}" in str(raised_exception)
    else:
        assert raised_exception.original_exception is None

    # 3. Vérifier les attributs spécifiques et le contexte
    context_kwargs_for_class = CONTEXT_ONLY_KWARGS_BY_CLASS.get(exc_class, [])

    for kwarg_name, expected_value in full_kwargs.items():
        if kwarg_name == 'original_exception':
            continue
        if kwarg_name == 'message' and exc_class is not InvalidStrategyParamsError: # message is positional
            continue

        is_direct_attr = True
        if kwarg_name in context_kwargs_for_class:
            is_direct_attr = False
        # For InvalidStrategyParamsError, its specific constructor args are direct attributes
        elif exc_class is InvalidStrategyParamsError and kwarg_name in ["strategy_name", "parameter_name", "details"]:
             is_direct_attr = True # Already handled by constructor
        # For other exceptions, check if it's a known direct attribute or should be in context
        elif hasattr(exc_class, kwarg_name) and not callable(getattr(exc_class, kwarg_name)) and kwarg_name not in raised_exception.context:
             # If it's a property/slot defined on the class and not explicitly put in context by the base
             is_direct_attr = True


        if is_direct_attr and hasattr(raised_exception, kwarg_name):
            assert getattr(raised_exception, kwarg_name) == expected_value, \
                f"Direct attribute {kwarg_name} of {exc_class.__name__} expected {expected_value}, got {getattr(raised_exception, kwarg_name)}"
        elif kwarg_name in raised_exception.context:
            assert raised_exception.context[kwarg_name] == expected_value, \
                f"Context attribute {kwarg_name} of {exc_class.__name__} expected {expected_value}, got {raised_exception.context[kwarg_name]}"
        else:
            # This case might indicate a kwarg was passed but not stored as a direct attribute or in context.
            # Or the logic to determine is_direct_attr needs refinement for some exceptions.
            # For now, we assume all relevant kwargs are either direct attributes or in context.
            pass # Allow if not found, means it was part of constructor args not meant for direct storage

        # Check if the kwarg appears in the string representation (usually via context)
        # unless it's a primary part of the message for InvalidStrategyParamsError
        is_isp_primary_arg = exc_class is InvalidStrategyParamsError and kwarg_name in ["strategy_name", "parameter_name", "details"]
        
        if not is_isp_primary_arg and expected_value is not None:
            # Check if present in the context part of the string: "[key=value]"
            # Need to handle boolean False correctly, as `str(False)` is "False"
            str_expected_value = str(expected_value)
            assert f"{kwarg_name}={str_expected_value}" in str(raised_exception)


def test_algobotexception_str_formatting():
    """Teste spécifiquement le formatage __str__ de AlgoBotException."""
    # Cas 1: Message simple
    exc1 = AlgoBotException("Simple error")
    assert str(exc1) == "Simple error"

    # Cas 2: Message avec original_exception
    orig_exc = ValueError("Original issue")
    exc2 = AlgoBotException("Error with cause", original_exception=orig_exc)
    assert str(exc2) == "Error with cause (Caused by: ValueError: Original issue)"

    # Cas 3: Message avec kwargs (contexte)
    exc3 = AlgoBotException("Error with context", param1="val1", count=10)
    str_exc3 = str(exc3)
    assert "Error with context" in str_exc3
    assert "param1=val1" in str_exc3
    assert "count=10" in str_exc3
    assert str_exc3.endswith("]") # Context is at the end

    # Cas 4: Message complet
    exc4 = AlgoBotException("Full error", original_exception=orig_exc, param1="val1", count=10)
    str_exc4 = str(exc4)
    assert "Full error" in str_exc4
    assert "Caused by: ValueError: Original issue" in str_exc4
    assert "param1=val1" in str_exc4
    assert "count=10" in str_exc4
    assert str_exc4.endswith("]")

def test_binance_invalid_symbol_error_attributes():
    symbol = "INVALIDPAIR"
    binance_code = -1121
    orig_exc = Exception("Binance raw error")
    
    with pytest.raises(BinanceInvalidSymbolError) as exc_info:
        raise BinanceInvalidSymbolError(
            message=f"Symbol {symbol} is not valid.", 
            symbol=symbol, 
            binance_code=binance_code, 
            original_exception=orig_exc,
            custom_context="some_market_check"
        )
    
    e = exc_info.value
    assert e.message == f"Symbol {symbol} is not valid."
    assert e.symbol == symbol
    assert e.api_name == "Binance" 
    assert e.binance_code == binance_code
    assert e.original_exception is orig_exc
    assert e.context.get("custom_context") == "some_market_check"
    
    str_repr = str(e)
    assert e.message in str_repr
    assert "Binance" in str_repr 
    assert str(binance_code) in str_repr 
    assert "Caused by: Exception: Binance raw error" in str_repr
    assert "custom_context=some_market_check" in str_repr
    assert f"symbol={symbol}" in str_repr # symbol is passed as kwarg to super, so it's in context

def test_invalid_strategy_params_error_constructor_and_str():
    strategy_name = "TestStrategy"
    parameter_name = "window_size"
    details = "must be an integer greater than 0"
    orig_exc = TypeError("Bad type for window_size")

    with pytest.raises(InvalidStrategyParamsError) as exc_info:
        raise InvalidStrategyParamsError(
            strategy_name=strategy_name,
            parameter_name=parameter_name,
            details=details,
            original_exception=orig_exc,
            attempted_value="abc" # This will go into context
        )
    
    e = exc_info.value
    expected_message = f"Invalid parameter '{parameter_name}' for strategy '{strategy_name}': {details}"
    assert e.message == expected_message # The main message
    assert e.strategy_name == strategy_name
    assert e.parameter_name == parameter_name
    assert e.details == details
    assert e.original_exception is orig_exc
    assert e.context.get("attempted_value") == "abc"

    str_repr = str(e)
    assert expected_message in str_repr # Main message part
    assert "Caused by: TypeError: Bad type for window_size" in str_repr # Original exception part
    
    # strategy_name, parameter_name, details are passed as kwargs to super, so they are in context
    assert f"strategy_name={strategy_name}" in str_repr 
    assert f"parameter_name={parameter_name}" in str_repr
    assert f"details={details}" in str_repr
    assert "attempted_value=abc" in str_repr # Other context