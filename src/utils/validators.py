import os
import re
from decimal import Decimal
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
from pydantic import (BaseModel, Field, ValidationError, condecimal, confloat,
                      conint, constr, validator)
from pydantic.types import PositiveFloat

# --- Type Aliases and Constants ---
ApiStr = constr(min_length=10) # Example constraint for API key/secret length
SymbolStr = constr(min_length=3, max_length=20, regex=r"^[A-Z0-9]+USDC$") # USDC pairs

# --- Pydantic Models ---

class OCOOrder(BaseModel):
    """
    Represents parameters for an OCO (One-Cancels-Other) order leg,
    typically used within a TradingSignal.
    """
    take_profit_price: PositiveFloat
    stop_loss_price: PositiveFloat
    stop_loss_limit_price: Optional[PositiveFloat] = None # Required for STOP_LOSS_LIMIT

    @validator('take_profit_price', 'stop_loss_price', 'stop_loss_limit_price', pre=True, allow_reuse=True)
    def ensure_positive_optional(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v

    @validator('stop_loss_limit_price', always=True)
    def check_stop_prices(cls, v, values):
        # For a SELL OCO (protecting a long): tp_price > entry > sl_price > sl_limit_price
        # For a BUY OCO (protecting a short): tp_price < entry < sl_price < sl_limit_price
        # This validation is partial as 'side' and 'entry' are not in this model.
        # More comprehensive validation would occur when an OCO order is actually constructed.
        if 'stop_loss_price' in values and v is not None:
            if values.get('side') == 'SELL' and v >= values['stop_loss_price']: # Example for SELL OCO
                 raise ValueError("For a SELL OCO, stop_loss_limit_price must be less than stop_loss_price.")
            # Add similar check for BUY OCO if side was available here.
        return v

class KlineData(BaseModel):
    """
    Validates a single Kline (candlestick) data point.
    """
    timestamp: conint(gt=0) # Unix MS timestamp
    open: PositiveFloat
    high: PositiveFloat
    low: PositiveFloat
    close: PositiveFloat
    volume: PositiveFloat

    @validator('high')
    def high_gte_open_close_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= Low')
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= Open')
        if 'close' in values and v < values['close']:
            raise ValueError('High must be >= Close')
        return v

    @validator('low')
    def low_lte_open_close(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= Open')
        if 'close' in values and v > values['close']:
            raise ValueError('Low must be <= Close')
        return v
    
    class Config:
        validate_assignment = True


class TradingSignal(BaseModel):
    """
    Validates the structure and content of a trading signal.
    """
    signal_id: constr(min_length=1) = Field(..., example="signal_20230101_BTCUSDC_001")
    timestamp_exchange: conint(gt=0) # Exchange timestamp of kline/event triggering signal
    timestamp_generated: conint(gt=0) # Bot's timestamp when signal was generated
    signal_type: Literal['LONG', 'SHORT']
    side: Literal['BUY', 'SELL'] # BUY for LONG entry, SELL for SHORT entry
    pair: SymbolStr # Validated by SymbolStr to end with USDC
    
    entry_price_target: Optional[PositiveFloat] = None # For LIMIT orders
    # For MARKET orders, this might be None, or current market price for reference
    
    quantity_base_asset: PositiveFloat # Quantity in the base asset (e.g., BTC amount for BTCUSDC)
    
    # Confidence of the signal, can be used for position sizing or filtering
    confidence: confloat(ge=0.0, le=1.0) = 1.0
    
    # OCO order parameters if this signal is for an entry that requires immediate TP/SL
    oco_order_params: Optional[OCOOrder] = None # Parameters to create an OCO order
    
    strategy_name: constr(min_length=1)
    metadata: Dict[str, Any] = {} # For additional info like indicator values

    @validator('side')
    def side_matches_signal_type(cls, v, values):
        if 'signal_type' in values:
            if values['signal_type'] == 'LONG' and v != 'BUY':
                raise ValueError("For LONG signal_type, side must be 'BUY'")
            if values['signal_type'] == 'SHORT' and v != 'SELL':
                raise ValueError("For SHORT signal_type, side must be 'SELL'")
        return v

    @validator('oco_order_params', always=True)
    def oco_prices_valid_for_side(cls, v, values):
        if v is None or 'side' not in values:
            return v
        
        # If entry is BUY (LONG), OCO is SELL. TP > SL.
        # If entry is SELL (SHORT), OCO is BUY. TP < SL.
        # This model's 'side' is for the entry order. OCO side is opposite.
        entry_side = values['side']
        
        if entry_side == 'BUY': # Protecting a LONG position, OCO will be SELL
            if v.take_profit_price <= v.stop_loss_price:
                raise ValueError("For LONG entry (SELL OCO), take_profit_price must be > stop_loss_price.")
            if v.stop_loss_limit_price is not None and v.stop_loss_limit_price >= v.stop_loss_price:
                raise ValueError("For LONG entry (SELL OCO), stop_loss_limit_price must be < stop_loss_price.")
        elif entry_side == 'SELL': # Protecting a SHORT position, OCO will be BUY
            if v.take_profit_price >= v.stop_loss_price:
                raise ValueError("For SHORT entry (BUY OCO), take_profit_price must be < stop_loss_price.")
            if v.stop_loss_limit_price is not None and v.stop_loss_limit_price <= v.stop_loss_price: # Price increases for BUY stop
                raise ValueError("For SHORT entry (BUY OCO), stop_loss_limit_price must be > stop_loss_price.")
        return v

    class Config:
        validate_assignment = True


class RiskParameters(BaseModel):
    """
    Validates risk management parameters.
    """
    max_position_pct_balance: confloat(gt=0, le=0.5) # Max % of balance for one position
    max_total_exposure_pct_balance: confloat(gt=0, le=1.0) # Max % of balance for all positions
    max_drawdown_pct_strategy: confloat(gt=0, le=0.5) # Max drawdown for a single strategy
    daily_loss_limit_pct_balance: confloat(gt=0, le=0.1) # Max daily loss for total balance
    
    # Per-pair limits (e.g., max notional value or max quantity)
    # Key: symbol string (e.g., "BTCUSDC"), Value: max exposure (e.g., in quote currency or base units)
    # This needs careful definition of what the value represents.
    # For this example, let's assume it's max notional in quote currency.
    position_notional_limits_quote: Dict[SymbolStr, PositiveFloat] = {}
    
    # Max number of open positions allowed simultaneously
    max_open_positions: conint(ge=1) = 10

    class Config:
        validate_assignment = True

# --- Decorators ---

def validate_inputs(**pydantic_models: Type[BaseModel]) -> Callable:
    """
    Decorator to validate function arguments against Pydantic models.
    Pass argument_name=ModelClass in decorator.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a dictionary of all passed arguments by name
            func_args = func.__code__.co_varnames[:func.__code__.co_argcount]
            all_passed_args = {**dict(zip(func_args, args)), **kwargs}

            for arg_name, model_class in pydantic_models.items():
                if arg_name in all_passed_args:
                    try:
                        # Pydantic models expect dicts for validation usually
                        # If the arg is already a dict, fine. If it's an object, convert if needed.
                        # For simplicity, assume it's a dict or Pydantic will handle.
                        model_class.model_validate(all_passed_args[arg_name])
                    except ValidationError as e:
                        # Consider logging here instead of just raising
                        raise ValueError(f"Argument '{arg_name}' validation failed for {func.__name__}: {e}") from e
                # else: # Argument not passed, Pydantic model might have defaults or it's optional
                    # print(f"Warning: Argument '{arg_name}' for validation not found in call to {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_dataframe_schema(
    df: pd.DataFrame,
    schema: Dict[str, Type],
    required_cols: Optional[List[str]] = None,
    check_nan: bool = True,
    check_inf: bool = True
) -> None:
    """
    Helper function to validate a DataFrame's schema, dtypes, and optionally NaNs/Infs.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema : Dict[str, Type]
        A dictionary where keys are column names and values are expected Python types
        (e.g., int, float, str, bool) or numpy dtypes (e.g., np.int64, np.float64).
        For datetime, use `np.datetime64` or check with `pd.api.types.is_datetime64_any_dtype`.
    required_cols : Optional[List[str]], optional
        A list of column names that must be present in the DataFrame.
        If None, all columns in `schema` are considered required.
    check_nan : bool, optional
        If True, checks for NaN values in schema-defined columns. Defaults to True.
    check_inf : bool, optional
        If True, checks for Inf values in schema-defined numeric columns. Defaults to True.

    Raises
    ------
    ValueError
        If validation fails (missing columns, wrong dtype, NaN/Inf found).
    TypeError
        If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cols_to_check = required_cols if required_cols is not None else list(schema.keys())

    # Check for missing columns
    missing_cols = [col for col in cols_to_check if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_cols)}")

    # Check data types and content
    for col_name, expected_type in schema.items():
        if col_name not in df.columns: # Only check columns present or specified in schema
            continue

        actual_dtype = df[col_name].dtype

        # Type checking
        type_match = False
        if expected_type == np.datetime64:
            type_match = pd.api.types.is_datetime64_any_dtype(actual_dtype)
        elif pd.api.types.is_object_dtype(actual_dtype) and expected_type == str:
             # For object dtype, ensure all elements are strings if expected_type is str
             # This check can be slow for large DFs.
             # if not df[col_name].empty and not df[col_name].apply(lambda x: isinstance(x, str) or pd.isna(x)).all():
             #    raise ValueError(f"Column '{col_name}' expected type {expected_type}, found mixed types in object dtype (not all str).")
             type_match = True # Assume object can hold strings; more specific check is costly
        elif pd.api.types.is_numeric_dtype(actual_dtype) and expected_type in [int, float, np.integer, np.floating, PositiveFloat]:
            # Check if convertible or actual type matches broad category
            if expected_type == int and not pd.api.types.is_integer_dtype(actual_dtype) and not (pd.api.types.is_float_dtype(actual_dtype) and (df[col_name]%1==0).all()):
                pass # Potential mismatch, will be caught if strict type needed by np.can_cast
            type_match = np.can_cast(actual_dtype, expected_type, casting='safe') \
                         or (expected_type == PositiveFloat and pd.api.types.is_float_dtype(actual_dtype)) \
                         or (expected_type == int and pd.api.types.is_integer_dtype(actual_dtype)) \
                         or (expected_type == float and pd.api.types.is_float_dtype(actual_dtype))

        elif actual_dtype == expected_type or np.issubdtype(actual_dtype, expected_type):
            type_match = True
        
        if not type_match:
            raise ValueError(f"Column '{col_name}' has dtype {actual_dtype}, expected {expected_type} or compatible.")

        # Check for NaN values
        if check_nan and df[col_name].isnull().any():
            raise ValueError(f"Column '{col_name}' contains NaN values.")

        # Check for Inf values in numeric columns
        if check_inf and pd.api.types.is_numeric_dtype(actual_dtype):
            if np.isinf(df[col_name].replace([np.inf, -np.inf], np.nan).dropna()).any(): # Check after dropping NaNs
                 # Check if any original non-NaN values were Inf
                 if np.isinf(df[col_name]).any():
                    raise ValueError(f"Column '{col_name}' contains Inf values.")


def validate_dataframe(schema: Dict[str, Type], required_cols: Optional[List[str]] = None,
                       check_nan: bool = True, check_inf: bool = True) -> Callable:
    """
    Decorator to validate a DataFrame passed as the first argument to a function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            validate_dataframe_schema(df, schema, required_cols, check_nan, check_inf)
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


# --- Core Validators ---

@validate_dataframe(
    schema={'open': PositiveFloat, 'high': PositiveFloat, 'low': PositiveFloat, 'close': PositiveFloat, 'volume': PositiveFloat},
    required_cols=['open', 'high', 'low', 'close', 'volume'],
    check_nan=True, check_inf=True
)
def validate_ohlcv_df_integrity(df: pd.DataFrame) -> bool:
    """
    Validates the integrity of an OHLCV DataFrame.
    Uses the @validate_dataframe decorator for schema checks.
    Adds specific OHLCV logic checks.

    Parameters
    ----------
    df : pd.DataFrame
        The OHLCV DataFrame with a DatetimeIndex.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If data integrity checks fail (e.g., high < low).
    TypeError
        If df.index is not a DatetimeIndex.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if not df.index.is_unique:
        raise ValueError("DataFrame DatetimeIndex must be unique.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame DatetimeIndex must be monotonic increasing.")

    # Check OHLCV relationships (H >= O, H >= C, H >= L and L <= O, L <= C)
    if not (df['high'] >= df['low']).all():
        raise ValueError("Data integrity violated: Not all 'high' >= 'low'.")
    if not (df['high'] >= df['open']).all():
        raise ValueError("Data integrity violated: Not all 'high' >= 'open'.")
    if not (df['high'] >= df['close']).all():
        raise ValueError("Data integrity violated: Not all 'high' >= 'close'.")
    if not (df['low'] <= df['open']).all():
        raise ValueError("Data integrity violated: Not all 'low' <= 'open'.")
    if not (df['low'] <= df['close']).all():
        raise ValueError("Data integrity violated: Not all 'low' <= 'close'.")
    
    # Check for non-positive prices/volume (already covered by PositiveFloat in schema)
    # if (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
    #     raise ValueError("OHLCV prices and volume must be positive.")
    return True


def validate_strategy_params(
    params: Dict[str, Any],
    strategy_model: Type[BaseModel] # Expect a Pydantic model for the strategy's params
) -> BaseModel:
    """
    Validates strategy parameters against a Pydantic model defining the strategy's requirements.

    Parameters
    ----------
    params : Dict[str, Any]
        The parameters to validate.
    strategy_model : Type[BaseModel]
        The Pydantic model class that defines the expected parameters for the strategy.

    Returns
    -------
    BaseModel
        The validated and parsed parameters object (an instance of strategy_model).

    Raises
    ------
    ValueError
        If validation fails due to mismatch with the strategy_model.
    """
    try:
        validated_params = strategy_model.model_validate(params)
        return validated_params
    except ValidationError as e:
        # Log detailed error: e.errors() provides structured info
        # For simplicity, raising a ValueError here.
        error_details = "\n".join([f"  - {err['loc'][0] if err['loc'] else 'field'}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Invalid strategy parameters for {strategy_model.__name__}:\n{error_details}") from e


def validate_signal_format(signal_data: Dict[str, Any]) -> TradingSignal:
    """
    Validates and parses a trading signal dictionary into a TradingSignal Pydantic model.

    Parameters
    ----------
    signal_data : Dict[str, Any]
        The raw signal data as a dictionary.

    Returns
    -------
    TradingSignal
        A validated TradingSignal object.

    Raises
    ------
    ValueError
        If the signal_data does not conform to the TradingSignal model.
    """
    try:
        return TradingSignal.model_validate(signal_data)
    except ValidationError as e:
        error_details = "\n".join([f"  - {err['loc'][0] if err['loc'] else 'field'}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Invalid trading signal format:\n{error_details}") from e


def validate_position_size(
    requested_size_base_asset: float,
    asset_price_quote: float, # Current price of base asset in quote currency
    account_balance_quote: float,
    risk_params: RiskParameters,
    symbol: SymbolStr,
    current_total_exposure_quote: float, # Current total exposure of all positions in quote
    current_open_positions_count: int
) -> bool:
    """
    Validates a requested position size against risk parameters and account balance.

    Parameters
    ----------
    requested_size_base_asset : float
        The requested quantity of the base asset.
    asset_price_quote : float
        The current price of the base asset in the quote currency.
    account_balance_quote : float
        The total account balance in the quote currency.
    risk_params : RiskParameters
        The Pydantic model containing risk limits.
    symbol : SymbolStr
        The trading symbol for per-pair limits.
    current_total_exposure_quote : float
        Current total exposure of all open positions in quote currency.
    current_open_positions_count : int
        Number of currently open positions.

    Returns
    -------
    bool
        True if the position size is valid.

    Raises
    ------
    ValueError
        If the position size violates any risk limits.
    """
    if requested_size_base_asset <= 0:
        raise ValueError("Requested position size (base asset) must be positive.")
    if asset_price_quote <= 0:
        raise ValueError("Asset price must be positive.")
    if account_balance_quote <= 0: # Or some minimum threshold
        raise ValueError("Account balance must be positive to open new positions.")

    requested_notional_quote = Decimal(str(requested_size_base_asset)) * Decimal(str(asset_price_quote))

    # 1. Check against max_position_pct_balance
    max_pos_notional_allowed = Decimal(str(account_balance_quote)) * Decimal(str(risk_params.max_position_pct_balance))
    if requested_notional_quote > max_pos_notional_allowed:
        raise ValueError(
            f"Requested notional {requested_notional_quote:.2f} USDC exceeds max per position "
            f"({risk_params.max_position_pct_balance*100:.1f}% of balance = {max_pos_notional_allowed:.2f} USDC)."
        )

    # 2. Check against per-symbol notional limits
    if symbol in risk_params.position_notional_limits_quote:
        symbol_notional_limit = Decimal(str(risk_params.position_notional_limits_quote[symbol]))
        if requested_notional_quote > symbol_notional_limit:
            raise ValueError(
                f"Requested notional {requested_notional_quote:.2f} for {symbol} exceeds "
                f"its limit of {symbol_notional_limit:.2f} USDC."
            )
            
    # 3. Check against max_total_exposure_pct_balance
    new_total_exposure = Decimal(str(current_total_exposure_quote)) + requested_notional_quote
    max_total_exposure_allowed = Decimal(str(account_balance_quote)) * Decimal(str(risk_params.max_total_exposure_pct_balance))
    if new_total_exposure > max_total_exposure_allowed:
        raise ValueError(
            f"New total exposure {new_total_exposure:.2f} USDC would exceed max total exposure "
            f"({risk_params.max_total_exposure_pct_balance*100:.1f}% of balance = {max_total_exposure_allowed:.2f} USDC)."
        )

    # 4. Check against max_open_positions
    # This check applies if this new position would be an *additional* open position.
    # If it's increasing an existing one, this specific check might not apply directly here,
    # but rather when deciding to open a *new symbol* position.
    # Assuming this is for opening a new position or one that increases the count:
    if current_open_positions_count + 1 > risk_params.max_open_positions:
        raise ValueError(
             f"Opening this position would exceed max open positions limit of {risk_params.max_open_positions} "
             f"(current: {current_open_positions_count})."
        )
        
    return True


def validate_api_credentials(api_key: str, api_secret: str) -> bool:
    """
    Validates the basic format of API credentials.
    Actual validation occurs when making an API call.

    Parameters
    ----------
    api_key : str
        The API key.
    api_secret : str
        The API secret.

    Returns
    -------
    bool
        True if basic format seems plausible.

    Raises
    ------
    ValueError
        If format is clearly invalid (e.g., empty).
    """
    if not api_key or not isinstance(api_key, str):
        raise ValueError("API key must be a non-empty string.")
    if not api_secret or not isinstance(api_secret, str):
        raise ValueError("API secret must be a non-empty string.")
    
    # Example: Binance API keys often have a certain length.
    # This is a loose check, actual format can vary.
    if len(api_key) < 20 or len(api_secret) < 20:
        # print("Warning: API key or secret seems unusually short.") # Or raise error for stricter check
        pass # Allow shorter ones for testing or other exchanges
    return True


# --- Range and Format Validators ---

def validate_price_range(
    price: float,
    symbol: str, # e.g., "BTCUSDC"
    historical_avg_price: Optional[float] = None, # Optional: for comparison
    max_deviation_pct: float = 0.5 # e.g., 50% deviation from historical average
) -> bool:
    """
    Validates if a price is within a "reasonable" range.
    This is highly conceptual and needs context (e.g., current market data, symbol filters).
    A simple implementation might check against extreme deviations if an average is known.

    Parameters
    ----------
    price : float
        The price to validate.
    symbol : str
        The trading symbol (for context, not used in this simplified example).
    historical_avg_price : Optional[float], optional
        An optional historical average price for comparison.
    max_deviation_pct : float, optional
        Maximum allowed percentage deviation from the historical average. Defaults to 0.5 (50%).

    Returns
    -------
    bool
        True if the price is considered within a reasonable range.

    Raises
    ------
    ValueError
        If the price is outside the defined reasonable range.
    """
    if price <= 0:
        raise ValueError(f"Price for {symbol} must be positive, got {price}.")

    if historical_avg_price is not None and historical_avg_price > 0:
        lower_bound = historical_avg_price * (1 - max_deviation_pct)
        upper_bound = historical_avg_price * (1 + max_deviation_pct)
        if not (lower_bound <= price <= upper_bound):
            raise ValueError(
                f"Price {price} for {symbol} is outside the reasonable range "
                f"[{lower_bound:.2f} - {upper_bound:.2f}] "
                f"(based on avg {historical_avg_price:.2f} and {max_deviation_pct*100}% deviation)."
            )
    # Add more checks: e.g., against Binance PRICE_FILTER (minPrice, maxPrice, tickSize)
    # This would require fetching symbol info, similar to exchange_utils.py
    return True

# Re-using from time_utils.py for consistency if available, or define locally
VALID_TIMEFRAME_STRINGS = ['1T','3T','5T','15T','30T','1H','2H','4H','6H','8H','12H','1D','3D','1W','1M', # Pandas style
                           '1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w'] # Common style

def validate_timeframe(timeframe: str) -> bool:
    """
    Validates if a timeframe string is recognized.

    Parameters
    ----------
    timeframe : str
        The timeframe string (e.g., "1m", "1H", "15T").

    Returns
    -------
    bool
        True if the timeframe is valid.

    Raises
    ------
    ValueError
        If the timeframe string is not recognized.
    """
    # This can be expanded with regex or by trying to convert to pandas offset
    if timeframe not in VALID_TIMEFRAME_STRINGS:
        try:
            # Check if it's a valid pandas frequency string
            pd.tseries.frequencies.to_offset(timeframe)
            return True
        except ValueError:
            raise ValueError(
                f"Invalid timeframe string: '{timeframe}'. "
                f"Supported examples: {', '.join(VALID_TIMEFRAME_STRINGS[:5])}..."
            )
    return True


# --- Composite Validators (Order Specific) ---

class BaseOrderParams(BaseModel):
    symbol: SymbolStr
    side: Literal["BUY", "SELL"]
    quantity_base_asset: PositiveFloat # Quantity in base asset
    # client_order_id: Optional[constr(max_length=36)] # User-defined ID

class LimitOrderParams(BaseOrderParams):
    price: PositiveFloat
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"

class MarketOrderParams(BaseOrderParams):
    # For MARKET orders, quantity_base_asset is usually used.
    # quote_order_qty: Optional[PositiveFloat] = None # Alternative for MARKET by quote asset amount
    pass

class OcoOrderParamsStructure(BaseModel): # For validating the structure of an OCO request
    symbol: SymbolStr
    side: Literal["BUY", "SELL"] # Side of the OCO order itself (e.g. SELL OCO for a long position)
    quantity: PositiveFloat
    price: PositiveFloat # Price of the limit leg (take profit)
    stop_price: PositiveFloat # Trigger price for the stop loss leg
    stop_limit_price: Optional[PositiveFloat] = None # Limit price for the stop loss leg (if STOP_LOSS_LIMIT)
    # ... other OCO specific params like listClientOrderId, stopLimitTimeInForce etc.

class OrderValidator:
    """
    Validates different types of orders.
    These methods would typically also interact with exchange_utils to check against symbol filters.
    """
    @staticmethod
    def validate_limit_order(order_data: Dict[str, Any]) -> LimitOrderParams:
        """Validates parameters for a LIMIT order."""
        try:
            # Further checks against exchange_utils.get_symbol_info().filters would go here
            # e.g., price % tickSize, quantity % stepSize, quantity vs min/maxQty, notional vs minNotional
            return LimitOrderParams.model_validate(order_data)
        except ValidationError as e:
            raise ValueError(f"Invalid LIMIT order parameters: {e}")

    @staticmethod
    def validate_market_order(order_data: Dict[str, Any]) -> MarketOrderParams:
        """Validates parameters for a MARKET order."""
        try:
            # Further checks: quantity vs min/maxQty, notional vs minNotional (if applicable to market)
            return MarketOrderParams.model_validate(order_data)
        except ValidationError as e:
            raise ValueError(f"Invalid MARKET order parameters: {e}")

    @staticmethod
    def validate_oco_submission(oco_data: Dict[str, Any]) -> OcoOrderParamsStructure:
        """Validates parameters for submitting an OCO order group."""
        try:
            # This validates the structure of the OCO request itself.
            # Individual legs (limit maker for TP, stop loss limit for SL) also need validation.
            # Price relationship checks (TP vs SL) are important.
            validated_oco = OcoOrderParamsStructure.model_validate(oco_data)
            
            # Example price relationship check for OCO
            # This logic is similar to TradingSignal.oco_order_params validator
            if validated_oco.side == 'SELL': # OCO is selling (protecting a long)
                if validated_oco.price <= validated_oco.stop_price:
                    raise ValueError("For SELL OCO, take_profit price (price field) must be > stop_price.")
                if validated_oco.stop_limit_price and validated_oco.stop_limit_price >= validated_oco.stop_price:
                    raise ValueError("For SELL OCO, stop_limit_price must be < stop_price.")
            elif validated_oco.side == 'BUY': # OCO is buying (protecting a short)
                if validated_oco.price >= validated_oco.stop_price:
                    raise ValueError("For BUY OCO, take_profit price (price field) must be < stop_price.")
                if validated_oco.stop_limit_price and validated_oco.stop_limit_price <= validated_oco.stop_price: # Price increases for buy stop
                    raise ValueError("For BUY OCO, stop_limit_price must be > stop_price.")
            return validated_oco
        except ValidationError as e:
            raise ValueError(f"Invalid OCO order parameters: {e}")


# --- Security Validators ---

def sanitize_input(
    value: Any,
    expected_type: Type,
    max_length: Optional[int] = None,
    allowed_pattern: Optional[str] = None # Regex pattern
) -> Any:
    """
    Basic input sanitization and type validation.
    This is a conceptual function; real sanitization is context-specific and complex.

    Parameters
    ----------
    value : Any
        The input value to sanitize/validate.
    expected_type : Type
        The Python type the value is expected to conform to after sanitization.
    max_length : Optional[int], optional
        Maximum allowed length if the value is a string.
    allowed_pattern : Optional[str], optional
        A regex pattern that the string value must match.

    Returns
    -------
    Any
        The sanitized value, converted to `expected_type` if possible.

    Raises
    ------
    ValueError
        If sanitization/validation fails.
    """
    if expected_type == str:
        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                raise ValueError(f"Input could not be converted to string: {type(value)}")
        if max_length is not None and len(value) > max_length:
            raise ValueError(f"Input string exceeds max length {max_length}.")
        if allowed_pattern:
            if not re.match(allowed_pattern, value):
                raise ValueError(f"Input string does not match allowed pattern: '{allowed_pattern}'.")
        # Basic sanitization: strip whitespace
        value = value.strip()
        # Further sanitization (e.g., escaping HTML, SQL) is highly context-dependent and
        # should be done with appropriate libraries (e.g., bleach, database ORM).
        return value
    
    elif expected_type in [int, float, Decimal, PositiveFloat]:
        try:
            if expected_type == Decimal:
                return Decimal(str(value))
            elif expected_type == PositiveFloat: # Pydantic type, check positivity
                val = float(value)
                if val <=0: raise ValueError("Expected PositiveFloat")
                return val
            return expected_type(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Input '{value}' could not be converted to {expected_type.__name__}: {e}") from e
    
    elif isinstance(expected_type, type(BaseModel)): # If expected_type is a Pydantic model class
        try:
            return expected_type.model_validate(value)
        except ValidationError as e:
            raise ValueError(f"Input does not conform to Pydantic model {expected_type.__name__}: {e}") from e
            
    if not isinstance(value, expected_type):
        raise ValueError(f"Input type {type(value)} does not match expected type {expected_type}.")
    
    return value


def validate_file_path(
    path_to_check: str,
    allowed_base_dirs: Optional[List[str]] = None,
    must_exist: bool = False,
    check_is_file: Optional[bool] = None # True for file, False for dir, None for either
) -> str:
    """
    Validates a file path for security and existence.
    Prevents path traversal and ensures path is within allowed directories if specified.

    Parameters
    ----------
    path_to_check : str
        The file or directory path to validate.
    allowed_base_dirs : Optional[List[str]], optional
        A list of absolute base directory paths. If provided, the `path_to_check`
        must resolve to be within one of these directories.
    must_exist : bool, optional
        If True, the path must exist on the filesystem. Defaults to False.
    check_is_file : Optional[bool], optional
        If True, path must be a file. If False, must be a directory.
        If None, no specific type check. Only applies if `must_exist` is True.

    Returns
    -------
    str
        The absolute, normalized path if validation passes.

    Raises
    ------
    ValueError
        If the path is invalid, insecure, or does not meet existence/type criteria.
    """
    if not isinstance(path_to_check, str) or not path_to_check:
        raise ValueError("Path to check must be a non-empty string.")

    # Normalize the path to resolve '..' etc. and get an absolute path
    abs_path = os.path.abspath(os.path.normpath(path_to_check))

    # Check if path is within allowed base directories
    if allowed_base_dirs:
        # Normalize allowed_base_dirs as well
        normalized_allowed_dirs = [os.path.abspath(os.path.normpath(d)) for d in allowed_base_dirs]
        
        is_allowed = False
        for allowed_dir in normalized_allowed_dirs:
            if os.path.commonpath([abs_path, allowed_dir]) == allowed_dir:
                is_allowed = True
                break
        if not is_allowed:
            raise ValueError(
                f"Path '{abs_path}' is not within allowed base directories: {allowed_base_dirs}"
            )

    # Check for existence and type if required
    if must_exist:
        if not os.path.exists(abs_path):
            raise ValueError(f"Path does not exist: '{abs_path}'")
        if check_is_file is True and not os.path.isfile(abs_path):
            raise ValueError(f"Path is not a file: '{abs_path}'")
        if check_is_file is False and not os.path.isdir(abs_path):
            raise ValueError(f"Path is not a directory: '{abs_path}'")
            
    # Potentially check for null bytes or other malicious patterns if path comes from untrusted input
    if '\0' in abs_path:
        raise ValueError("Path contains null bytes, which is not allowed.")

    return abs_path


if __name__ == '__main__':
    print("--- Testing validators.py ---")

    # KlineData Validation
    try:
        kline = KlineData(timestamp=1672531200000, open=100, high=100, low=101, close=98, volume=1000)
    except ValidationError as e:
        print(f"KlineData validation error (expected): {e.errors()[0]['msg']}") # high < low

    valid_kline_dict = {"timestamp": 1672531200000, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000.0}
    kline_obj = KlineData.model_validate(valid_kline_dict)
    print(f"Valid KlineData: {kline_obj.low=}, {kline_obj.high=}")


    # TradingSignal Validation
    valid_signal_data = {
        "signal_id": "sig1", "timestamp_exchange": to_unix_ms("2023-01-01 10:00:00"),
        "timestamp_generated": to_unix_ms("2023-01-01 10:00:05"),
        "signal_type": "LONG", "side": "BUY", "pair": "BTCUSDC",
        "entry_price_target": 25000.0, "quantity_base_asset": 0.1, "confidence": 0.8,
        "strategy_name": "MyStrategy",
        "oco_order_params": {
            "take_profit_price": 26000.0, "stop_loss_price": 24000.0, "stop_loss_limit_price": 23950.0
        }
    }
    try:
        signal_obj = validate_signal_format(valid_signal_data)
        print(f"Valid TradingSignal: {signal_obj.pair}, TP: {signal_obj.oco_order_params.take_profit_price if signal_obj.oco_order_params else 'N/A'}") # type: ignore
    except ValueError as e:
        print(f"TradingSignal validation error: {e}")

    invalid_signal_data = valid_signal_data.copy()
    invalid_signal_data["pair"] = "BTCETH" # Invalid pair
    try:
        validate_signal_format(invalid_signal_data)
    except ValueError as e:
        print(f"TradingSignal validation error for pair (expected): {e}")


    # RiskParameters Validation
    risk_config = {
        "max_position_pct_balance": 0.1, "max_total_exposure_pct_balance": 0.5,
        "max_drawdown_pct_strategy": 0.2, "daily_loss_limit_pct_balance": 0.05,
        "position_notional_limits_quote": {"BTCUSDC": 10000.0, "ETHUSDC": 5000.0},
        "max_open_positions": 5
    }
    try:
        risk_params_obj = RiskParameters.model_validate(risk_config)
        print(f"Valid RiskParameters: Max BTCUSDC notional {risk_params_obj.position_notional_limits_quote.get('BTCUSDC')}")
    except ValidationError as e:
        print(f"RiskParameters validation error: {e}")

    # DataFrame Validation
    ohlcv_df_data = {
        'open': [10.0, 11.0, 10.5, 12.0], 'high': [12.0, 11.5, 10.0, 12.5], # Deliberate error in 3rd row H < L
        'low': [9.0, 10.5, 10.2, 11.5], 'close': [11.0, 10.5, 10.1, 12.2],
        'volume': [100.0, 150.0, 120.0, 200.0]
    }
    ohlcv_idx = pd.date_range("2023-01-01 09:00", periods=4, freq='15T', tz='UTC')
    df_ohlcv_invalid = pd.DataFrame(ohlcv_df_data, index=ohlcv_idx)
    
    try:
        validate_ohlcv_df_integrity(df_ohlcv_invalid)
    except ValueError as e:
        print(f"OHLCV DataFrame validation error (expected for H<L): {e}")
    
    df_ohlcv_valid = df_ohlcv_invalid.copy()
    df_ohlcv_valid.loc[df_ohlcv_valid.index[2], 'high'] = 10.8 # Fix error
    try:
        if validate_ohlcv_df_integrity(df_ohlcv_valid):
            print("Valid OHLCV DataFrame passed integrity check.")
    except ValueError as e:
        print(f"OHLCV DataFrame validation error (unexpected): {e}")


    # API Credentials
    try:
        validate_api_credentials("mykey", "")
    except ValueError as e:
        print(f"API credential validation error (expected): {e}")
    if validate_api_credentials("valid_api_key_string_example", "valid_api_secret_string_example"):
        print("API credentials format seems plausible.")

    # Timeframe
    try:
        validate_timeframe("10x")
    except ValueError as e:
        print(f"Timeframe validation error (expected): {e}")
    if validate_timeframe("15m") and validate_timeframe("1H"):
        print("Timeframes '15m' and '1H' are valid.")

    # OrderValidator (Limit Order Example)
    limit_order_data = {"symbol": "ETHUSDC", "side": "BUY", "quantity_base_asset": 0.5, "price": 2000.0}
    try:
        validated_limit_order = OrderValidator.validate_limit_order(limit_order_data)
        print(f"Validated LIMIT order: {validated_limit_order.symbol} Qty {validated_limit_order.quantity_base_asset}")
    except ValueError as e:
        print(f"LIMIT order validation error: {e}")

    # File Path Validation
    # Create dummy files/dirs for testing this part if run directly
    # For now, conceptual test:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # This should pass if this script is in a 'src/utils' like structure and 'src' is allowed
        # For a standalone test, let's use current_script_dir as allowed.
        allowed_dirs = [current_script_dir, os.path.join(current_script_dir, "..", "data")] # Example
        
        # Create a dummy file for testing must_exist
        dummy_file_path = os.path.join(current_script_dir, "dummy_test_file.txt")
        with open(dummy_file_path, "w") as f:
            f.write("test")

        valid_path = validate_file_path(dummy_file_path, allowed_base_dirs=[current_script_dir], must_exist=True, check_is_file=True)
        print(f"Validated file path: {valid_path}")
        
        os.remove(dummy_file_path) # Clean up

        # Test path traversal attempt (should fail if allowed_base_dirs is restrictive enough)
        # validate_file_path("../../../etc/passwd", allowed_base_dirs=[current_script_dir])
        # This depends on the OS and how restrictive allowed_base_dirs is.
        # A more direct test:
        # validate_file_path("/etc/passwd", allowed_base_dirs=["/safe/data/only"]) -> ValueError
        
    except ValueError as e:
        print(f"File path validation error: {e}")
    except Exception as e: # Catch other potential errors during test setup
        print(f"Error in file path test setup: {e}")


    # validate_inputs decorator example
    class SampleModel(BaseModel):
        x: int
        y: str = "default"

    @validate_inputs(data=SampleModel, another_arg=RiskParameters)
    def process_data(data: Dict, another_arg: Dict, non_validated_arg: bool = True):
        print(f"Processing validated data: {data}, {another_arg['max_open_positions']}, {non_validated_arg}")

    try:
        process_data(data={"x": 10, "y": "test"}, another_arg=risk_config) # Valid
        process_data(data={"x": "bad_int"}, another_arg=risk_config)    # Invalid data['x']
    except ValueError as e:
        print(f"validate_inputs decorator error (expected for bad_int): {e}")
    
    try:
        invalid_risk_for_decorator = risk_config.copy()
        invalid_risk_for_decorator["max_open_positions"] = -1 # Invalid
        process_data(data={"x": 10}, another_arg=invalid_risk_for_decorator)
    except ValueError as e:
        print(f"validate_inputs decorator error (expected for invalid risk): {e}")

