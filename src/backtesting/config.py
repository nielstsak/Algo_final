# src/backtesting/config.py
from pydantic import BaseModel, Field

class BacktestConfig(BaseModel):
    """
    Defines the backtest environment settings, 
    which remain constant during an optimization session.
    """
    initial_capital: float = Field(default=10000.0, gt=0, description="Initial capital for the backtest.")
    fees: float = Field(default=0.001, ge=0, description="Trading fees/commission rate per trade.")
    slippage: float = Field(default=0.0005, ge=0, description="Slippage rate per trade.")
    leverage: float = Field(default=1.0, gt=0, description="Leverage to apply.")
