# src/strategies/params.py
from pydantic import BaseModel, Field

class BaseFixedParams(BaseModel):
    """
    Base model for strategy parameters that remain fixed during an optimization session.
    """
    indicator_frequency: str = Field(default='1h', description="The main indicator frequency for the strategy.")

class BaseOptimizableParams(BaseModel):
    """
    Base model for strategy parameters that will be optimized.
    Fields in child classes will define the search space using Pydantic's Field annotations.
    """
    pass
