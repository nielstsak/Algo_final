# src/optimization/config.py
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

class WFOConfig(BaseModel):
    """Configuration for Walk-Forward Optimization."""
    is_days: int = Field(..., gt=0, description="Duration in days of the In-Sample optimization period.")
    oos_days: int = Field(..., gt=0, description="Duration in days of the Out-of-Sample validation period.")
    gap_days: int = Field(default=0, ge=0, description="Duration in days of the quarantine period between IS and OOS.")
    n_splits: int = Field(..., gt=0, description="Number of Walk-Forward folds (splits) to generate.")
    expanding_window: bool = Field(default=False, description="True for an expanding window, False for a sliding window.")

class OptimizationConfig(BaseModel):
    """Defines the parameters for the optimization session itself."""
    method: Literal['simple', 'wfo'] = Field(..., description="Optimization method: 'simple' or 'wfo' (Walk-Forward Optimization).")
    metric_to_optimize: str = Field(default='sharpe_ratio', description="The performance metric to optimize (e.g., 'sharpe_ratio', 'profit_factor').")
    n_trials: int = Field(default=100, gt=0, description="Number of optimization trials to run.")
    sampler: str = Field(default='tpe', description="Optuna sampler algorithm (e.g., 'tpe', 'cmaes', 'random').")
    pruner: str = Field(default='median', description="Optuna pruner algorithm (e.g., 'median', 'hyperband').")
    wfo_config: Optional[WFOConfig] = Field(default=None, description="Configuration for WFO method. Required if method is 'wfo'.")

    @validator('wfo_config', always=True)
    def check_wfo_config(cls, v, values):
        """
        Validates that wfo_config is provided when the optimization method is 'wfo'.
        """
        if values.get('method') == 'wfo' and v is None:
            raise ValueError("wfo_config is required when method is 'wfo'.")
        return v
