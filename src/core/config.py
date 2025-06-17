# src/core/config.py

"""
Provides a centralized, type-safe configuration system for the application.

This module uses Pydantic to define a hierarchical configuration structure,
allowing for easy validation, default value management, and loading from
external files (e.g., YAML). It serves as the single source of truth for
all configurable parameters across the framework.
"""

from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field, validator

from src.core.exceptions import ConfigurationError


# --- Sub-models for configuration sections ---

class DataConfig(BaseModel):
    """Configuration for data sources and storage."""
    storage_path: str = "data/crypto"
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "1h"


class FeeConfig(BaseModel):
    """Configuration for transaction fee models."""
    model: str = "percentage"  # e.g., "percentage", "none"
    percentage: float = 0.1

    @validator('percentage')
    def fee_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Fee percentage cannot be negative")
        return v


class SlippageConfig(BaseModel):
    """Configuration for slippage models."""
    model: str = "percentage"  # e.g., "percentage", "none"
    percentage: float = 0.05

    @validator('percentage')
    def slippage_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Slippage percentage cannot be negative")
        return v


class BacktestConfig(BaseModel):
    """Configuration for the backtesting engine."""
    initial_cash: float = 100_000.0
    fee: FeeConfig = Field(default_factory=FeeConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)


class WFOConfig(BaseModel):
    """Configuration for Walk-Forward Optimization."""
    train_size: int = 2000
    test_size: int = 500
    n_splits: int = 10
    step_size: Optional[int] = None


class OptimizerConfig(BaseModel):
    """Configuration for the optimization algorithm (e.g., Optuna)."""
    direction: str = "maximize"
    storage: Optional[str] = "sqlite:///optimization_study.db"


class OptimizationConfig(BaseModel):
    """Top-level configuration for an optimization run."""
    metric_to_optimize: str = "sharpe_ratio"
    n_trials: int = 100
    wfo: WFOConfig = Field(default_factory=WFOConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


# --- Top-level Application Configuration ---

class AppConfig(BaseModel):
    """
    The main configuration model for the entire trading framework.
    """
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


# --- Loading function ---

def load_config(path: str) -> AppConfig:
    """
    Loads a YAML configuration file from the given path and parses it
    into a validated AppConfig object.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        AppConfig: A validated application configuration object.

    Raises:
        ConfigurationError: If the file cannot be found, parsed, or validated.
    """
    try:
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        if raw_config is None:
            return AppConfig() # Return default config if file is empty

        return AppConfig(**raw_config)

    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found at: {path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        # Catches Pydantic's ValidationError and others
        raise ConfigurationError(f"Configuration validation failed: {e}")
