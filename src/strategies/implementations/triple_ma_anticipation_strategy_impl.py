from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, model_validator

from src.strategies.base_strategy import BaseStrategy
from src.core.exceptions import SignalGenerationError
from src.strategies.params import BaseFixedParams, BaseOptimizableParams
from src.data.enriched_dataframe import EnrichedDataFrame
from src.strategies import technical_indicators as ti

class TripleMAAnticipationFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class TripleMAAnticipationOptimizableParams(BaseOptimizableParams):
    fast_period: int = Field(default=10, gt=0, description="Période de la SMA rapide.")
    medium_period: int = Field(default=20, gt=0, description="Période de la SMA moyenne.")
    slow_period: int = Field(default=50, gt=0, description="Période de la SMA lente.")
    stop_loss_pct: float = Field(default=0.03, gt=0, lt=1, description="Pourcentage de Stop Loss.")
    take_profit_pct: float = Field(default=0.06, gt=0, lt=1, description="Pourcentage de Take Profit.")
    position_sizing_pct_capital: float = Field(default=0.01, gt=0, lt=1, description="Pourcentage du capital à risquer.")
    anticipation_threshold_pct: float = Field(default=0.001, ge=0, description="Seuil d'anticipation du croisement (en % des prix).")

    @model_validator(mode='after')
    def check_periods_logic(self) -> 'TripleMAAnticipationOptimizableParams':
        if not (self.fast_period < self.medium_period < self.slow_period):
            raise ValueError("Les périodes des SMA doivent être dans l'ordre croissant : fast < medium < slow.")
        return self

class TripleMAAnticipationStrategy(BaseStrategy):
    name: str = "TripleMAAnticipationStrategy"
    version: str = "1.6.0" # Version mise à jour
    description: str = "Stratégie de triple SMA qui anticipe les croisements, refactorisée."

    fixed_params_model = TripleMAAnticipationFixedParams
    optimizable_params_model = TripleMAAnticipationOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)

        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = self.get_param('slow_period') + 2
        
        self.fast_sma_col = f"SMA_{self.get_param('fast_period')}"
        self.medium_sma_col = f"SMA_{self.get_param('medium_period')}"
        self.slow_sma_col = f"SMA_{self.get_param('slow_period')}"

        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        df = self._prepare_indicator_data(data)
        freq = self.get_param('indicator_frequency')
        close_col = f"K_{freq}_close" if f"K_{freq}_close" in df.columns else "close"
        
        if close_col not in df.columns:
            raise SignalGenerationError(f"Colonne source '{close_col}' manquante.", self.name)

        fast_sma = ti.calculate_sma(df[close_col], period=self.get_param('fast_period'))
        medium_sma = ti.calculate_sma(df[close_col], period=self.get_param('medium_period'))
        slow_sma = ti.calculate_sma(df[close_col], period=self.get_param('slow_period'))

        fast_sma.name = self.fast_sma_col
        medium_sma.name = self.medium_sma_col
        slow_sma.name = self.slow_sma_col

        indicators_df = pd.concat([fast_sma, medium_sma, slow_sma], axis=1)
        final_df = df.join(indicators_df)

        if 'close' not in final_df.columns:
            final_df['close'] = final_df[close_col]

        self._indicators_cache = final_df
        logger.info(f"{log_pref} Indicateurs calculés.")
        return final_df

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        if indicators_df.empty: return pd.DataFrame()
        df = indicators_df.copy()
            
        required_cols = [self.fast_sma_col, self.medium_sma_col, self.slow_sma_col, 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise SignalGenerationError(f"Colonnes requises manquantes: {missing}", strategy_name=self.name)

        threshold = self.get_param('anticipation_threshold_pct')

        golden_cross_confirmed = (df[self.fast_sma_col] > df[self.medium_sma_col]) & (df[self.medium_sma_col] > df[self.slow_sma_col])
        anticipation_long = df[self.fast_sma_col] > (df[self.medium_sma_col] * (1 - threshold))
        entry_long = golden_cross_confirmed & anticipation_long

        death_cross_confirmed = (df[self.fast_sma_col] < df[self.medium_sma_col]) & (df[self.medium_sma_col] < df[self.slow_sma_col])
        anticipation_short = df[self.fast_sma_col] < (df[self.medium_sma_col] * (1 + threshold))
        entry_short = death_cross_confirmed & anticipation_short
        
        exit_long = (df[self.fast_sma_col] < df[self.medium_sma_col]) & (df[self.fast_sma_col].shift(1) >= df[self.medium_sma_col].shift(1))
        exit_short = (df[self.fast_sma_col] > df[self.medium_sma_col]) & (df[self.fast_sma_col].shift(1) <= df[self.medium_sma_col].shift(1))

        signals = pd.DataFrame(index=df.index)
        signals['entry_long'] = entry_long
        signals['entry_short'] = entry_short
        signals['exit_long'] = exit_long
        signals['exit_short'] = exit_short

        self._signals = signals
        return self._signals.copy()
