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
from src.strategies.registry import StrategyRegistry

class SMACrossFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class SMACrossOptimizableParams(BaseOptimizableParams):
    fast_period: int = Field(default=10, gt=0, description="Période de la SMA rapide.")
    slow_period: int = Field(default=30, gt=0, description="Période de la SMA lente.")
    stop_loss_pct: float = Field(default=0.02, gt=0, lt=1, description="Pourcentage de Stop Loss.")
    take_profit_pct: float = Field(default=0.04, gt=0, lt=1, description="Pourcentage de Take Profit.")
    position_sizing_pct_capital: float = Field(default=0.02, gt=0, lt=1, description="Pourcentage du capital à risquer par trade.")

    @model_validator(mode='after')
    def check_periods_logic(self) -> 'SMACrossOptimizableParams':
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period doit être inférieur à slow_period.")
        return self

@StrategyRegistry.register("SMACrossStrategy")
class SMACrossStrategy(BaseStrategy):
    name: str = "SMACrossStrategy"
    version: str = "3.5.0" # Version mise à jour
    description: str = "Stratégie de croisement de SMA, refactorisée pour utiliser les fonctions d'indicateurs."

    fixed_params_model = SMACrossFixedParams
    optimizable_params_model = SMACrossOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)
        
        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = self.get_param('slow_period') + 2
        
        self.fast_sma_col = f"SMA_{self.get_param('fast_period')}"
        self.slow_sma_col = f"SMA_{self.get_param('slow_period')}"
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")

    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        df = self._prepare_indicator_data(data)
        freq = self.get_param('indicator_frequency')
        close_col = f"K_{freq}_close" if f"K_{freq}_close" in df.columns else "close"

        if close_col not in df.columns:
            raise SignalGenerationError(f"Colonne source '{close_col}' manquante.", self.name)
        
        # Calcul des indicateurs via les fonctions
        fast_sma = ti.calculate_sma(df[close_col], period=self.get_param('fast_period'))
        slow_sma = ti.calculate_sma(df[close_col], period=self.get_param('slow_period'))
        
        # Attribution des noms de colonnes
        fast_sma.name = self.fast_sma_col
        slow_sma.name = self.slow_sma_col

        # Concaténation et jointure
        indicators_df = pd.concat([fast_sma, slow_sma], axis=1)
        final_df = df.join(indicators_df)

        if 'close' not in final_df.columns:
            final_df['close'] = final_df[close_col]

        self._indicators_cache = final_df
        logger.info(f"{log_pref} Indicateurs calculés avec succès.")
        return final_df

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        if indicators_df.empty: return pd.DataFrame()
        df = indicators_df.copy()
        
        required_cols = [self.fast_sma_col, self.slow_sma_col, 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise SignalGenerationError(f"Colonnes requises manquantes: {missing}", strategy_name=self.name)

        fast_sma = df[self.fast_sma_col]
        slow_sma = df[self.slow_sma_col]

        entry_long = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        entry_short = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        
        signals = pd.DataFrame(index=df.index)
        signals['entry_long'] = entry_long
        signals['entry_short'] = entry_short
        signals['exit_long'] = entry_short.copy()
        signals['exit_short'] = entry_long.copy()
        
        sl_pct = self.get_param('stop_loss_pct')
        tp_pct = self.get_param('take_profit_pct')

        signals['sl'] = np.where(entry_long, df['close'] * (1 - sl_pct), np.where(entry_short, df['close'] * (1 + sl_pct), np.nan))
        signals['tp'] = np.where(entry_long, df['close'] * (1 + tp_pct), np.where(entry_short, df['close'] * (1 - tp_pct), np.nan))
        
        self._signals = signals
        return self._signals.copy()
