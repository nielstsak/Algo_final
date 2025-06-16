from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field
import numpy as np

from src.strategies.base_strategy import BaseStrategy
from src.core.exceptions import SignalGenerationError
from src.strategies.params import BaseFixedParams, BaseOptimizableParams
from src.data.enriched_dataframe import EnrichedDataFrame
from src.strategies import technical_indicators as ti

class PsarReversalOtocoFixedParams(BaseFixedParams):
    indicator_frequency: str = Field(default='1h', description="Fréquence pour le calcul des indicateurs.")

class PsarReversalOtocoOptimizableParams(BaseOptimizableParams):
    psar_inc: float = Field(default=0.02, gt=0, description="Incrément pour le Parabolic SAR.")
    psar_max: float = Field(default=0.2, gt=0, description="Valeur maximale pour le Parabolic SAR.")
    atr_period: int = Field(default=14, gt=0, description="Période de l'ATR pour le calcul du SL/TP.")
    atr_multiplier_sl: float = Field(default=2.0, gt=0, description="Multiplicateur de l'ATR pour le Stop Loss.")
    atr_multiplier_tp: float = Field(default=3.0, gt=0, description="Multiplicateur de l'ATR pour le Take Profit.")
    otoco_candle_body_ratio_thld: float = Field(default=0.7, ge=0, le=1, description="Seuil du ratio corps/range pour le pattern OTOCO.")
    otoco_wick_ratio_thld: float = Field(default=0.1, ge=0, le=1, description="Seuil du ratio mèche/range pour le pattern OTOCO.")
    position_sizing_pct_capital: float = Field(default=0.01, gt=0, lt=1, description="Pourcentage du capital à risquer par trade.")

class PsarReversalOtocoStrategy(BaseStrategy):
    name: str = "PsarReversalOtocoStrategy"
    version: str = "3.6.0" # Version mise à jour
    description: str = "Combine les retournements de PSAR avec le pattern OTOCO pour confirmation."

    fixed_params_model = PsarReversalOtocoFixedParams
    optimizable_params_model = PsarReversalOtocoOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"
        super().__init__(params, **kwargs)
        
        self.required_timeframes = [self.get_param('indicator_frequency')]
        self.min_required_periods = self.get_param('atr_period') + 2
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")
    
    def calculate_indicators(self, data: Union[Dict[str, pd.DataFrame], EnrichedDataFrame]) -> pd.DataFrame:
        log_pref = self.strategy_name_log_prefix
        df = self._prepare_indicator_data(data)
        freq = self.get_param('indicator_frequency')
        
        # Détermination des colonnes sources
        close_col = f"K_{freq}_close" if f"K_{freq}_close" in df.columns else "close"
        high_col = f"K_{freq}_high" if f"K_{freq}_high" in df.columns else "high"
        low_col = f"K_{freq}_low" if f"K_{freq}_low" in df.columns else "low"
        open_col = f"K_{freq}_open" if f"K_{freq}_open" in df.columns else "open"

        required_cols = [open_col, high_col, low_col, close_col]
        if not all(c in df.columns for c in required_cols):
            raise SignalGenerationError(f"Colonnes OHLC manquantes. Attendu: {required_cols}", self.name)
        
        # Création d'un DataFrame temporaire avec les noms de colonnes standardisés
        temp_df = df[required_cols].rename(columns={
            open_col: 'open', high_col: 'high', low_col: 'low', close_col: 'close'
        })

        # Calcul des indicateurs
        psar_df = ti.calculate_psar(temp_df['high'], temp_df['low'], temp_df['close'], 
                                    acceleration=self.get_param('psar_inc'), 
                                    maximum=self.get_param('psar_max'))
        # Renomme la colonne PSAR pour la simplicité (pandas-ta peut retourner plusieurs colonnes)
        psar_col_name = next((col for col in psar_df.columns if 'psarl' in col.lower()), 'psar')
        psar_df.rename(columns={psar_col_name: 'psar'}, inplace=True)

        atr = ti.calculate_atr(temp_df['high'], temp_df['low'], temp_df['close'], period=self.get_param('atr_period'))
        
        otoco = ti.calculate_otoco_pattern(temp_df, 
                                           self.get_param('otoco_candle_body_ratio_thld'), 
                                           self.get_param('otoco_wick_ratio_thld'))

        # Concaténation et jointure
        indicators_df = pd.concat([psar_df['psar'], atr, otoco], axis=1)
        final_df = df.join(indicators_df)

        if 'close' not in final_df.columns:
            final_df['close'] = final_df[close_col]

        self._indicators_cache = final_df
        logger.info(f"{log_pref} Indicateurs calculés.")
        return final_df

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        df = indicators_df.copy()
            
        required_cols = ['close', 'psar', 'otoco_long', 'otoco_short', 'atr']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise SignalGenerationError(f"Colonnes requises manquantes: {missing}", strategy_name=self.name)
        
        df['psar_reversal_up'] = (df['close'] > df['psar']) & (df['close'].shift(1) < df['psar'].shift(1))
        df['psar_reversal_down'] = (df['close'] < df['psar']) & (df['close'].shift(1) > df['psar'].shift(1))

        df['entry_long'] = df['psar_reversal_up'] & df['otoco_long']
        df['entry_short'] = df['psar_reversal_down'] & df['otoco_short']
        
        df['exit_long'] = df['psar_reversal_down']
        df['exit_short'] = df['psar_reversal_up']

        sl_mult = self.get_param('atr_multiplier_sl')
        tp_mult = self.get_param('atr_multiplier_tp')

        df['sl'] = np.nan
        df['tp'] = np.nan

        long_entries = df['entry_long']
        df.loc[long_entries, 'sl'] = df['close'] - (df['atr'] * sl_mult)
        df.loc[long_entries, 'tp'] = df['close'] + (df['atr'] * tp_mult)

        short_entries = df['entry_short']
        df.loc[short_entries, 'sl'] = df['close'] + (df['atr'] * sl_mult)
        df.loc[short_entries, 'tp'] = df['close'] - (df['atr'] * tp_mult)
        
        # Renomme les colonnes de signaux pour la compatibilité avec vectorbt
        final_signals = df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']]
        
        self._signals = final_signals
        return self._signals.copy()
