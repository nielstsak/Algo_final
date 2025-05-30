# src/data/kline_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger

from src.core.constants import Kline
from src.core.exceptions import DataError, KlineValidationError


class KlineProcessor:
    """
    Processeur pour la manipulation et le resampling des klines.
    Gère la génération de klines multi-fréquences (standard et glissantes).
    """
    
    def __init__(self):
        """Initialise le processeur de klines."""
        # Mapping des intervalles Binance vers les fréquences pandas
        self.interval_mapping = {
            '1m': '1T',
            '3m': '3T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '8h': '8H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W',
            '1M': '1M'
        }
        
        # Règles d'agrégation standard
        self.aggregation_rules = Kline.AGGREGATION_RULES_STANDARD.copy()
        
        logger.debug("KlineProcessor initialized")
        
    def resample_klines(
        self,
        df: pd.DataFrame,
        target_freq: str,
        rolling: bool = False,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Génère des klines de fréquence supérieure à partir de klines 1m.
        
        Args:
            df: DataFrame avec klines source (doit avoir un index temporel)
            target_freq: Fréquence cible ('3m', '5m', '15m', '1h', etc.)
            rolling: Si True, génère des klines glissantes
            window_size: Taille de fenêtre pour les klines glissantes (si None, déduit de target_freq)
            
        Returns:
            DataFrame avec klines resamplées
            
        Raises:
            Data