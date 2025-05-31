# src/data/storage/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.core.constants import Kline


class BaseStorage(ABC):
    """
    Interface abstraite pour les systèmes de stockage de klines.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialise le système de stockage."""
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Ferme proprement les connexions."""
        pass
        
    @abstractmethod
    async def store_klines(
        self,
        df: pd.DataFrame,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> None:
        """
        Stocke les klines dans le système de stockage.
        
        Args:
            df: DataFrame contenant les klines
            pair: Paire de trading
            interval: Intervalle des klines
        """
        pass
        
    @abstractmethod
    async def get_klines(
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les klines depuis le stockage.
        
        Args:
            pair: Paire de trading
            interval: Intervalle des klines
            start_time: Date de début (incluse)
            end_time: Date de fin (incluse)
            limit: Nombre maximum de klines à retourner
            
        Returns:
            DataFrame avec les klines ou None si aucune donnée
        """
        pass
        
    @abstractmethod
    async def get_available_pairs(self) -> List[str]:
        """
        Retourne la liste des paires disponibles dans le stockage.
        
        Returns:
            Liste des paires
        """
        pass
        
    @abstractmethod
    async def get_data_range(
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Retourne la plage de dates disponible pour une paire.
        
        Args:
            pair: Paire de trading
            interval: Intervalle des klines
            
        Returns:
            Tuple (start_date, end_date) ou None si aucune donnée
        """
        pass
        
    @abstractmethod
    async def delete_old_data(
        self,
        cutoff_date: datetime
    ) -> Dict[str, int]:
        """
        Supprime les données antérieures à une date donnée.
        
        Args:
            cutoff_date: Date limite (les données avant cette date sont supprimées)
            
        Returns:
            Dictionnaire {pair: nombre_de_lignes_supprimées}
        """
        pass
        
    async def get_statistics(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Dict[str, Any]:
        """
        Retourne des statistiques sur les données stockées pour une paire.
        
        Args:
            pair: Paire de trading
            interval: Intervalle des klines
            
        Returns:
            Dictionnaire avec les statistiques
        """
        data_range = await self.get_data_range(pair, interval)
        if not data_range:
            return {"exists": False}
            
        df = await self.get_klines(pair, interval, limit=1)
        if df is None or df.empty:
            return {"exists": False}
            
        return {
            "exists": True,
            "start_date": data_range[0],
            "end_date": data_range[1],
            "duration_days": (data_range[1] - data_range[0]).days,
            "sample_count": len(df) if df is not None else 0
        }