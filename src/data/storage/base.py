# src/data/storage/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.core.constants import Kline


class BaseStorage(ABC):
    """
    Interface abstraite définissant le contrat pour les systèmes de stockage.
    Toute classe de stockage (ex: Parquet, PostgreSQL) doit hériter de BaseStorage
    et implémenter toutes ses méthodes abstraites.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialise le système de stockage, comme la création de répertoires
        ou l'établissement de connexions à la base de données.
        """
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Ferme proprement les connexions et libère les ressources."""
        pass
        
    @abstractmethod
    async def store_klines(
        self,
        df: pd.DataFrame,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> None:
        """
        Stocke un DataFrame de klines.
        
        Args:
            df: DataFrame contenant les données klines à stocker.
            pair: La paire de trading (ex: "BTCUSDT").
            interval: L'intervalle des klines (ex: "1m").
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
        Récupère les klines depuis le système de stockage.
        
        Args:
            pair: La paire de trading.
            interval: L'intervalle des klines.
            start_time: Date de début (inclusive, UTC).
            end_time: Date de fin (inclusive, UTC).
            limit: Nombre maximum de klines à retourner depuis la fin de la période.
            
        Returns:
            Un DataFrame contenant les klines, ou None si aucune donnée n'est trouvée.
        """
        pass
        
    @abstractmethod
    async def get_available_pairs(self) -> List[str]:
        """
        Retourne la liste unique et triée des paires de trading disponibles dans le stockage.
        
        Returns:
            Une liste de strings représentant les paires.
        """
        pass
        
    @abstractmethod
    async def get_data_range(
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Retourne la plage de dates (premier et dernier timestamp) disponible pour une paire et un intervalle.
        
        Args:
            pair: La paire de trading.
            interval: L'intervalle des klines.
            
        Returns:
            Un tuple (start_date, end_date) en UTC, ou None si aucune donnée n'est trouvée.
        """
        pass
        
    @abstractmethod
    async def delete_old_data(
        self,
        cutoff_date: datetime
    ) -> Dict[str, int]:
        """
        Supprime toutes les données antérieures à une date de coupure spécifiée.
        
        Args:
            cutoff_date: La date limite (UTC). Les données avant cette date seront supprimées.
            
        Returns:
            Un dictionnaire résumant le nombre de lignes supprimées par table/paire.
        """
        pass
        
    @abstractmethod
    async def get_statistics(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Dict[str, Any]:
        """
        Calcule et retourne des statistiques de base sur les données stockées pour une paire.
        
        Args:
            pair: La paire de trading.
            interval: L'intervalle des klines.
            
        Returns:
            Un dictionnaire de statistiques (ex: nombre de lignes, dates, etc.).
        """
        pass

    @abstractmethod
    async def optimize_storage(self) -> Dict[str, Any]:
        """
        Effectue des opérations de maintenance pour optimiser le stockage,
        comme la fusion de petits fichiers ou le VACUUM d'une table de base de données.

        Returns:
            Un dictionnaire résumant les résultats de l'optimisation.
        """
        pass
