# src/data/data_manager.py
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any
import pandas as pd
import numpy as np
from loguru import logger
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from src.core.config import settings
from src.core.constants import Kline, StorageTypes, System
from src.core.exceptions import (
    DataError,
    DataDownloadError,
    StorageError,
    ConfigurationError
)
from src.data.binance_client import BinanceDataClient
from src.data.kline_processor import KlineProcessor
from src.data.storage.base import BaseStorage
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.postgres_storage import PostgresStorage
from src.data.storage.cache import CacheManager


class DownloadProgress:
    """Classe pour suivre la progression du téléchargement."""
    
    def __init__(self, total_pairs: int, total_days: int):
        self.total_pairs = total_pairs
        self.total_days = total_days
        self.pairs_completed = 0
        self.current_pair = ""
        self.current_pair_progress = 0
        self.current_pair_total = 0
        self.start_time = time.time()
        self.errors: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def update_pair(self, pair: str, klines_downloaded: int, klines_total: int):
        """Met à jour la progression pour une paire."""
        with self._lock:
            self.current_pair = pair
            self.current_pair_progress = klines_downloaded
            self.current_pair_total = klines_total
            
    def complete_pair(self, pair: str):
        """Marque une paire comme complétée."""
        with self._lock:
            self.pairs_completed += 1
            logger.success(f"Completed {pair} ({self.pairs_completed}/{self.total_pairs})")
            
    def add_error(self, pair: str, error: str):
        """Ajoute une erreur."""
        with self._lock:
            self.errors.append({
                "pair": pair,
                "error": error,
                "timestamp": datetime.now()
            })
            
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de progression."""
        with self._lock:
            elapsed = time.time() - self.start_time
            pairs_remaining = self.total_pairs - self.pairs_completed
            
            # Estimation du temps restant
            if self.pairs_completed > 0:
                avg_time_per_pair = elapsed / self.pairs_completed
                eta_seconds = avg_time_per_pair * pairs_remaining
            else:
                eta_seconds = 0
                
            return {
                "pairs_completed": self.pairs_completed,
                "total_pairs": self.total_pairs,
                "current_pair": self.current_pair,
                "current_pair_progress": self.current_pair_progress,
                "current_pair_total": self.current_pair_total,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
                "errors_count": len(self.errors),
                "progress_percentage": (self.pairs_completed / self.total_pairs * 100) if self.total_pairs > 0 else 0
            }


class DownloadCheckpoint:
    """Gère les checkpoints pour la reprise après échec."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "download_checkpoint.json"
        
    def save(self, state: Dict[str, Any]):
        """Sauvegarde l'état actuel."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug(f"Checkpoint saved: {state}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def load(self) -> Optional[Dict[str, Any]]:
        """Charge le dernier checkpoint."""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Checkpoint loaded: {state}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
            
    def clear(self):
        """Supprime le checkpoint."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")


class DataManager:
    """
    Gestionnaire principal des données.
    Coordonne le téléchargement, le traitement et le stockage des données.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de données."""
        # Configuration du stockage
        self.storage_type = settings.data.storage_type
        self.storage: Optional[BaseStorage] = None
        
        # Processeur de klines
        self.kline_processor = KlineProcessor()
        
        # Cache manager
        self.cache_manager = CacheManager() if settings.data.cache_enabled else None
        
        # Client Binance
        self.binance_client: Optional[BinanceDataClient] = None
        
        # Checkpoint pour la reprise
        self.checkpoint = DownloadCheckpoint(settings.data.storage_path / ".checkpoints")
        
        # Thread pool pour le traitement parallèle
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"DataManager initialized with {self.storage_type} storage")
        
    async def initialize(self):
        """Initialise les composants asynchrones."""
        # Initialiser le stockage
        if self.storage_type == StorageTypes.PARQUET:
            self.storage = ParquetStorage(settings.data.storage_path)
        elif self.storage_type == StorageTypes.POSTGRES:
            if not settings.database or not settings.database.url:
                raise ConfigurationError("PostgreSQL URL not configured")
            self.storage = PostgresStorage(settings.database.url)
        else:
            raise ConfigurationError(f"Unknown storage type: {self.storage_type}")
            
        await self.storage.initialize()
        
        # Initialiser le cache
        if self.cache_manager:
            await self.cache_manager.initialize()
            
        logger.success("DataManager initialized successfully")
        
    async def close(self):
        """Ferme proprement les connexions."""
        if self.storage:
            await self.storage.close()
        if self.cache_manager:
            await self.cache_manager.close()
        if self.binance_client:
            await self.binance_client.close()
        self.executor.shutdown(wait=True)
        logger.info("DataManager closed")
        
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
        
    async def download_historical_data(
        self,
        pairs: List[str],
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = Kline.INTERVAL_1MINUTE,
        resume: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Télécharge les données historiques pour plusieurs paires.
        
        Args:
            pairs: Liste des paires à télécharger
            start_date: Date de début
            end_date: Date de fin (défaut: maintenant)
            interval: Intervalle des klines (défaut: 1m)
            resume: Reprendre depuis le dernier checkpoint si disponible
            progress_callback: Callback pour suivre la progression
            
        Returns:
            Dictionnaire avec les statistiques du téléchargement
        """
        # Conversion des dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Timestamps en millisecondes pour Binance
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        # Charger le checkpoint si demandé
        checkpoint_state = None
        if resume:
            checkpoint_state = self.checkpoint.load()
            if checkpoint_state:
                # Vérifier que les paramètres correspondent
                if (checkpoint_state.get("start_ts") == start_ts and
                    checkpoint_state.get("end_ts") == end_ts and
                    checkpoint_state.get("interval") == interval and
                    set(checkpoint_state.get("pairs", [])) == set(pairs)):
                    logger.info("Resuming from checkpoint")
                else:
                    logger.warning("Checkpoint parameters don't match, starting fresh")
                    checkpoint_state = None
                    self.checkpoint.clear()
                    
        # Initialiser la progression
        total_days = (end_date - start_date).days
        progress = DownloadProgress(len(pairs), total_days)
        
        # Déterminer les paires à traiter
        if checkpoint_state:
            completed_pairs = set(checkpoint_state.get("completed_pairs", []))
            pairs_to_process = [p for p in pairs if p not in completed_pairs]
            progress.pairs_completed = len(completed_pairs)
        else:
            pairs_to_process = pairs
            
        # Initialiser le client Binance
        async with BinanceDataClient() as client:
            self.binance_client = client
            
            # Télécharger chaque paire
            for pair in pairs_to_process:
                try:
                    logger.info(f"Downloading {pair} from {start_date} to {end_date}")
                    
                    # Callback de progression pour cette paire
                    def pair_progress(downloaded, total):
                        progress.update_pair(pair, downloaded, total)
                        if progress_callback:
                            progress_callback(progress.get_stats())
                            
                    # Télécharger les klines
                    klines = await client.fetch_klines_batch(
                        symbol=pair,
                        interval=interval,
                        start_time=start_ts,
                        end_time=end_ts,
                        progress_callback=pair_progress
                    )
                    
                    if not klines:
                        logger.warning(f"No data downloaded for {pair}")
                        progress.add_error(pair, "No data available")
                        continue
                        
                    # Convertir en DataFrame
                    df = self._klines_to_dataframe(klines, pair)
                    
                    # Valider les données
                    validation_errors = self._validate_dataframe(df)
                    if validation_errors:
                        logger.error(f"Validation errors for {pair}: {validation_errors}")
                        progress.add_error(pair, f"Validation failed: {validation_errors}")
                        continue
                        
                    # Stocker les données
                    await self._store_klines(df, pair, interval)
                    
                    # Marquer comme complété
                    progress.complete_pair(pair)
                    
                    # Sauvegarder le checkpoint
                    if resume:
                        completed_pairs = [p for p in pairs if p in pairs[:pairs.index(pair)+1]]
                        self.checkpoint.save({
                            "pairs": pairs,
                            "completed_pairs": completed_pairs,
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "interval": interval,
                            "last_update": datetime.now()
                        })
                        
                except Exception as e:
                    logger.error(f"Error downloading {pair}: {e}")
                    progress.add_error(pair, str(e))
                    
        # Nettoyer le checkpoint si tout est terminé
        if progress.pairs_completed == len(pairs):
            self.checkpoint.clear()
            
        # Statistiques finales
        stats = progress.get_stats()
        stats["success"] = progress.pairs_completed == len(pairs)
        
        if stats["errors_count"] > 0:
            logger.warning(f"Download completed with {stats['errors_count']} errors")
            for error in progress.errors:
                logger.error(f"  - {error['pair']}: {error['error']}")
        else:
            logger.success("Download completed successfully")
            
        return stats
        
    def _klines_to_dataframe(self, klines: List[Dict], pair: str) -> pd.DataFrame:
        """Convertit les klines en DataFrame pandas."""
        df = pd.DataFrame(klines)
        
        # Renommer les colonnes selon notre convention
        df.rename(columns={
            'open_time': 'kline_open_time',
            'close_time': 'kline_close_time',
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price',
            'volume': 'base_asset_volume'
        }, inplace=True)
        
        # Ajouter la paire
        df['pair'] = pair
        
        # Convertir les timestamps en datetime
        df['kline_open_time'] = pd.to_datetime(df['kline_open_time'], unit='ms', utc=True)
        df['kline_close_time'] = pd.to_datetime(df['kline_close_time'], unit='ms', utc=True)
        
        # Marquer toutes les klines comme fermées (historique)
        df['is_kline_closed'] = True
        
        # Trier par temps
        df.sort_values('kline_open_time', inplace=True)
        
        # Définir l'index
        df.set_index('kline_open_time', inplace=True)
        
        return df
        
    def _validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """
        Valide un DataFrame de klines.
        
        Returns:
            Liste des erreurs trouvées
        """
        errors = []
        
        # Vérifier les colonnes requises
        required_columns = [
            'pair', 'kline_close_time', 'open_price', 'high_price',
            'low_price', 'close_price', 'base_asset_volume'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
            
        # Vérifier qu'il n'y a pas de valeurs manquantes
        if df[required_columns].isnull().any().any():
            errors.append("DataFrame contains null values")
            
        # Vérifier la cohérence OHLC
        invalid_ohlc = df[
            (df['high_price'] < df['low_price']) |
            (df['high_price'] < df['open_price']) |
            (df['high_price'] < df['close_price']) |
            (df['low_price'] > df['open_price']) |
            (df['low_price'] > df['close_price'])
        ]
        if not invalid_ohlc.empty:
            errors.append(f"Invalid OHLC data in {len(invalid_ohlc)} rows")
            
        # Vérifier les volumes négatifs
        if (df['base_asset_volume'] < 0).any():
            errors.append("Negative volumes found")
            
        # Vérifier l'ordre temporel
        if not df.index.is_monotonic_increasing:
            errors.append("Data is not sorted by time")
            
        # Vérifier les doublons
        duplicates = df.index.duplicated()
        if duplicates.any():
            errors.append(f"Found {duplicates.sum()} duplicate timestamps")
            
        return errors
        
    async def _store_klines(self, df: pd.DataFrame, pair: str, interval: str):
        """Stocke les klines dans le système de stockage configuré."""
        try:
            await self.storage.store_klines(df, pair, interval)
            logger.debug(f"Stored {len(df)} klines for {pair}")
            
            # Invalider le cache pour cette paire
            if self.cache_manager:
                await self.cache_manager.invalidate_pattern(f"klines:{pair}:*")
                
        except Exception as e:
            logger.error(f"Failed to store klines for {pair}: {e}")
            raise StorageError(f"Failed to store klines: {e}", original_exception=e)
            
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
            start_time: Début de la période
            end_time: Fin de la période
            limit: Nombre maximum de klines
            
        Returns:
            DataFrame avec les klines ou None si aucune donnée
        """
        # Vérifier le cache
        cache_key = None
        if self.cache_manager:
            cache_key = self.cache_manager.generate_key(
                "klines", pair, interval, 
                start_time.isoformat() if start_time else "none",
                end_time.isoformat() if end_time else "none",
                str(limit) if limit else "all"
            )
            
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
                
        # Récupérer depuis le stockage
        try:
            df = await self.storage.get_klines(
                pair=pair,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Mettre en cache si des données sont trouvées
            if df is not None and not df.empty and self.cache_manager and cache_key:
                await self.cache_manager.set(cache_key, df)
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to get klines for {pair}: {e}")
            raise DataError(f"Failed to retrieve klines: {e}", original_exception=e)
            
    async def get_available_pairs(self) -> List[str]:
        """Retourne la liste des paires disponibles dans le stockage."""
        try:
            return await self.storage.get_available_pairs()
        except Exception as e:
            logger.error(f"Failed to get available pairs: {e}")
            raise DataError("Failed to get available pairs", original_exception=e)
            
    async def get_data_range(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Optional[Tuple[datetime, datetime]]:
        """
        Retourne la plage de dates disponible pour une paire.
        
        Returns:
            Tuple (start_date, end_date) ou None si aucune donnée
        """
        try:
            return await self.storage.get_data_range(pair, interval)
        except Exception as e:
            logger.error(f"Failed to get data range for {pair}: {e}")
            raise DataError("Failed to get data range", original_exception=e)
            
    async def update_latest_data(
        self,
        pairs: Optional[List[str]] = None,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> Dict[str, Any]:
        """
        Met à jour les données avec les dernières klines disponibles.
        
        Args:
            pairs: Liste des paires à mettre à jour (défaut: toutes)
            interval: Intervalle des klines
            
        Returns:
            Statistiques de la mise à jour
        """
        stats = {
            "pairs_updated": 0,
            "klines_added": 0,
            "errors": []
        }
        
        # Obtenir les paires à mettre à jour
        if pairs is None:
            pairs = await self.get_available_pairs()
            
        async with BinanceDataClient() as client:
            for pair in pairs:
                try:
                    # Obtenir la dernière date dans le stockage
                    data_range = await self.get_data_range(pair, interval)
                    if not data_range:
                        logger.warning(f"No existing data for {pair}, skipping update")
                        continue
                        
                    last_date = data_range[1]
                    
                    # Télécharger depuis la dernière date
                    start_ts = int(last_date.timestamp() * 1000) + 1  # +1ms pour éviter les doublons
                    
                    klines = await client.fetch_klines_batch(
                        symbol=pair,
                        interval=interval,
                        start_time=start_ts
                    )
                    
                    if klines:
                        df = self._klines_to_dataframe(klines, pair)
                        await self._store_klines(df, pair, interval)
                        
                        stats["pairs_updated"] += 1
                        stats["klines_added"] += len(klines)
                        logger.info(f"Updated {pair} with {len(klines)} new klines")
                    else:
                        logger.debug(f"No new data for {pair}")
                        
                except Exception as e:
                    logger.error(f"Failed to update {pair}: {e}")
                    stats["errors"].append({"pair": pair, "error": str(e)})
                    
        logger.info(f"Update completed: {stats['pairs_updated']} pairs, {stats['klines_added']} klines")
        return stats
        
    async def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """
        Nettoie les données plus anciennes que le nombre de jours spécifié.
        
        Args:
            days_to_keep: Nombre de jours de données à conserver
            
        Returns:
            Nombre de lignes supprimées par paire
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        logger.info(f"Cleaning up data older than {cutoff_date}")
        
        try:
            deleted_counts = await self.storage.delete_old_data(cutoff_date)
            
            # Invalider tout le cache après nettoyage
            if self.cache_manager:
                await self.cache_manager.clear()
                
            total_deleted = sum(deleted_counts.values())
            logger.info(f"Cleanup completed: {total_deleted} klines deleted")
            
            return deleted_counts
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise DataError("Failed to cleanup old data", original_exception=e)


# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du DataManager."""
    
    async with DataManager() as dm:
        # Télécharger des données historiques
        def progress_callback(stats):
            print(f"Progress: {stats['progress_percentage']:.1f}% - "
                  f"Current: {stats['current_pair']} "
                  f"({stats['current_pair_progress']}/{stats['current_pair_total']})")
            
        stats = await dm.download_historical_data(
            pairs=["BTCUSDC", "ETHUSDC"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            progress_callback=progress_callback
        )
        
        print(f"Download stats: {stats}")
        
        # Récupérer des données
        df = await dm.get_klines(
            pair="BTCUSDC",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2)
        )
        
        if df is not None:
            print(f"Retrieved {len(df)} klines")
            print(df.head())
            
        # Mettre à jour avec les dernières données
        update_stats = await dm.update_latest_data(pairs=["BTCUSDC"])
        print(f"Update stats: {update_stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())