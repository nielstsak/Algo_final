# src/data/storage/parquet_storage.py
import asyncio
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from src.core.config import settings
from src.core.constants import Kline, System
from src.core.exceptions import ParquetStorageError, StorageError
from src.data.storage.base import BaseStorage


class ParquetStorage(BaseStorage):
    """
    Implémentation du stockage en fichiers Parquet.
    Utilise un fichier unique par paire/intervalle pour la simplicité et la performance
    sur des datasets de taille modérée.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        thread_pool_storage_workers = getattr(settings.system, 'thread_pool_workers_storage', System.THREAD_POOL_WORKERS_STORAGE_DEFAULT)
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_storage_workers)
        self._initialized = False
        logger.info(f"ParquetStorage (single-file mode) initialized with path: {self.storage_path}")

    async def initialize(self) -> None:
        """S'assure que le répertoire de stockage de base existe."""
        if self._initialized:
            return
        try:
            # Wrapper la création du dossier dans une fonction pour une exécution propre
            def _create_dir_if_not_exists(path: Path):
                path.mkdir(parents=True, exist_ok=True)

            await asyncio.get_event_loop().run_in_executor(
                self.executor, _create_dir_if_not_exists, self.storage_path
            )
            self._initialized = True
            logger.success(f"ParquetStorage initialized at {self.storage_path}")
        except Exception as e:
            raise StorageError(f"Failed to initialize ParquetStorage: {e}", original_exception=e)

    async def close(self) -> None:
        """Ferme le pool de threads."""
        self.executor.shutdown(wait=True)
        logger.info("ParquetStorage closed")

    def _validate_initialization(self):
        if not self._initialized:
            raise StorageError("ParquetStorage not initialized. Call initialize() first.")

    def _get_file_path(self, pair: str, interval: str) -> Path:
        """Retourne le chemin vers le fichier parquet unique pour une paire/intervalle."""
        safe_interval = interval.replace('/', '_')
        return self.storage_path / f"{pair.upper()}_{safe_interval}.parquet"

    def _prepare_dataframe_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare le DataFrame avant de le sauvegarder, en s'assurant que l'index est une colonne 'timestamp'."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ParquetStorageError("DataFrame must have a DatetimeIndex to be stored.")
        df_copy = df.copy()
        if 'timestamp' not in df_copy.columns:
            df_copy.reset_index(inplace=True)
        return df_copy

    def _read_parquet_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Lit un fichier Parquet unique et le retourne en tant que DataFrame avec un DatetimeIndex."""
        if not file_path.exists():
            return None
        try:
            table = pq.read_table(file_path)
            if table.num_rows == 0:
                return pd.DataFrame()
            df = table.to_pandas(date_as_object=False, self_destruct=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error reading parquet file {file_path}: {e}", exc_info=True)
            raise ParquetStorageError(f"Failed to read parquet file: {file_path}", original_exception=e)

    async def store_klines(self, df: pd.DataFrame, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> None:
        """Stocke un DataFrame dans un fichier Parquet unique, en fusionnant avec les données existantes."""
        self._validate_initialization()
        if df.empty:
            return

        file_path = self._get_file_path(pair, interval)
        df_to_store = df.copy() # Travailler sur une copie

        def _read_merge_write():
            existing_df = self._read_parquet_file(file_path)

            if existing_df is not None and not existing_df.empty:
                combined_df = pd.concat([existing_df, df_to_store])
                final_df = combined_df[~combined_df.index.duplicated(keep='last')]
                final_df.sort_index(inplace=True)
            else:
                final_df = df_to_store.sort_index()

            if final_df.empty:
                return 0
            
            final_df_prepared = self._prepare_dataframe_for_storage(final_df)

            arrow_table = pa.Table.from_pandas(final_df_prepared, preserve_index=False)
            pq.write_table(arrow_table, file_path, compression='snappy')
            return len(df_to_store)

        await asyncio.get_event_loop().run_in_executor(self.executor, _read_merge_write)
        logger.info(f"Stored/Updated {len(df_to_store)} klines for {pair} at {interval} in single file: {file_path.name}")

    async def get_klines(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Récupère les klines à partir du fichier Parquet unique et applique les filtres."""
        self._validate_initialization()
        file_path = self._get_file_path(pair, interval)

        def _read_and_filter():
            df = self._read_parquet_file(file_path)
            if df is None or df.empty:
                return None

            if start_time: df = df[df.index >= start_time]
            if end_time: df = df[df.index <= end_time]
            if limit: df = df.tail(limit)

            return df if not df.empty else None

        return await asyncio.get_event_loop().run_in_executor(self.executor, _read_and_filter)

    async def get_available_pairs(self) -> List[str]:
        """Liste les paires disponibles en scannant les noms de fichiers."""
        self._validate_initialization()
        if not self.storage_path.exists(): return []
        return sorted({f.stem.split('_')[0] for f in self.storage_path.glob('*.parquet') if '_' in f.stem})

    async def get_data_range(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Optional[Tuple[datetime, datetime]]:
        """Détermine la plage de données pour une paire en lisant son fichier unique."""
        self._validate_initialization()
        file_path = self._get_file_path(pair, interval)
        if not file_path.exists(): return None
        
        def _get_range():
            df = self._read_parquet_file(file_path)
            if df is not None and not df.empty:
                return df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
            return None
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _get_range)

    async def delete_old_data(self, cutoff_date: datetime) -> Dict[str, int]:
        """Supprime les données antérieures à une date donnée en réécrivant les fichiers affectés."""
        self._validate_initialization()
        if cutoff_date.tzinfo is None:
            cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
        
        deleted_summary: Dict[str, int] = {}
        for file_path in list(self.storage_path.glob('*.parquet')):
            identifier = file_path.stem
            try:
                df = await asyncio.get_event_loop().run_in_executor(self._read_parquet_file, file_path)
                if df is None or df.empty: continue
                
                rows_before = len(df)
                df_kept = df[df.index >= cutoff_date]
                deleted_count = rows_before - len(df_kept)

                if deleted_count > 0:
                    if not df_kept.empty:
                        arrow_table = pa.Table.from_pandas(df_kept.reset_index())
                        pq.write_table(arrow_table, file_path, compression='snappy')
                    else:
                        file_path.unlink()
                deleted_summary[identifier] = deleted_count
            except Exception as e:
                logger.error(f"Failed to process old data for {identifier}: {e}")
                deleted_summary[identifier] = -1
        return deleted_summary

    async def get_statistics(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Dict[str, Any]:
        """Calcule et retourne des statistiques pour une paire/intervalle."""
        self._validate_initialization()
        df = await self.get_klines(pair, interval)
        if df is None or df.empty: return {"exists": False}

        data_range = await self.get_data_range(pair, interval)
        stats = {
            "exists": True, "pair": pair, "interval": interval, "count": len(df),
            "start_date": data_range[0] if data_range else None,
            "end_date": data_range[1] if data_range else None,
        }
        return stats

    async def optimize_storage(self) -> Dict[str, Any]:
        """Ré-écrit chaque fichier Parquet pour optimiser sa taille."""
        self._validate_initialization()
        stats = {"files_processed": 0, "size_before_mb": 0.0, "size_after_mb": 0.0, "errors": []}
        
        for file_path in list(self.storage_path.glob("*.parquet")):
            try:
                size_before = file_path.stat().st_size / (1024*1024)
                df = await asyncio.get_event_loop().run_in_executor(self._read_parquet_file, file_path)
                if df is None or df.empty: continue
                
                arrow_table = pa.Table.from_pandas(df.reset_index())
                pq.write_table(arrow_table, file_path, compression='snappy')
                
                size_after = file_path.stat().st_size / (1024*1024)
                stats["files_processed"] += 1
                stats["size_before_mb"] += size_before
                stats["size_after_mb"] += size_after
            except Exception as e:
                logger.error(f"Failed to optimize {file_path.name}: {e}")
                stats["errors"].append(file_path.name)
        
        logger.info(f"Storage optimization completed: {stats}")
        return stats
