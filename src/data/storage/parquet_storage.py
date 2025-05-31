# src/data/storage/parquet_storage.py
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from src.core.config import settings
from src.core.constants import Kline
from src.core.exceptions import ParquetStorageError, StorageError
from src.data.storage.base import BaseStorage


class ParquetStorage(BaseStorage):
    """
    Implémentation du stockage en fichiers Parquet.
    Utilise le partitionnement par paire/année/mois pour optimiser les performances.
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialise le stockage Parquet.
        
        Args:
            storage_path: Chemin racine pour le stockage des fichiers
        """
        self.storage_path = storage_path
        self.partition_cols = settings.data.parquet_partition_cols
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False
        
        logger.info(f"ParquetStorage initialized with path: {self.storage_path}")
        
    async def initialize(self) -> None:
        """Initialise le système de stockage."""
        try:
            # Créer le répertoire de stockage s'il n'existe pas
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.storage_path.mkdir,
                True,  # parents
                True   # exist_ok
            )
            
            self._initialized = True
            logger.success(f"ParquetStorage initialized at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ParquetStorage: {e}")
            raise StorageError(
                f"Failed to initialize ParquetStorage: {e}",
                original_exception=e
            )
            
    async def close(self) -> None:
        """Ferme proprement les connexions."""
        self.executor.shutdown(wait=True)
        logger.info("ParquetStorage closed")
        
    def _validate_initialization(self):
        """Vérifie que le storage est initialisé."""
        if not self._initialized:
            raise StorageError("ParquetStorage not initialized. Call initialize() first.")
            
    def _get_table_path(self, pair: str, interval: str) -> Path:
        """
        Construit le chemin pour une table (paire/intervalle).
        
        Args:
            pair: Paire de trading
            interval: Intervalle des klines
            
        Returns:
            Chemin de la table
        """
        # Normaliser l'intervalle (1m -> 1m, 1M -> 1M)
        safe_interval = interval.replace('/', '_')
        return self.storage_path / f"{pair}_{safe_interval}"
        
    def _prepare_dataframe_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare un DataFrame pour le stockage Parquet.
        
        Args:
            df: DataFrame à préparer
            
        Returns:
            DataFrame préparé
        """
        df_copy = df.copy()
        
        # S'assurer que l'index est une colonne
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
            
        # Ajouter les colonnes de partitionnement si nécessaire
        if 'year' in self.partition_cols and 'year' not in df_copy.columns:
            df_copy['year'] = df_copy['kline_open_time'].dt.year
            
        if 'month' in self.partition_cols and 'month' not in df_copy.columns:
            df_copy['month'] = df_copy['kline_open_time'].dt.month
            
        if 'day' in self.partition_cols and 'day' not in df_copy.columns:
            df_copy['day'] = df_copy['kline_open_time'].dt.day
            
        # Convertir les types pour optimiser le stockage
        numeric_columns = [
            'open_price', 'high_price', 'low_price', 'close_price',
            'base_asset_volume', 'quote_asset_volume',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype('float64')
                
        # Convertir les timestamps en UTC si nécessaire
        timestamp_columns = ['kline_open_time', 'kline_close_time']
        for col in timestamp_columns:
            if col in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = pd.to_datetime(df_copy[col], utc=True)
                
        return df_copy
        
    def _read_parquet_table(self, table_path: Path, filters: Optional[List] = None) -> Optional[pd.DataFrame]:
        """
        Lit une table Parquet avec des filtres optionnels.
        
        Args:
            table_path: Chemin de la table
            filters: Filtres PyArrow
            
        Returns:
            DataFrame ou None si la table n'existe pas
        """
        if not table_path.exists():
            return None
            
        try:
            # Lire avec les filtres
            df = pq.read_table(
                table_path,
                filters=filters,
                use_pandas_metadata=True
            ).to_pandas()
            
            # Définir l'index
            if 'kline_open_time' in df.columns:
                df.set_index('kline_open_time', inplace=True)
                df.sort_index(inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Error reading parquet table {table_path}: {e}")
            raise ParquetStorageError(
                f"Failed to read parquet table: {e}",
                original_exception=e
            )
            
    async def store_klines(
        self,
        df: pd.DataFrame,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> None:
        """
        Stocke les klines dans des fichiers Parquet.
        
        Args:
            df: DataFrame contenant les klines
            pair: Paire de trading
            interval: Intervalle des klines
        """
        self._validate_initialization()
        
        try:
            table_path = self._get_table_path(pair, interval)
            
            # Préparer le DataFrame
            df_prepared = self._prepare_dataframe_for_storage(df)
            
            # Fonction de stockage synchrone
            def _store():
                # Lire les données existantes si présentes
                existing_df = self._read_parquet_table(table_path)
                
                if existing_df is not None:
                    # Fusionner avec les données existantes
                    combined = pd.concat([existing_df, df_prepared])
                    
                    # Supprimer les doublons (garder le plus récent)
                    if 'kline_open_time' in combined.columns:
                        combined = combined.drop_duplicates(
                            subset=['kline_open_time'],
                            keep='last'
                        )
                    
                    # Trier par temps
                    combined.sort_values('kline_open_time', inplace=True)
                    
                    df_to_write = combined
                else:
                    df_to_write = df_prepared
                    
                # Écrire la table partitionnée
                table = pa.Table.from_pandas(df_to_write, preserve_index=False)
                
                pq.write_to_dataset(
                    table,
                    root_path=table_path,
                    partition_cols=self.partition_cols,
                    existing_data_behavior='delete_matching',
                    compression='snappy'
                )
                
                return len(df_to_write)
                
            # Exécuter dans le thread pool
            rows_written = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                _store
            )
            
            logger.info(f"Stored {rows_written} klines for {pair} at {interval} interval")
            
        except Exception as e:
            logger.error(f"Failed to store klines for {pair}: {e}")
            raise ParquetStorageError(
                f"Failed to store klines: {e}",
                original_exception=e
            )
            
    async def get_klines(
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les klines depuis les fichiers Parquet.
        
        Args:
            pair: Paire de trading
            interval: Intervalle des klines
            start_time: Date de début (incluse)
            end_time: Date de fin (incluse)
            limit: Nombre maximum de klines à retourner
            
        Returns:
            DataFrame avec les klines ou None si aucune donnée
        """
        self._validate_initialization()
        
        try:
            table_path = self._get_table_path(pair, interval)
            
            if not table_path.exists():
                logger.debug(f"No data found for {pair} at {interval}")
                return None
                
            # Construire les filtres PyArrow
            filters = []
            
            if start_time is not None:
                # Ajouter les filtres sur year/month si utilisés dans le partitionnement
                if 'year' in self.partition_cols:
                    filters.append(('year', '>=', start_time.year))
                if 'month' in self.partition_cols and 'year' in self.partition_cols:
                    # Filtre plus complexe pour year/month
                    filters.append(
                        (('year', '>', start_time.year), 
                         ('year', '==', start_time.year), ('month', '>=', start_time.month))
                    )
                    
            if end_time is not None:
                if 'year' in self.partition_cols:
                    filters.append(('year', '<=', end_time.year))
                if 'month' in self.partition_cols and 'year' in self.partition_cols:
                    filters.append(
                        (('year', '<', end_time.year),
                         ('year', '==', end_time.year), ('month', '<=', end_time.month))
                    )
                    
            # Fonction de lecture synchrone
            def _read():
                df = self._read_parquet_table(table_path, filters if filters else None)
                
                if df is None or df.empty:
                    return None
                    
                # Appliquer les filtres temporels précis
                if start_time is not None:
                    df = df[df.index >= start_time]
                if end_time is not None:
                    df = df[df.index <= end_time]
                    
                # Appliquer la limite
                if limit is not None and len(df) > limit:
                    df = df.iloc[-limit:]  # Prendre les plus récentes
                    
                return df
                
            # Exécuter dans le thread pool
            df = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                _read
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get klines for {pair}: {e}")
            raise ParquetStorageError(
                f"Failed to get klines: {e}",
                original_exception=e
            )
            
    async def get_available_pairs(self) -> List[str]:
        """
        Retourne la liste des paires disponibles dans le stockage.
        
        Returns:
            Liste des paires
        """
        self._validate_initialization()
        
        try:
            # Fonction de listing synchrone
            def _list_pairs():
                pairs = set()
                
                # Parcourir les répertoires
                for path in self.storage_path.iterdir():
                    if path.is_dir() and '_' in path.name:
                        # Extraire la paire du nom (format: BTCUSDC_1m)
                        pair = path.name.split('_')[0]
                        if pair.upper() == pair:  # Vérifier que c'est en majuscules
                            pairs.add(pair)
                            
                return sorted(list(pairs))
                
            # Exécuter dans le thread pool
            pairs = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                _list_pairs
            )
            
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to get available pairs: {e}")
            raise ParquetStorageError(
                f"Failed to get available pairs: {e}",
                original_exception=e
            )
            
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
        self._validate_initialization()
        
        try:
            table_path = self._get_table_path(pair, interval)
            
            if not table_path.exists():
                return None
                
            # Fonction de calcul synchrone
            def _get_range():
                # Utiliser les métadonnées Parquet pour optimiser
                parquet_file = pq.ParquetDataset(table_path)
                
                # Récupérer les statistiques des colonnes timestamp
                min_timestamp = None
                max_timestamp = None
                
                for piece in parquet_file.pieces:
                    metadata = piece.get_metadata()
                    for i in range(metadata.num_row_groups):
                        row_group = metadata.row_group(i)
                        
                        # Chercher la colonne kline_open_time
                        for j in range(row_group.num_columns):
                            column = row_group.column(j)
                            if metadata.schema.column(j).name == 'kline_open_time':
                                stats = column.statistics
                                if stats and stats.has_min_max:
                                    col_min = pd.Timestamp(stats.min)
                                    col_max = pd.Timestamp(stats.max)
                                    
                                    if min_timestamp is None or col_min < min_timestamp:
                                        min_timestamp = col_min
                                    if max_timestamp is None or col_max > max_timestamp:
                                        max_timestamp = col_max
                                        
                if min_timestamp and max_timestamp:
                    return (min_timestamp, max_timestamp)
                    
                # Fallback: lire les données pour obtenir la plage
                df = self._read_parquet_table(table_path)
                if df is not None and not df.empty:
                    return (df.index.min(), df.index.max())
                    
                return None
                
            # Exécuter dans le thread pool
            data_range = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                _get_range
            )
            
            return data_range
            
        except Exception as e:
            logger.error(f"Failed to get data range for {pair}: {e}")
            raise ParquetStorageError(
                f"Failed to get data range: {e}",
                original_exception=e
            )
            
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
        self._validate_initialization()
        
        try:
            deleted_counts = {}
            
            # Fonction de suppression synchrone
            def _delete_for_pair(table_path: Path, pair: str) -> int:
                if not table_path.exists():
                    return 0
                    
                # Lire toutes les données
                df = self._read_parquet_table(table_path)
                if df is None or df.empty:
                    return 0
                    
                # Filtrer les données à conserver
                df_keep = df[df.index >= cutoff_date]
                
                deleted_count = len(df) - len(df_keep)
                
                if deleted_count > 0:
                    # Supprimer l'ancienne table
                    import shutil
                    shutil.rmtree(table_path)
                    
                    # Réécrire les données conservées
                    if not df_keep.empty:
                        df_keep_prepared = self._prepare_dataframe_for_storage(df_keep)
                        table = pa.Table.from_pandas(df_keep_prepared, preserve_index=False)
                        
                        pq.write_to_dataset(
                            table,
                            root_path=table_path,
                            partition_cols=self.partition_cols,
                            existing_data_behavior='delete_matching',
                            compression='snappy'
                        )
                        
                return deleted_count
                
            # Parcourir toutes les tables
            tasks = []
            for path in self.storage_path.iterdir():
                if path.is_dir() and '_' in path.name:
                    pair = path.name.split('_')[0]
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        _delete_for_pair,
                        path,
                        pair
                    )
                    tasks.append((pair, task))
                    
            # Attendre toutes les tâches
            for pair, task in tasks:
                deleted_count = await task
                if deleted_count > 0:
                    deleted_counts[pair] = deleted_count
                    logger.info(f"Deleted {deleted_count} old klines for {pair}")
                    
            return deleted_counts
            
        except Exception as e:
            logger.error(f"Failed to delete old data: {e}")
            raise ParquetStorageError(
                f"Failed to delete old data: {e}",
                original_exception=e
            )
            
    async def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimise le stockage (compaction, réorganisation).
        
        Returns:
            Statistiques d'optimisation
        """
        self._validate_initialization()
        
        stats = {
            "tables_optimized": 0,
            "size_before_mb": 0,
            "size_after_mb": 0,
            "time_taken_seconds": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Fonction d'optimisation synchrone
            def _optimize_table(table_path: Path) -> Tuple[float, float]:
                if not table_path.exists():
                    return 0, 0
                    
                # Calculer la taille avant
                size_before = sum(f.stat().st_size for f in table_path.rglob('*.parquet'))
                
                # Lire et réécrire avec compaction
                df = self._read_parquet_table(table_path)
                if df is not None and not df.empty:
                    # Supprimer l'ancienne table
                    import shutil
                    shutil.rmtree(table_path)
                    
                    # Réécrire avec compression optimale
                    df_prepared = self._prepare_dataframe_for_storage(df)
                    table = pa.Table.from_pandas(df_prepared, preserve_index=False)
                    
                    pq.write_to_dataset(
                        table,
                        root_path=table_path,
                        partition_cols=self.partition_cols,
                        existing_data_behavior='delete_matching',
                        compression='snappy',
                        max_partitions=1024  # Limiter la fragmentation
                    )
                    
                # Calculer la taille après
                size_after = sum(f.stat().st_size for f in table_path.rglob('*.parquet'))
                
                return size_before / 1024 / 1024, size_after / 1024 / 1024  # En MB
                
            # Optimiser toutes les tables
            tasks = []
            for path in self.storage_path.iterdir():
                if path.is_dir() and '_' in path.name:
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        _optimize_table,
                        path
                    )
                    tasks.append(task)
                    
            # Attendre toutes les tâches
            for task in tasks:
                size_before, size_after = await task
                stats["size_before_mb"] += size_before
                stats["size_after_mb"] += size_after
                stats["tables_optimized"] += 1
                
            stats["time_taken_seconds"] = (datetime.now() - start_time).total_seconds()
            stats["space_saved_mb"] = stats["size_before_mb"] - stats["size_after_mb"]
            stats["compression_ratio"] = (
                (1 - stats["size_after_mb"] / stats["size_before_mb"]) * 100
                if stats["size_before_mb"] > 0 else 0
            )
            
            logger.info(
                f"Storage optimization completed: "
                f"{stats['tables_optimized']} tables, "
                f"{stats['space_saved_mb']:.2f} MB saved "
                f"({stats['compression_ratio']:.1f}% compression)"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to optimize storage: {e}")
            raise ParquetStorageError(
                f"Failed to optimize storage: {e}",
                original_exception=e
            )