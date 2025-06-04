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
    
    def __init__(self, storage_path : Path):
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
            # Fonction synchrone pour la gestion de chemin
            def _ensure_directory_exists(path: Path):
                if path.is_file(): # Si le chemin existe et est un fichier
                    logger.warning(f"Storage path {path} exists as a file. Attempting to remove and create directory.")
                    try:
                        path.unlink() # Supprimer le fichier
                        path.mkdir(parents=True, exist_ok=False) # Tenter de créer le répertoire
                        logger.info(f"Successfully removed file and created directory at {path}.")
                    except Exception as e_replace:
                        # Envelopper l'exception originale pour un meilleur diagnostic
                        raise StorageError(f"Storage path {path} is a file and could not be replaced by a directory: {e_replace}", original_exception=e_replace)
                elif not path.is_dir(): # Si le chemin n'est pas un répertoire (donc n'existe pas ou est un autre type de fichier)
                    logger.info(f"Storage directory {path} does not exist or is not a directory. Creating it.")
                    path.mkdir(parents=True, exist_ok=True) # exist_ok=True est sûr ici
                    logger.info(f"Successfully created storage directory {path}.")
                else:
                    # Le chemin existe déjà et est un répertoire
                    logger.debug(f"Storage directory {path} already exists.")

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                _ensure_directory_exists,
                self.storage_path
            )
            
            self._initialized = True
            logger.success(f"ParquetStorage initialized at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ParquetStorage: {e}")
            # S'assurer que l'exception est bien une StorageError si elle vient de nos vérifications
            if not isinstance(e, StorageError):
                raise StorageError(
                    f"Failed to initialize ParquetStorage: {e}",
                    original_exception=e
                )
            else:
                raise # Relancer la StorageError déjà formatée
            
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
                
            # Construire les filtres PyArrow DNF
            final_dnf_filters = []

            # Helper to create DNF for a single condition (e.g., start or end)
            def create_time_dnf(time_val: datetime, is_start: bool, partition_cols: List[str]) -> Optional[List[List[Tuple[str, str, Any]]]]:
                dnf = []
                year_partition = 'year' in partition_cols
                month_partition = 'month' in partition_cols

                if year_partition and month_partition:
                    if is_start:
                        # (year > time_val.year) OR (year == time_val.year AND month >= time_val.month)
                        dnf = [
                            [('year', '>', time_val.year)],
                            [('year', '==', time_val.year), ('month', '>=', time_val.month)]
                        ]
                    else: # is_end
                        # (year < time_val.year) OR (year == time_val.year AND month <= time_val.month)
                        dnf = [
                            [('year', '<', time_val.year)],
                            [('year', '==', time_val.year), ('month', '<=', time_val.month)]
                        ]
                elif year_partition: # Only year partition
                    if is_start:
                        dnf = [[('year', '>=', time_val.year)]]
                    else: # is_end
                        dnf = [[('year', '<=', time_val.year)]]
                # If no relevant partitions, DNF remains empty, relying on Pandas filtering
                return dnf if dnf else None

            start_dnf = None
            if start_time:
                start_dnf = create_time_dnf(start_time, True, self.partition_cols)

            end_dnf = None
            if end_time:
                end_dnf = create_time_dnf(end_time, False, self.partition_cols)

            if start_dnf and end_dnf:
                # Combine S_dnf AND E_dnf:
                # (S_conj1 OR S_conj2 ...) AND (E_conj1 OR E_conj2 ...)
                # = (S_conj1 AND E_conj1) OR (S_conj1 AND E_conj2) OR ...
                for s_conj in start_dnf:
                    for e_conj in end_dnf:
                        # Check for contradictions like year > X AND year < X if SY == EY for simple cases
                        # More complex contradiction checks are harder. Assume valid ranges for now.
                        final_dnf_filters.append(s_conj + e_conj)
            elif start_dnf:
                final_dnf_filters = start_dnf
            elif end_dnf:
                final_dnf_filters = end_dnf
            
            # Ensure final_dnf_filters is None if empty, so pq.read_table doesn't receive an empty list
            # which might be interpreted as "match nothing".
            # If final_dnf_filters is [], it means no partition filters were applicable or generated.
            # pq.read_table with filters=None or filters=[] (if it means no filters) reads all partitions.
            # According to pyarrow docs, filters=None reads all. An empty list for filters might be an error or select nothing.
            # It's safer to pass None if no filters are intended.
            pq_filters_arg = final_dnf_filters if final_dnf_filters else None

            # Fonction de lecture synchrone
            def _read():
                # Pass None if final_dnf_filters is empty
                df = self._read_parquet_table(table_path, filters=pq_filters_arg)
                
                if df is None or df.empty:
                    return None
                    
                # Appliquer les filtres temporels précis sur le DataFrame chargé
                # This is crucial as partition filters are coarse.
                if start_time is not None:
                    df = df[df.index >= start_time] # Assumes index is DatetimeIndex
                if end_time is not None:
                    df = df[df.index <= end_time] # Assumes index is DatetimeIndex
                    
                # Appliquer la limite (après all filters, typically on sorted data)
                # The problem description doesn't specify sorting for limit,
                # but usually, limit implies latest N records if time-series.
                # The original code sorts by index after loading in _read_parquet_table.
                # If limit is to be applied, it should generally be on the final, sorted DataFrame.
                if limit is not None:
                    if not df.empty:
                        # Ensure DataFrame is sorted by index if not already
                        if not df.index.is_monotonic_increasing:
                             df.sort_index(inplace=True) # Sort if not already sorted for consistent limit application
                        # Take the most recent 'limit' records if data is chronological
                        # Or first 'limit' records if that's the desire for other types of limits.
                        # Assuming "latest N" for time series:
                        df = df.iloc[-limit:] 
                    
                return df
                
            # Exécuter dans le thread pool
            df = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                _read
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get klines for {pair}: {e}")
            # Ensure original exception type is preserved if it's already a ParquetStorageError
            if isinstance(e, ParquetStorageError):
                raise
            raise ParquetStorageError(
                f"Failed to get klines for {pair}: {e}",
                original_exception=e
            )

    # ... (rest of the class)
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