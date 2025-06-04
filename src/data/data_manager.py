# src/data/data_manager.py
import asyncio
import json
from datetime import datetime, timedelta, timezone # Added timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any
import pandas as pd
import numpy as np
from loguru import logger
import time
from concurrent.futures import ThreadPoolExecutor
import threading # Used by DownloadProgress, not directly by DataManager async methods

from src.core.config import settings
from src.core.constants import Kline, StorageTypes, System
from src.core.exceptions import (
    DataError,
    DataDownloadError,
    StorageError,
    ConfigurationError,
    KlineValidationError # Assurez-vous que KlineValidationError est importé
)
from src.data.binance_client import BinanceDataClient
from src.data.kline_processor import KlineProcessor
from src.data.storage.base import BaseStorage
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.postgres_storage import PostgresStorage
from src.data.storage.cache import CacheManager
import inspect # For debug logging, can be removed later


class DownloadProgress:
    """Classe pour suivre la progression du téléchargement."""
    
    def __init__(self, total_pairs: int, total_days: int):
        self.total_pairs = total_pairs
        self.total_days = total_days # This is an estimate for display
        self.pairs_completed = 0
        self.current_pair = ""
        self.current_pair_progress = 0 # klines downloaded for current_pair in current batch
        self.current_pair_total = 0    # klines estimated for current_pair in current batch
        self.current_pair_klines_total_downloaded_for_session = 0 # Total klines for pair in this session
        self.start_time = time.time()
        self.errors: List[Dict[str, Any]] = []
        self._lock = threading.Lock() # Lock for thread-safe updates if progress is updated from threads
        
    def update_pair_batch_progress(self, pair: str, klines_in_batch: int, estimated_total_for_pair_batch: int):
        """Met à jour la progression pour un batch de la paire actuelle."""
        with self._lock:
            self.current_pair = pair
            self.current_pair_progress = klines_in_batch
            self.current_pair_total = estimated_total_for_pair_batch
            self.current_pair_klines_total_downloaded_for_session += klines_in_batch
            
    def complete_pair(self, pair: str):
        """Marque une paire comme complétée."""
        with self._lock:
            self.pairs_completed += 1
            self.current_pair_klines_total_downloaded_for_session = 0 # Reset for next pair
            logger.success(f"Completed download for {pair} ({self.pairs_completed}/{self.total_pairs})")
            
    def add_error(self, pair: str, error: str):
        """Ajoute une erreur."""
        with self._lock:
            self.errors.append({
                "pair": pair,
                "error": error,
                "timestamp": datetime.now(timezone.utc) # Use timezone.utc
            })
            
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de progression."""
        with self._lock:
            elapsed = time.time() - self.start_time
            pairs_remaining = self.total_pairs - self.pairs_completed
            
            eta_seconds = 0
            if self.pairs_completed > 0 and self.total_pairs > self.pairs_completed:
                avg_time_per_pair = elapsed / self.pairs_completed
                eta_seconds = avg_time_per_pair * pairs_remaining
            elif self.total_pairs == 0: # Avoid division by zero if no pairs
                eta_seconds = 0

            return {
                "pairs_completed": self.pairs_completed,
                "total_pairs": self.total_pairs,
                "current_pair": self.current_pair,
                "current_pair_batch_progress_klines": self.current_pair_progress,
                "current_pair_batch_estimated_total_klines": self.current_pair_total,
                "current_pair_session_total_klines": self.current_pair_klines_total_downloaded_for_session,
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
        self.checkpoint_file = self.checkpoint_dir / "dm_download_checkpoint.json"
        
    def save(self, state: Dict[str, Any]):
        """Sauvegarde l'état actuel."""
        try:
            if 'last_update' in state and isinstance(state['last_update'], datetime):
                state['last_update'] = state['last_update'].isoformat()
            if 'start_date' in state and isinstance(state['start_date'], datetime):
                 state['start_date'] = state['start_date'].isoformat()

            with open(self.checkpoint_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"DataManager Checkpoint saved: {state}")
        except Exception as e:
            logger.error(f"DataManager: Failed to save checkpoint: {e}")
            
    def load(self) -> Optional[Dict[str, Any]]:
        """Charge le dernier checkpoint."""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            if 'last_update' in state and isinstance(state['last_update'], str):
                state['last_update'] = datetime.fromisoformat(state['last_update'])
            if 'start_date' in state and isinstance(state['start_date'], str): 
                state['start_date'] = datetime.fromisoformat(state['start_date']).replace(tzinfo=timezone.utc)

            logger.info(f"DataManager Checkpoint loaded: {state}")
            return state
        except Exception as e:
            logger.error(f"DataManager: Failed to load checkpoint: {e}")
            return None
            
    def clear(self):
        """Supprime le checkpoint."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("DataManager Checkpoint cleared")
            except OSError as e:
                logger.error(f"DataManager: Error clearing checkpoint file: {e}")


class DataManager:
    """
    Gestionnaire principal des données.
    Coordonne le téléchargement, le traitement et le stockage des données.
    """
    
    def __init__(self):
        self.storage_type = settings.data.storage_type
        self.storage: Optional[BaseStorage] = None
        self.kline_processor = KlineProcessor()
        self.cache_manager = CacheManager() if settings.data.cache_enabled else None
        self.binance_client: Optional[BinanceDataClient] = None
        self.checkpoint = DownloadCheckpoint(settings.data.storage_path / ".dm_checkpoints")
        
        thread_pool_workers = getattr(settings.system, 'thread_pool_workers', 4) if hasattr(settings, 'system') else 4
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_workers)
        self._is_initialized = False # Flag to track initialization status
        
        logger.info(f"DataManager instance created with {self.storage_type} storage. Call initialize() to setup backends.")
            
    async def initialize(self):
        """Initialise les composants asynchrones (backends de stockage, cache, client API)."""
        if self._is_initialized:
            logger.debug("DataManager components already initialized.")
            return

        logger.info(f"Initializing DataManager components...")
        
        if self.storage_type == StorageTypes.PARQUET:
            self.storage = ParquetStorage(settings.data.storage_path)
        elif self.storage_type == StorageTypes.POSTGRES:
            if not settings.database or not settings.database.url:
                raise ConfigurationError("PostgreSQL URL not configured for DataManager")
            self.storage = PostgresStorage(str(settings.database.url))
        else:
            raise ConfigurationError(f"Unknown storage type in DataManager: {self.storage_type}")
            
        if self.storage:
            await self.storage.initialize()
        
        if self.cache_manager:
            await self.cache_manager.initialize()
        
        self.binance_client = BinanceDataClient() 
        await self.binance_client.initialize()
        
        self._is_initialized = True # Set flag after successful initialization
        logger.success("DataManager initialized successfully (storage, cache, API client)")
        
    async def close(self):
        logger.info("Closing DataManager resources...")
        if self.storage and hasattr(self.storage, 'close'):
            await self.storage.close()
        if self.cache_manager and hasattr(self.cache_manager, 'close'):
            await self.cache_manager.close()
        if self.binance_client and hasattr(self.binance_client, 'close'):
            await self.binance_client.close()
        self.executor.shutdown(wait=True)
        self._is_initialized = False # Reset flag on close
        logger.info("DataManager resources closed.")
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def download_historical_data(
        self,
        pairs: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = Kline.INTERVAL_1MINUTE,
        resume: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        
        if not self._is_initialized: # Ensure initialized before proceeding
            await self.initialize()

        final_start_date: Optional[datetime] = None
        if isinstance(start_date, str):
            dt_naive_start = pd.to_datetime(start_date)
            final_start_date = dt_naive_start.replace(tzinfo=timezone.utc) if dt_naive_start.tzinfo is None else dt_naive_start.astimezone(timezone.utc)
        elif isinstance(start_date, datetime):
            final_start_date = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date.astimezone(timezone.utc)

        final_end_date: datetime
        if end_date is None:
            final_end_date = datetime.now(timezone.utc)
        elif isinstance(end_date, str):
            dt_naive_end = pd.to_datetime(end_date)
            final_end_date = dt_naive_end.replace(tzinfo=timezone.utc) if dt_naive_end.tzinfo is None else dt_naive_end.astimezone(timezone.utc)
        elif isinstance(end_date, datetime):
            final_end_date = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date.astimezone(timezone.utc)
        
        start_ts: Optional[int] = int(final_start_date.timestamp() * 1000) if final_start_date else None
        end_ts: int = int(final_end_date.timestamp() * 1000)

        if start_ts and end_ts <= start_ts:
            logger.warning(f"End date ({final_end_date}) is not after start date ({final_start_date}). No data will be downloaded.")
            return {"success": True, "message": "End date is not after start date.", "pairs_processed": 0, "klines_added": 0, "errors": []}

        dm_checkpoint_state = None
        if resume:
            dm_checkpoint_state = self.checkpoint.load()
            if dm_checkpoint_state and (
                set(dm_checkpoint_state.get("pending_pairs", [])) != set(pairs) or
                dm_checkpoint_state.get("interval") != interval
            ):
                logger.warning("DataManager checkpoint parameters mismatch. Clearing and starting fresh.")
                self.checkpoint.clear()
                dm_checkpoint_state = None
        
        total_days_for_progress_display = 30
        if final_start_date:
            total_days_for_progress_display = max(1, (final_end_date - final_start_date).days)
        
        overall_progress_tracker = DownloadProgress(len(pairs), total_days_for_progress_display)
        
        pairs_to_process_dm = list(pairs)
        last_processed_timestamp_for_pair: Dict[str, int] = {}

        if dm_checkpoint_state:
            processed_in_checkpoint = dm_checkpoint_state.get("completed_pairs_in_dm", [])
            pairs_to_process_dm = [p for p in pairs if p not in processed_in_checkpoint]
            overall_progress_tracker.pairs_completed = len(processed_in_checkpoint)
            
            current_pair_from_chk = dm_checkpoint_state.get("current_pair_dm")
            last_ts_from_chk = dm_checkpoint_state.get("current_pair_last_ts_dm")
            if current_pair_from_chk in pairs_to_process_dm and last_ts_from_chk:
                logger.info(f"DataManager resuming {current_pair_from_chk} from DM checkpoint ts {last_ts_from_chk + 1}")
                effective_resume_ts = last_ts_from_chk + 1
                if start_ts is None or effective_resume_ts > start_ts:
                    last_processed_timestamp_for_pair[current_pair_from_chk] = effective_resume_ts
        
        if not self.binance_client: 
            raise DataError("Binance client not available in DataManager for download.")

        for pair in pairs_to_process_dm:
            try:
                logger.info(f"DataManager: Downloading {pair} from {final_start_date or 'earliest'} to {final_end_date}")
                current_pair_effective_start_ts = last_processed_timestamp_for_pair.get(pair, start_ts)

                def binance_client_progress_callback(klines_in_batch: int, estimated_total_for_batch: int, last_kline_ts_in_batch: Optional[int] = None):
                    overall_progress_tracker.update_pair_batch_progress(pair, klines_in_batch, estimated_total_for_batch)
                    if progress_callback:
                        cli_stats = {
                            "current_pair_klines_downloaded_batch": klines_in_batch,
                            "current_pair_klines_total_downloaded": overall_progress_tracker.current_pair_klines_total_downloaded_for_session,
                            "current_pair_klines_estimated_total": estimated_total_for_batch,
                            "current_pair_last_kline_timestamp_ms": last_kline_ts_in_batch 
                        }
                        progress_callback(cli_stats)
                    if resume and last_kline_ts_in_batch is not None:
                        self.checkpoint.save({
                            "pending_pairs": [p_item for p_item in pairs_to_process_dm if p_item != pair and p_item not in [e['pair'] for e in overall_progress_tracker.errors]],
                            "completed_pairs_in_dm": overall_progress_tracker.pairs_completed,
                            "interval": interval,
                            "current_pair_dm": pair, 
                            "current_pair_last_ts_dm": last_kline_ts_in_batch,
                            "last_update": datetime.now(timezone.utc)
                        })
                
                klines = await self.binance_client.fetch_klines_batch(
                    symbol=pair, interval=interval,
                    start_time=current_pair_effective_start_ts, end_time=end_ts,
                    progress_callback=binance_client_progress_callback
                )
                
                if not klines:
                    logger.warning(f"DataManager: No data downloaded for {pair} in the given range.")
                    overall_progress_tracker.add_error(pair, "No data available for the period.")
                else:
                    df = self._klines_to_dataframe(klines, pair)
                    validation_errors = self._validate_dataframe(df)
                    if validation_errors:
                        logger.error(f"DataManager: Validation errors for {pair}: {validation_errors}")
                        overall_progress_tracker.add_error(pair, f"Validation failed: {validation_errors}")
                    else:
                        await self._store_klines(df, pair, interval)
                
                overall_progress_tracker.complete_pair(pair)

            except Exception as e:
                logger.error(f"DataManager: Error downloading or processing {pair}: {e}", exc_info=True)
                overall_progress_tracker.add_error(pair, str(e))

        all_pairs_accounted_for = True
        for p_item in pairs_to_process_dm:
            is_completed_successfully = p_item in [overall_progress_tracker.current_pair if i == overall_progress_tracker.pairs_completed -1 else "" for i in range(overall_progress_tracker.pairs_completed)] 
            is_errored = p_item in [e['pair'] for e in overall_progress_tracker.errors]
            if not (is_completed_successfully or is_errored) and overall_progress_tracker.pairs_completed < overall_progress_tracker.total_pairs :
                 all_pairs_accounted_for = False
                 break
        
        if all_pairs_accounted_for and overall_progress_tracker.pairs_completed == len(pairs_to_process_dm):
            self.checkpoint.clear()
            
        final_stats = overall_progress_tracker.get_stats()
        final_stats["success"] = overall_progress_tracker.pairs_completed == len(pairs) and not final_stats["errors_count"]
        
        if final_stats["errors_count"] > 0:
            logger.warning(f"DataManager: Download completed with {final_stats['errors_count']} errors for some pairs.")
        else:
            logger.success("DataManager: Download process completed successfully for all assigned pairs.")
        return final_stats
        
    def _klines_to_dataframe(self, klines: List[Dict[str, Any]], pair: str) -> pd.DataFrame:
        if not klines: return pd.DataFrame()
        df = pd.DataFrame(klines)
        rename_map = {
            'open_time': 'kline_open_time', 'open': 'open_price', 'high': 'high_price',
            'low': 'low_price', 'close': 'close_price', 'volume': 'base_asset_volume',
            'close_time': 'kline_close_time', 'quote_asset_volume': 'quote_asset_volume',
            'number_of_trades': 'number_of_trades', 
            'taker_buy_base_asset_volume': 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume': 'taker_buy_quote_asset_volume',
        }
        # Garder seulement les colonnes qui existent dans le DataFrame avant de renommer
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=actual_rename_map, inplace=True)

        # S'assurer que la colonne 'kline_open_time' existe après le renommage potentiel
        if 'kline_open_time' not in df.columns:
            # Si 'open_time' était le nom original et n'a pas été renommé (parce qu'il n'existait pas)
            # ou si la colonne attendue n'est tout simplement pas là.
            logger.error("Timestamp column 'kline_open_time' (or 'open_time') is missing from klines data.")
            # Retourner un DataFrame vide ou lever une exception selon la criticité
            return pd.DataFrame()


        df['pair'] = pair
        df['kline_open_time'] = pd.to_datetime(df['kline_open_time'], unit='ms', utc=True)
        
        if 'kline_close_time' in df.columns: # Vérifier si la colonne existe
            df['kline_close_time'] = pd.to_datetime(df['kline_close_time'], unit='ms', utc=True)
        
        df['is_kline_closed'] = True 
        
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 
                        'base_asset_volume', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.sort_values('kline_open_time', inplace=True)
        df.set_index('kline_open_time', inplace=True)
        return df
        
    def _validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        errors = []
        if df.empty: return errors
        required_columns = ['pair', 'open_price', 'high_price', 'low_price', 'close_price', 'base_asset_volume']
        if not isinstance(df.index, pd.DatetimeIndex): errors.append("Index is not DatetimeIndex.")
        elif df.index.name != 'kline_open_time': errors.append(f"Index name is '{df.index.name}', expected 'kline_open_time'.")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
            # Ne pas retourner immédiatement, continuer les autres vérifications si possible
        
        # Vérifier les nuls seulement si les colonnes existent
        cols_to_check_nulls = [col for col in required_columns if col in df.columns]
        if cols_to_check_nulls and df[cols_to_check_nulls].isnull().any().any():
            errors.append(f"Null values in required columns: {df[cols_to_check_nulls].isnull().sum()[df[cols_to_check_nulls].isnull().sum() > 0].to_dict()}")
        
        ohlc_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in ohlc_cols):
            invalid_ohlc = df[(df['high_price'] < df['low_price']) | (df['high_price'] < df['open_price']) | \
                              (df['high_price'] < df['close_price']) | (df['low_price'] > df['open_price']) | \
                              (df['low_price'] > df['close_price'])]
            if not invalid_ohlc.empty: errors.append(f"Invalid OHLC data in {len(invalid_ohlc)} rows.")
        # else: errors.append("Skipping OHLC consistency check due to missing or non-numeric OHLC columns.") # Optionnel: logguer si le check est sauté

        if 'base_asset_volume' in df.columns and pd.api.types.is_numeric_dtype(df['base_asset_volume']):
            if (df['base_asset_volume'] < 0).any(): errors.append("Negative volumes found.")
        
        if isinstance(df.index, pd.DatetimeIndex): # Vérifier à nouveau au cas où l'index n'était pas DatetimeIndex au début
            if not df.index.is_monotonic_increasing: errors.append("Data not sorted by time.")
            if df.index.has_duplicates: errors.append(f"Found {df.index.duplicated().sum()} duplicate timestamps.")
        return errors
        
    async def _store_klines(self, df: pd.DataFrame, pair: str, interval: str): 
        if df.empty:
            logger.debug(f"No klines to store for {pair} {interval}.")
            return
        if not self.storage: raise StorageError("Storage not initialized.")
        try:
            await self.storage.store_klines(df, pair, interval)
            logger.debug(f"Stored {len(df)} klines for {pair} ({interval})")
            if self.cache_manager:
                await self.cache_manager.invalidate_pattern(f"klines:{pair}:{interval}:*")
        except Exception as e:
            logger.error(f"DataManager: Failed to store klines for {pair} ({interval}): {e}")
            raise StorageError(f"Failed to store klines: {e}", original_exception=e)
            
    async def get_klines( 
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True 
    ) -> Optional[pd.DataFrame]:
        if not self._is_initialized: # Assurer l'initialisation
            await self.initialize()
        if not self.storage: # Vérifier après l'initialisation
             raise StorageError("Storage not initialized in DataManager.get_klines.")

        final_start_time = start_time.replace(tzinfo=timezone.utc) if start_time and start_time.tzinfo is None else start_time
        final_end_time = end_time.replace(tzinfo=timezone.utc) if end_time and end_time.tzinfo is None else end_time
        
        cache_key = None
        if use_cache and self.cache_manager and self.cache_manager.is_available(): 
            key_parts = ["klines", pair, interval]
            key_parts.append(final_start_time.isoformat() if final_start_time else "None")
            key_parts.append(final_end_time.isoformat() if final_end_time else "None")
            key_parts.append(str(limit) if limit else "None")
            cache_key = self.cache_manager.generate_key(*key_parts)
            
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None and isinstance(cached_data, pd.DataFrame): 
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data.copy()
            logger.debug(f"Cache miss for {cache_key}")
            
        try:
            df = await self.storage.get_klines(
                pair=pair, interval=interval, start_time=final_start_time,
                end_time=final_end_time, limit=limit
            )
            
            # CORRECTION: L'appel à resample_klines ici est potentiellement problématique.
            # Si le but est de s'assurer que les données sont à la fréquence 'interval',
            # et que self.storage.get_klines retourne déjà des données à cette fréquence,
            # cet appel est redondant ou pourrait introduire des erreurs si mal configuré.
            # Si le but était de calculer des klines glissantes, les paramètres étaient incorrects.
            # Pour l'instant, nous allons supposer que self.storage.get_klines retourne
            # les données à la bonne fréquence 'interval' et que ce resampling n'est pas nécessaire ici.
            # La stratégie elle-même est responsable de resampler pour ses indicateurs si besoin.
            
            # Ligne originale commentée :
            # if df is not None and not df.empty and self.kline_processor:
            #     try:
            #         interval_td_str = interval.replace('m','min').replace('h','H').replace('d','D').replace('w','W')
            #         if 'M' in interval_td_str and interval_td_str.endswith('M'):
            #             interval_td_str = interval_td_str.replace('M', 'ME')
            #         interval_td = pd.to_timedelta(interval_td_str)
            #         expected_ms = interval_td.total_seconds() * 1000
            #         df = self.kline_processor.resample_klines(df.copy(), pair, interval, expected_ms) # Appel incorrect
            #     except Exception as e_proc:
            #         logger.error(f"Error processing klines from storage for {pair} {interval}: {e_proc}")
            #         # Ne pas retourner df potentiellement mal traité. Retourner None ou lever.
            #         # Pour l'instant, on logue l'erreur et on continue avec le df non processé par cette étape.
            #         # Si le df original du storage est None ou vide, il sera retourné comme tel.

            if df is not None and not df.empty:
                # S'assurer que l'index est un DatetimeIndex si ce n'est pas déjà le cas
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'kline_open_time' in df.columns: # Nom de colonne après _klines_to_dataframe
                        df = df.set_index(pd.to_datetime(df['kline_open_time'], unit='ms', utc=True))
                        df.index.name = 'kline_open_time' # Assurer le nom de l'index
                    else: # Fallback si la colonne timestamp n'est pas là
                        logger.warning(f"Cannot set DatetimeIndex for {pair} {interval}, 'kline_open_time' missing.")
                
                # Vérifier la validité des klines récupérées (optionnel mais recommandé)
                is_valid, validation_errors = self.kline_processor.validate_klines(df.copy()) # Utiliser une copie pour la validation
                if not is_valid:
                    logger.warning(f"Validation failed for klines {pair} {interval} from storage: {validation_errors}")
                    # Décider si retourner les données invalides ou None/lever une exception
                    # Pour l'instant, on retourne les données avec un avertissement.
            
            if df is not None and not df.empty and use_cache and self.cache_manager and cache_key: 
                await self.cache_manager.set(cache_key, df.copy())
            return df
        except Exception as e:
            logger.error(f"DataManager: Failed to get klines for {pair}: {e}", exc_info=True)
            raise DataError(f"Failed to retrieve klines: {e}", original_exception=e)

    async def get_available_pairs(self) -> List[str]:
        if not self._is_initialized: await self.initialize()
        if not self.storage: raise StorageError("Storage not initialized.")
        try:
            return await self.storage.get_available_pairs()
        except Exception as e:
            logger.error(f"DataManager: Failed to get available pairs: {e}")
            raise DataError("Failed to get available pairs", original_exception=e)
            
    async def get_data_range(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Optional[Tuple[datetime, datetime]]:
        if not self._is_initialized: await self.initialize()
        if not self.storage: raise StorageError("Storage not initialized.")
        try:
            return await self.storage.get_data_range(pair, interval)
        except Exception as e:
            logger.error(f"DataManager: Failed to get data range for {pair}: {e}")
            raise DataError("Failed to get data range", original_exception=e)
            
    async def update_latest_data(
        self,
        pairs: Optional[List[str]] = None,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> Dict[str, Any]:
        if not self._is_initialized: await self.initialize()
        stats = {"pairs_updated": 0, "klines_added": 0, "errors": []}
        if not self.storage: raise StorageError("Storage not initialized.")
        target_pairs = pairs if pairs is not None else await self.get_available_pairs()
        if not target_pairs:
            logger.info("No pairs to update.")
            return stats
        if not self.binance_client: raise DataError("Binance client not available in DataManager for update.")

        for pair in target_pairs:
            try:
                data_range = await self.get_data_range(pair, interval)
                last_stored_utc_dt: Optional[datetime] = data_range[1] if data_range and data_range[1] else None
                fetch_start_time_ms: Optional[int] = int(last_stored_utc_dt.timestamp() * 1000) + 1 if last_stored_utc_dt else None
                logger.info(f"Updating {pair} ({interval}) from {last_stored_utc_dt + timedelta(milliseconds=1) if last_stored_utc_dt else 'earliest'}")
                
                new_klines = await self.binance_client.fetch_klines_batch(
                    symbol=pair, interval=interval, start_time=fetch_start_time_ms, end_time=None
                )
                
                if new_klines:
                    df_new = self._klines_to_dataframe(new_klines, pair)
                    if not df_new.empty:
                        await self._store_klines(df_new, pair, interval)
                        stats["pairs_updated"] += 1
                        stats["klines_added"] += len(new_klines)
                        logger.info(f"Updated {pair} ({interval}) with {len(new_klines)} new klines.")
                else:
                    logger.debug(f"No new data for {pair} ({interval}).")
            except Exception as e:
                logger.error(f"DataManager: Failed to update {pair} ({interval}): {e}")
                stats["errors"].append({"pair": pair, "interval": interval, "error": str(e)})
        logger.info(f"Update latest data completed: {stats['pairs_updated']} pairs updated, {stats['klines_added']} klines added.")
        if stats["errors"]: logger.warning(f"Encountered {len(stats['errors'])} errors during update.")
        return stats
            
    async def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        if not self._is_initialized: await self.initialize()
        if not self.storage: raise StorageError("Storage not initialized.")
        cutoff_date_utc = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        logger.info(f"DataManager: Cleaning up data older than {cutoff_date_utc} (UTC)")
        try:
            deleted_counts = await self.storage.delete_old_data(cutoff_date_utc)
            if self.cache_manager: await self.cache_manager.clear()
            total_deleted = sum(deleted_counts.values())
            logger.info(f"DataManager: Cleanup completed. {total_deleted} klines deleted.")
            return deleted_counts
        except Exception as e:
            logger.error(f"DataManager: Failed to cleanup old data: {e}")
            raise DataError("Failed to cleanup old data", original_exception=e)

async def _main_test_datamanager():
    logger.info("DataManager Test: Starting example usage.")
    try:
        async with DataManager() as dm: 
            test_pair = "BTCUSDT"
            test_start_date_str = "2024-01-01T00:00:00"
            test_end_date_str = "2024-01-01T01:00:00" # Short period for testing
            
            test_start_date = datetime.fromisoformat(test_start_date_str).replace(tzinfo=timezone.utc)
            test_end_date = datetime.fromisoformat(test_end_date_str).replace(tzinfo=timezone.utc)

            logger.info(f"DataManager Test: Downloading data for {test_pair} from {test_start_date} to {test_end_date}...")
            download_results = await dm.download_historical_data(
                pairs=[test_pair],
                start_date=test_start_date,
                end_date=test_end_date,
                interval="1m"
            )
            logger.info(f"DataManager Test: Download results: {download_results}")

            logger.info(f"DataManager Test: Fetching klines for {test_pair}...")
            kline_data = await dm.get_klines( 
                pair=test_pair,
                interval="1m",
                start_time=test_start_date,
                end_time=test_end_date,
                use_cache=False 
            )

            if kline_data is not None and not kline_data.empty:
                logger.success(f"DataManager Test: Successfully fetched {len(kline_data)} klines for {test_pair}.")
                logger.info(f"DataManager Test: Sample data:\n{kline_data.head()}")
            else:
                logger.warning(f"DataManager Test: No data returned for {test_pair}.")
            
    except Exception as e:
        logger.exception(f"DataManager Test: An error occurred: {e}")

if __name__ == '__main__':
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    try:
        asyncio.run(_main_test_datamanager())
    except KeyboardInterrupt:
        logger.info("DataManager test (manual run) interrupted.")
    except Exception as e:
        logger.error(f"Error running DataManager main_test: {e}", exc_info=True)

