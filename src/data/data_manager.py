# src/data/data_manager.py
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
from loguru import logger
import time
from concurrent.futures import ThreadPoolExecutor

from src.core.config import get_settings
from src.core.constants import Kline, StorageTypes, System
from src.core.exceptions import (
    DataError,
    DataDownloadError,
    StorageError,
    ConfigurationError,
    KlineValidationError
)
from src.data.binance_client import BinanceDataClient
from src.data.kline_processor import KlineProcessor
from src.data.storage.base import BaseStorage
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.postgres_storage import PostgresStorage
from src.data.storage.cache import CacheManager
from src.data.enriched_dataframe import EnrichedDataFrame

class DataManager:
    """
    Gestionnaire principal des données.
    Coordonne le téléchargement, le traitement, le stockage et la récupération des données de marché.
    Sert de façade centrale pour toutes les opérations de données.
    """

    def __init__(self):
        """Initialise les composants du DataManager."""
        settings = get_settings()
        self.storage_type = settings.data.storage_type
        self.storage: Optional[BaseStorage] = None
        self.kline_processor = KlineProcessor()
        self.cache_manager = CacheManager() if settings.data.cache_enabled else None
        self.binance_client: Optional[BinanceDataClient] = None

        thread_pool_workers = getattr(settings.system, 'thread_pool_workers', System.THREAD_POOL_WORKERS_DEFAULT)
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_workers)
        self._is_initialized = False

        logger.info(f"DataManager instance created with {self.storage_type} storage. Call initialize() to setup.")

    async def initialize(self):
        """Initialise de manière asynchrone les connexions (stockage, cache, client API)."""
        if self._is_initialized:
            return

        logger.info("Initializing DataManager components...")

        settings = get_settings()
        if self.storage_type == StorageTypes.PARQUET.value:
            self.storage = ParquetStorage(settings.data.storage_path)
        elif self.storage_type == StorageTypes.POSTGRES.value:
            if not settings.database or not settings.database.url:
                raise ConfigurationError("PostgreSQL URL not configured for DataManager")
            self.storage = PostgresStorage(str(settings.database.url))
        else:
            raise ConfigurationError(f"Unknown storage type in DataManager: {self.storage_type}")

        # Initialisation concurrente des composants
        init_tasks = [
            self.storage.initialize(),
            self.cache_manager.initialize() if self.cache_manager else asyncio.sleep(0),
        ]
        self.binance_client = BinanceDataClient()
        init_tasks.append(self.binance_client.initialize())
        
        await asyncio.gather(*init_tasks)

        self._is_initialized = True
        logger.success("DataManager initialized successfully (storage, cache, API client)")

    async def close(self):
        """Ferme proprement toutes les connexions et ressources."""
        logger.info("Closing DataManager resources...")
        close_tasks = [
            self.storage.close() if self.storage else asyncio.sleep(0),
            self.cache_manager.close() if self.cache_manager else asyncio.sleep(0),
            self.binance_client.close() if self.binance_client else asyncio.sleep(0),
        ]
        await asyncio.gather(*close_tasks)
        self.executor.shutdown(wait=True)
        self._is_initialized = False
        logger.info("DataManager resources closed.")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def download_historical_data(
        self,
        pairs: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = Kline.INTERVAL_1MINUTE,
        process_rolling_klines: bool = True
    ) -> Dict[str, Any]:
        """
        Orchestre le téléchargement, le traitement et le stockage des données historiques pour une liste de paires.

        Args:
            pairs: Liste des paires à télécharger (ex: ["BTCUSDT", "ETHUSDT"]).
            start_date: Date de début du téléchargement.
            end_date: Date de fin. Si None, la date actuelle est utilisée.
            interval: Intervalle des klines (ex: "1m").
            process_rolling_klines: Si True, calcule et stocke les klines enrichies (rolling).

        Returns:
            Un dictionnaire résumant le succès de l'opération.
        """
        if not self._is_initialized or not self.binance_client or not self.storage:
            raise DataError("DataManager must be initialized before downloading data.")

        final_end_date = end_date or datetime.now(timezone.utc)
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(final_end_date.timestamp() * 1000)
        
        summary = {"success": True, "pairs_processed": 0, "total_klines_added": 0, "errors": []}

        for pair in pairs:
            try:
                logger.info(f"Downloading data for {pair} from {start_date} to {final_end_date}")
                klines_list = await self.binance_client.fetch_klines_batch(
                    symbol=pair, interval=interval, start_time=start_ts, end_time=end_ts
                )

                if not klines_list:
                    logger.warning(f"No data returned from API for {pair} in the specified range.")
                    continue

                df_raw = self._klines_to_dataframe(klines_list, pair)
                
                df_to_store: pd.DataFrame
                if process_rolling_klines and interval == Kline.INTERVAL_1MINUTE:
                    logger.info(f"Processing enriched (rolling) klines for {pair}...")
                    df_to_store = self.kline_processor.process_to_multi_rolling_klines(df_raw)
                else:
                    df_to_store = df_raw
                
                if df_to_store.empty:
                    logger.warning(f"DataFrame for {pair} is empty after processing. Nothing to store.")
                    continue

                await self.storage.store_klines(df_to_store, pair, interval)
                summary["pairs_processed"] += 1
                summary["total_klines_added"] += len(df_to_store)

            except Exception as e:
                logger.error(f"Failed to process pair {pair}: {e}", exc_info=True)
                summary["success"] = False
                summary["errors"].append({"pair": pair, "error": str(e)})

        logger.info(f"Download process finished. Summary: {summary}")
        return summary

    async def get_klines(
        self,
        pair: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les klines depuis le cache ou le stockage principal.
        
        Returns:
            Un DataFrame de klines ou None si aucune donnée n'est trouvée.
        """
        if not self._is_initialized or not self.storage:
            raise DataError("DataManager and its storage must be initialized.")
        
        # Gestion du cache
        cache_key = None
        if use_cache and self.cache_manager and self.cache_manager.is_available():
            key_parts = ["klines", pair, interval, str(start_time), str(end_time), str(limit)]
            cache_key = self.cache_manager.generate_key(*key_parts)
            cached_df = await self.cache_manager.get(cache_key)
            if cached_df is not None:
                logger.debug(f"Cache hit for klines: {pair}/{interval}")
                return cached_df

        # Récupération depuis le stockage si le cache est vide
        df_from_storage = await self.storage.get_klines(pair, interval, start_time, end_time, limit)

        if df_from_storage is None:
            return None

        # Mise en cache des données récupérées
        if use_cache and self.cache_manager and self.cache_manager.is_available() and cache_key:
            await self.cache_manager.set(cache_key, df_from_storage.copy())
            
        return df_from_storage

    async def get_enriched_klines(
        self,
        pair: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[EnrichedDataFrame]:
        """
        Charge les données de klines enrichies (avec vues multi-fréquences) et les retourne
        encapsulées dans un objet EnrichedDataFrame pour une utilisation sécurisée.
        """
        # Les données enrichies sont stockées comme des klines "1m" dans le stockage.
        df = await self.get_klines(
            pair=pair,
            interval=Kline.INTERVAL_1MINUTE,
            start_time=start_time,
            end_time=end_time
        )
        if df is None or df.empty:
            logger.warning(f"No enriched data found for {pair} in the specified time range.")
            return None
        
        return EnrichedDataFrame(df)

    def _klines_to_dataframe(self, klines: List[Dict[str, Any]], pair: str) -> pd.DataFrame:
        """Convertit une liste de klines de l'API en un DataFrame pandas formaté."""
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines)
        rename_map = {
            'open_time': 'timestamp', 'open': 'open', 'high': 'high',
            'low': 'low', 'close': 'close', 'volume': 'volume',
            'close_time': 'kline_close_time',
            'quote_asset_volume': 'quote_asset_volume',
            'number_of_trades': 'number_of_trades',
            'taker_buy_base_asset_volume': 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume': 'taker_buy_quote_asset_volume',
        }
        df.rename(columns=rename_map, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce').astype('Int64')

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    async def get_available_pairs(self) -> List[str]:
        """Retourne la liste des paires pour lesquelles des données sont stockées."""
        if not self._is_initialized or not self.storage:
            raise DataError("DataManager and its storage must be initialized.")
        return await self.storage.get_available_pairs()

    async def get_data_range(self, pair: str, interval: str = "1m") -> Optional[Tuple[datetime, datetime]]:
        """Retourne la plage de dates (min, max) disponible pour une paire et un intervalle."""
        if not self._is_initialized or not self.storage:
            raise DataError("DataManager and its storage must be initialized.")
        return await self.storage.get_data_range(pair, interval)

    async def update_latest_data(self, pairs: List[str], interval: str = "1m"):
        """Met à jour les données pour une liste de paires jusqu'à la date la plus récente."""
        if not self._is_initialized or not self.storage:
            raise DataError("DataManager and its storage must be initialized.")
            
        for pair in pairs:
            data_range = await self.get_data_range(pair, interval)
            start_update_date = data_range[1] + timedelta(minutes=1) if data_range else datetime(2020, 1, 1, tzinfo=timezone.utc)
            await self.download_historical_data([pair], start_update_date, interval=interval)

