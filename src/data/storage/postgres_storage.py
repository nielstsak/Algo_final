# src/data/storage/postgres_storage.py
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime
import asyncio
from sqlalchemy import create_engine, text, select, delete, func, and_, or_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import insert
from loguru import logger

from src.core.config import settings
from src.core.constants import Kline
from src.core.exceptions import PostgresStorageError, StorageError
from src.core.database.models import Kline1M, Base
from src.data.storage.base import BaseStorage


class PostgresStorage(BaseStorage):
    """
    Implémentation du stockage en base de données PostgreSQL.
    Utilise SQLAlchemy avec support asynchrone pour les opérations.
    """
    
    def __init__(self, database_url: Union[str, Any]):
        """
        Initialise le stockage PostgreSQL.
        
        Args:
            database_url: URL de connexion à la base de données
        """
        # Convertir l'URL pour le support asynchrone
        if isinstance(database_url, str):
            self.database_url = database_url
            # Remplacer le driver pour asyncpg
            if self.database_url.startswith("postgresql://"):
                self.async_database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
            elif self.database_url.startswith("postgresql+psycopg2://"):
                self.async_database_url = self.database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
            else:
                self.async_database_url = self.database_url
        else:
            # Si c'est déjà un objet URL SQLAlchemy
            self.database_url = str(database_url)
            self.async_database_url = self.database_url.replace("psycopg2", "asyncpg")
            
        self.engine: Optional[AsyncEngine] = None
        self.async_session_maker: Optional[sessionmaker] = None
        self._initialized = False
        
        logger.info("PostgresStorage initialized")
        
    async def initialize(self) -> None:
        """Initialise la connexion à la base de données."""
        try:
            # Créer le moteur asynchrone
            self.engine = create_async_engine(
                self.async_database_url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout_seconds,
                pool_recycle=settings.database.pool_recycle_seconds,
                pool_pre_ping=settings.database.pool_pre_ping_enabled,
                echo=False  # Mettre à True pour debug SQL
            )
            
            # Créer la session factory
            self.async_session_maker = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Tester la connexion
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                
            self._initialized = True
            logger.success("PostgresStorage initialized and connected")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgresStorage: {e}")
            raise StorageError(
                f"Failed to initialize PostgresStorage: {e}",
                original_exception=e
            )
            
    async def close(self) -> None:
        """Ferme proprement les connexions."""
        if self.engine:
            await self.engine.dispose()
            logger.info("PostgresStorage connections closed")
            
    def _validate_initialization(self):
        """Vérifie que le storage est initialisé."""
        if not self._initialized or not self.engine or not self.async_session_maker:
            raise StorageError("PostgresStorage not initialized. Call initialize() first.")
            
    def _dataframe_to_kline_models(self, df: pd.DataFrame, pair: str) -> List[Dict[str, Any]]:
        """
        Convertit un DataFrame en liste de dictionnaires pour insertion.
        
        Args:
            df: DataFrame contenant les klines
            pair: Paire de trading
            
        Returns:
            Liste de dictionnaires représentant les klines
        """
        records = []
        
        # Copier et préparer le DataFrame
        df_copy = df.copy()
        
        # S'assurer que l'index est dans une colonne
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
            if 'index' in df_copy.columns and 'kline_open_time' not in df_copy.columns:
                df_copy.rename(columns={'index': 'kline_open_time'}, inplace=True)
                
        # Convertir chaque ligne en dictionnaire
        for _, row in df_copy.iterrows():
            record = {
                'pair': pair,
                'kline_open_time': pd.to_datetime(row['kline_open_time'], utc=True),
                'kline_close_time': pd.to_datetime(row.get('kline_close_time', row['kline_open_time'] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)), utc=True),
                'open_price': float(row['open_price']),
                'high_price': float(row['high_price']),
                'low_price': float(row['low_price']),
                'close_price': float(row['close_price']),
                'base_asset_volume': float(row['base_asset_volume']),
                'quote_asset_volume': float(row.get('quote_asset_volume', 0)),
                'number_of_trades': int(row.get('number_of_trades', 0)),
                'taker_buy_base_asset_volume': float(row.get('taker_buy_base_asset_volume', 0)),
                'taker_buy_quote_asset_volume': float(row.get('taker_buy_quote_asset_volume', 0)),
                'is_kline_closed': bool(row.get('is_kline_closed', True))
            }
            records.append(record)
            
        return records
        
    async def store_klines(
        self,
        df: pd.DataFrame,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> None:
        """
        Stocke les klines dans PostgreSQL.
        
        Args:
            df: DataFrame contenant les klines
            pair: Paire de trading
            interval: Intervalle des klines (actuellement seul 1m est supporté)
        """
        self._validate_initialization()
        
        if interval != Kline.INTERVAL_1MINUTE:
            raise NotImplementedError(f"PostgreSQL storage currently only supports 1m interval, got {interval}")
            
        try:
            # Convertir le DataFrame en records
            records = self._dataframe_to_kline_models(df, pair)
            
            if not records:
                logger.warning(f"No records to store for {pair}")
                return
                
            async with self.async_session_maker() as session:
                # Utiliser INSERT ... ON CONFLICT pour gérer les doublons
                stmt = insert(Kline1M).values(records)
                
                # En cas de conflit sur (pair, kline_open_time), mettre à jour
                stmt = stmt.on_conflict_do_update(
                    index_elements=['pair', 'kline_open_time'],
                    set_={
                        'open_price': stmt.excluded.open_price,
                        'high_price': stmt.excluded.high_price,
                        'low_price': stmt.excluded.low_price,
                        'close_price': stmt.excluded.close_price,
                        'base_asset_volume': stmt.excluded.base_asset_volume,
                        'quote_asset_volume': stmt.excluded.quote_asset_volume,
                        'number_of_trades': stmt.excluded.number_of_trades,
                        'taker_buy_base_asset_volume': stmt.excluded.taker_buy_base_asset_volume,
                        'taker_buy_quote_asset_volume': stmt.excluded.taker_buy_quote_asset_volume,
                        'is_kline_closed': stmt.excluded.is_kline_closed,
                        'kline_close_time': stmt.excluded.kline_close_time
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
                
            logger.info(f"Stored {len(records)} klines for {pair}")
            
        except Exception as e:
            logger.error(f"Failed to store klines for {pair}: {e}")
            raise PostgresStorageError(
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
        Récupère les klines depuis PostgreSQL.
        
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
        
        if interval != Kline.INTERVAL_1MINUTE:
            raise NotImplementedError(f"PostgreSQL storage currently only supports 1m interval, got {interval}")
            
        try:
            async with self.async_session_maker() as session:
                # Construire la requête
                query = select(Kline1M).where(Kline1M.pair == pair)
                
                # Ajouter les filtres temporels
                if start_time is not None:
                    query = query.where(Kline1M.kline_open_time >= start_time)
                if end_time is not None:
                    query = query.where(Kline1M.kline_open_time <= end_time)
                    
                # Trier par temps
                query = query.order_by(Kline1M.kline_open_time)
                
                # Appliquer la limite
                if limit is not None:
                    query = query.limit(limit)
                    
                # Exécuter la requête
                result = await session.execute(query)
                klines = result.scalars().all()
                
                if not klines:
                    logger.debug(f"No klines found for {pair}")
                    return None
                    
                # Convertir en DataFrame
                data = []
                for kline in klines:
                    data.append({
                        'kline_open_time': kline.kline_open_time,
                        'kline_close_time': kline.kline_close_time,
                        'open_price': float(kline.open_price),
                        'high_price': float(kline.high_price),
                        'low_price': float(kline.low_price),
                        'close_price': float(kline.close_price),
                        'base_asset_volume': float(kline.base_asset_volume),
                        'quote_asset_volume': float(kline.quote_asset_volume),
                        'number_of_trades': kline.number_of_trades,
                        'taker_buy_base_asset_volume': float(kline.taker_buy_base_asset_volume),
                        'taker_buy_quote_asset_volume': float(kline.taker_buy_quote_asset_volume),
                        'is_kline_closed': kline.is_kline_closed,
                        'pair': kline.pair
                    })
                    
                df = pd.DataFrame(data)
                df.set_index('kline_open_time', inplace=True)
                df.sort_index(inplace=True)
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get klines for {pair}: {e}")
            raise PostgresStorageError(
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
            async with self.async_session_maker() as session:
                # Requête pour obtenir les paires distinctes
                query = select(Kline1M.pair).distinct()
                result = await session.execute(query)
                pairs = [row[0] for row in result]
                
                return sorted(pairs)
                
        except Exception as e:
            logger.error(f"Failed to get available pairs: {e}")
            raise PostgresStorageError(
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
        
        if interval != Kline.INTERVAL_1MINUTE:
            raise NotImplementedError(f"PostgreSQL storage currently only supports 1m interval, got {interval}")
            
        try:
            async with self.async_session_maker() as session:
                # Requête pour obtenir min et max timestamp
                query = select(
                    func.min(Kline1M.kline_open_time),
                    func.max(Kline1M.kline_open_time)
                ).where(Kline1M.pair == pair)
                
                result = await session.execute(query)
                row = result.first()
                
                if row and row[0] is not None and row[1] is not None:
                    return (row[0], row[1])
                    
                return None
                
        except Exception as e:
            logger.error(f"Failed to get data range for {pair}: {e}")
            raise PostgresStorageError(
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
            
            # Obtenir d'abord les paires disponibles
            pairs = await self.get_available_pairs()
            
            async with self.async_session_maker() as session:
                for pair in pairs:
                    # Compter les lignes à supprimer
                    count_query = select(func.count()).where(
                        and_(
                            Kline1M.pair == pair,
                            Kline1M.kline_open_time < cutoff_date
                        )
                    )
                    result = await session.execute(count_query)
                    count = result.scalar()
                    
                    if count > 0:
                        # Supprimer les lignes
                        delete_stmt = delete(Kline1M).where(
                            and_(
                                Kline1M.pair == pair,
                                Kline1M.kline_open_time < cutoff_date
                            )
                        )
                        await session.execute(delete_stmt)
                        deleted_counts[pair] = count
                        logger.info(f"Deleted {count} old klines for {pair}")
                        
                await session.commit()
                
            return deleted_counts
            
        except Exception as e:
            logger.error(f"Failed to delete old data: {e}")
            raise PostgresStorageError(
                f"Failed to delete old data: {e}",
                original_exception=e
            )
            
    async def get_statistics(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Dict[str, Any]:
        """
        Retourne des statistiques détaillées pour une paire.
        
        Args:
            pair: Paire de trading
            interval: Intervalle des klines
            
        Returns:
            Dictionnaire avec les statistiques
        """
        self._validate_initialization()
        
        if interval != Kline.INTERVAL_1MINUTE:
            raise NotImplementedError(f"PostgreSQL storage currently only supports 1m interval, got {interval}")
            
        try:
            async with self.async_session_maker() as session:
                # Requête pour obtenir les statistiques
                query = select(
                    func.count(Kline1M.id).label('count'),
                    func.min(Kline1M.kline_open_time).label('min_time'),
                    func.max(Kline1M.kline_open_time).label('max_time'),
                    func.avg(Kline1M.close_price).label('avg_price'),
                    func.min(Kline1M.low_price).label('min_price'),
                    func.max(Kline1M.high_price).label('max_price'),
                    func.sum(Kline1M.base_asset_volume).label('total_volume'),
                    func.avg(Kline1M.base_asset_volume).label('avg_volume')
                ).where(Kline1M.pair == pair)
                
                result = await session.execute(query)
                row = result.first()
                
                if not row or row.count == 0:
                    return {"exists": False}
                    
                return {
                    "exists": True,
                    "count": row.count,
                    "start_date": row.min_time,
                    "end_date": row.max_time,
                    "duration_days": (row.max_time - row.min_time).days if row.min_time and row.max_time else 0,
                    "avg_price": float(row.avg_price) if row.avg_price else 0,
                    "min_price": float(row.min_price) if row.min_price else 0,
                    "max_price": float(row.max_price) if row.max_price else 0,
                    "total_volume": float(row.total_volume) if row.total_volume else 0,
                    "avg_volume": float(row.avg_volume) if row.avg_volume else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics for {pair}: {e}")
            raise PostgresStorageError(
                f"Failed to get statistics: {e}",
                original_exception=e
            )
            
    async def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimise les tables PostgreSQL (VACUUM, ANALYZE).
        
        Returns:
            Statistiques d'optimisation
        """
        self._validate_initialization()
        
        stats = {
            "tables_optimized": 0,
            "time_taken_seconds": 0
        }
        
        start_time = datetime.now()
        
        try:
            async with self.engine.begin() as conn:
                # VACUUM et ANALYZE sur la table klines_1m
                await conn.execute(text("VACUUM ANALYZE klines_1m"))
                stats["tables_optimized"] = 1
                
                # Récupérer les statistiques de la table
                result = await conn.execute(
                    text("""
                        SELECT 
                            pg_size_pretty(pg_total_relation_size('klines_1m')) as total_size,
                            pg_size_pretty(pg_relation_size('klines_1m')) as table_size,
                            pg_size_pretty(pg_indexes_size('klines_1m')) as indexes_size,
                            n_live_tup as row_count,
                            n_dead_tup as dead_rows
                        FROM pg_stat_user_tables 
                        WHERE relname = 'klines_1m'
                    """)
                )
                
                row = result.first()
                if row:
                    stats.update({
                        "total_size": row.total_size,
                        "table_size": row.table_size,
                        "indexes_size": row.indexes_size,
                        "row_count": row.row_count,
                        "dead_rows_before": row.dead_rows
                    })
                    
            stats["time_taken_seconds"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"PostgreSQL optimization completed in {stats['time_taken_seconds']:.2f}s. "
                f"Table size: {stats.get('total_size', 'N/A')}"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to optimize PostgreSQL storage: {e}")
            raise PostgresStorageError(
                f"Failed to optimize storage: {e}",
                original_exception=e
            )