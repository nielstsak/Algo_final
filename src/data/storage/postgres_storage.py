# src/data/storage/postgres_storage.py
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime, timezone # Ajout de timezone
import asyncio
from sqlalchemy import create_engine, text, select, delete, func, and_, or_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool # Pas utilisé directement mais bon à garder si la config change
from sqlalchemy.dialects.postgresql import insert
from loguru import logger

from src.core.config import settings
from src.core.constants import Kline
from src.core.exceptions import PostgresStorageError, StorageError
from src.core.database.models import Kline1M, Base # Assurez-vous que Base est importé si utilisé pour create_all
from src.data.storage.base import BaseStorage


class PostgresStorage(BaseStorage):
    """
    Implémentation du stockage en base de données PostgreSQL.
    Utilise SQLAlchemy avec support asynchrone pour les opérations.
    Cette version est adaptée pour extraire les données K_1m_* d'un DataFrame potentiellement large.
    """
    
    def __init__(self, database_url: Union[str, Any]):
        """
        Initialise le stockage PostgreSQL.
        
        Args:
            database_url: URL de connexion à la base de données
        """
        if isinstance(database_url, str):
            self.database_url = database_url
            if self.database_url.startswith("postgresql://"):
                self.async_database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
            elif self.database_url.startswith("postgresql+psycopg2://"): # Common sync driver
                self.async_database_url = self.database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
            elif not self.database_url.startswith("postgresql+asyncpg://"): # If some other dialect, ensure it's for asyncpg
                 self.async_database_url = f"postgresql+asyncpg://{self.database_url.split('://', 1)[-1]}" if '://' in self.database_url else f"postgresql+asyncpg://{self.database_url}"
            else: # Already asyncpg
                self.async_database_url = self.database_url
        else: # Si c'est déjà un objet URL SQLAlchemy (moins probable pour config)
            self.database_url = str(database_url)
            self.async_database_url = self.database_url.replace("psycopg2", "asyncpg").replace("postgresql://", "postgresql+asyncpg://")
            
        self.engine: Optional[AsyncEngine] = None
        self.async_session_maker: Optional[sessionmaker] = None # type: ignore
        self._initialized = False
        
        logger.info(f"PostgresStorage initialized with async URL: {self.async_database_url}")
        
    async def initialize(self) -> None:
        """Initialise la connexion à la base de données et crée les tables si elles n'existent pas."""
        try:
            self.engine = create_async_engine(
                self.async_database_url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout_seconds,
                pool_recycle=settings.database.pool_recycle_seconds,
                pool_pre_ping=settings.database.pool_pre_ping_enabled,
                echo=settings.database.echo_sql # Utiliser le paramètre de configuration
            )
            
            self.async_session_maker = sessionmaker( # type: ignore
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Créer les tables (ex: klines_1m) si elles n'existent pas.
            # Ceci est généralement fait une fois au démarrage de l'application.
            async with self.engine.begin() as conn:
                # await conn.run_sync(Base.metadata.drop_all) # Pour tester la recréation
                await conn.run_sync(Base.metadata.create_all)
            
            # Tester la connexion
            async with self.engine.connect() as conn: # Utiliser connect() pour un test simple
                await conn.execute(text("SELECT 1"))
                
            self._initialized = True
            logger.success("PostgresStorage initialized, connected, and tables ensured.")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgresStorage: {e}", exc_info=True)
            # Garder l'engine et session_maker à None si l'initialisation échoue
            self.engine = None
            self.async_session_maker = None
            self._initialized = False
            raise StorageError(
                f"Failed to initialize PostgresStorage: {e}",
                original_exception=e
            )
            
    async def close(self) -> None:
        """Ferme proprement les connexions."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.async_session_maker = None
            self._initialized = False
            logger.info("PostgresStorage connections closed")
            
    def _validate_initialization(self):
        """Vérifie que le storage est initialisé."""
        if not self._initialized or not self.engine or not self.async_session_maker:
            raise StorageError("PostgresStorage not initialized. Call initialize() first.")
            
    def _extract_1m_data_from_wide_df(self, df_wide: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Extrait et renomme les colonnes K_1m_* d'un DataFrame large pour correspondre
        au schéma de la table klines_1m.
        L'index du df_wide est supposé être 'timestamp' (open_time de la kline 1m).
        """
        
        # Colonnes attendues par la table klines_1m (noms de modèle/BDD) et leur source K_1m_*
        # Le 'timestamp' de df_wide (index après reset) est kline_open_time.
        cols_map = {
            # 'kline_open_time': 'timestamp', # Provient de l'index de df_wide
            'open_price': f"K_1m_{Kline.OHLCV_OPEN}",
            'high_price': f"K_1m_{Kline.OHLCV_HIGH}",
            'low_price': f"K_1m_{Kline.OHLCV_LOW}",
            'close_price': f"K_1m_{Kline.OHLCV_CLOSE}",
            'base_asset_volume': f"K_1m_{Kline.OHLCV_VOLUME}",
            'kline_close_time': f"K_1m_{Kline.OHLCV_KLINE_CLOSE_TIME}",
            'quote_asset_volume': f"K_1m_{Kline.OHLCV_QUOTE_ASSET_VOLUME}",
            'number_of_trades': f"K_1m_{Kline.OHLCV_NUMBER_OF_TRADES}",
            'taker_buy_base_asset_volume': f"K_1m_{Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME}",
            'taker_buy_quote_asset_volume': f"K_1m_{Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME}",
            'is_kline_closed': f"K_1m_{Kline.OHLCV_IS_KLINE_CLOSED}",
        }

        df_1m_only = pd.DataFrame()
        
        # Gérer l'index 'timestamp'
        current_df = df_wide.copy()
        if isinstance(current_df.index, pd.DatetimeIndex) and \
           (current_df.index.name == 'timestamp' or Kline.OHLCV_TIMESTAMP in str(current_df.index.name)):
            current_df.reset_index(inplace=True) # 'timestamp' devient une colonne

        if 'timestamp' not in current_df.columns:
            logger.error("Wide DataFrame does not have 'timestamp' column after potential index reset.")
            return pd.DataFrame() # Retourner vide si pas de timestamp
        
        df_1m_only['kline_open_time'] = current_df['timestamp']


        # Extraire et renommer les colonnes
        for db_col, source_col in cols_map.items():
            if source_col in current_df.columns:
                df_1m_only[db_col] = current_df[source_col]
            else:
                # Certaines colonnes peuvent être optionnelles ou avoir des valeurs par défaut dans le modèle
                logger.debug(f"Source column '{source_col}' for DB column '{db_col}' not found in wide DataFrame. Will rely on DB default or NaN if applicable.")
                df_1m_only[db_col] = np.nan # Mettre NaN si la colonne source manque

        # Ajouter la colonne 'pair'
        df_1m_only['pair'] = pair
        
        # S'assurer que les types sont corrects pour la conversion en records
        if 'kline_open_time' in df_1m_only.columns:
            df_1m_only['kline_open_time'] = pd.to_datetime(df_1m_only['kline_open_time'], utc=True, errors='coerce')
        if 'kline_close_time' in df_1m_only.columns:
            df_1m_only['kline_close_time'] = pd.to_datetime(df_1m_only['kline_close_time'], utc=True, errors='coerce')

        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 
                        'base_asset_volume', 'quote_asset_volume', 
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            if col in df_1m_only.columns:
                df_1m_only[col] = pd.to_numeric(df_1m_only[col], errors='coerce').astype(float) # float pour gérer NaNs

        if 'number_of_trades' in df_1m_only.columns:
             df_1m_only['number_of_trades'] = pd.to_numeric(df_1m_only['number_of_trades'], errors='coerce').astype('Int64') # Int64 pour gérer NaNs

        if 'is_kline_closed' in df_1m_only.columns:
            df_1m_only['is_kline_closed'] = df_1m_only['is_kline_closed'].astype('boolean') # Pandas nullable boolean

        # Supprimer les lignes où kline_open_time est NaT (résultat de coerce par exemple)
        df_1m_only.dropna(subset=['kline_open_time'], inplace=True)

        return df_1m_only


    def _dataframe_to_kline_models(self, df_1m_records: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convertit un DataFrame (déjà formaté pour klines_1m) en liste de dictionnaires pour insertion.
        """
        records = []
        for _, row in df_1m_records.iterrows():
            record = row.to_dict()
            # Remplacer les NaNs par None pour la base de données, et s'assurer que les types sont corrects
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                # Les types devraient déjà être bons grâce à _extract_1m_data_from_wide_df
                # mais une vérification supplémentaire pourrait être ajoutée ici si nécessaire.
            records.append(record)
        return records
        
    async def store_klines(
        self,
        df: pd.DataFrame, # Peut être le DataFrame large issu de KlineProcessor
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> None:
        self._validate_initialization()
        
        if interval != Kline.INTERVAL_1MINUTE:
            logger.warning(f"PostgreSQL storage currently only processes and stores 1m interval data from input. "
                           f"Requested interval was {interval}, but only K_1m_* columns will be used.")
            # On ne lève pas d'erreur, on essaie d'extraire les données 1m.
        
        if df.empty:
            logger.debug(f"Input DataFrame is empty for {pair}. Nothing to store in PostgreSQL.")
            return

        try:
            # Extraire les données 1m du DataFrame potentiellement large
            df_1m_to_store = self._extract_1m_data_from_wide_df(df, pair)

            if df_1m_to_store.empty:
                logger.warning(f"No 1-minute data could be extracted from the input DataFrame for {pair}. Nothing stored in PostgreSQL.")
                return
                
            records_to_insert = self._dataframe_to_kline_models(df_1m_to_store)
            
            if not records_to_insert:
                logger.warning(f"No valid records to store in PostgreSQL for {pair} after processing.")
                return
                
            async with self.async_session_maker() as session: # type: ignore
                stmt = insert(Kline1M).values(records_to_insert)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Kline1M.pair, Kline1M.kline_open_time], # Assurez-vous que ce sont les bonnes colonnes de contrainte unique
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
                
            logger.info(f"Stored/Updated {len(records_to_insert)} 1-minute klines in PostgreSQL for {pair}")
            
        except Exception as e:
            logger.error(f"Failed to store 1-minute klines in PostgreSQL for {pair}: {e}", exc_info=True)
            # Ne pas relancer PostgresStorageError si c'est déjà une, sinon l'envelopper.
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to store 1-minute klines: {e}", original_exception=e)
            
    async def get_klines(
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        self._validate_initialization()
        
        if interval != Kline.INTERVAL_1MINUTE:
            logger.warning(f"PostgreSQL storage currently only supports fetching 1m interval, got {interval}. Returning None or error.")
            # Selon la politique, on pourrait retourner None ou lever une NotImplementedError.
            # Pour être cohérent avec le message, on peut lever une erreur.
            raise NotImplementedError(f"PostgreSQL storage currently only supports fetching 1m interval, got {interval}")
            
        try:
            async with self.async_session_maker() as session: # type: ignore
                stmt = select(Kline1M).where(Kline1M.pair == pair)
                
                if start_time:
                    if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
                    stmt = stmt.where(Kline1M.kline_open_time >= start_time)
                if end_time:
                    if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
                    stmt = stmt.where(Kline1M.kline_open_time <= end_time)
                    
                stmt = stmt.order_by(Kline1M.kline_open_time.asc()) # Ou .desc() si on veut les plus récents avec limit
                
                if limit is not None:
                    # Si on veut les N plus récents, il faut trier par ordre décroissant puis limiter,
                    # puis retrier par ordre croissant si nécessaire.
                    # Pour un simple tail(limit) sur des données déjà triées asc, on lit tout puis on prend le tail.
                    # Ou, si on veut les N plus récents, on fait order_by(desc).limit(N) puis on re-sort asc.
                    # Pour l'instant, on assume que limit prend les N premières lignes après le tri ascendant.
                    # Si on veut les N plus récents, la requête devrait être:
                    # subquery = select(Kline1M).where(...).order_by(Kline1M.kline_open_time.desc()).limit(limit).subquery()
                    # query = select(subquery).order_by(subquery.c.kline_open_time.asc())
                    # Pour simplifier, si limit est utilisé, on récupère les N plus récents en triant desc.
                    if limit > 0 : # S'assurer que limit est positif
                        stmt = stmt.order_by(Kline1M.kline_open_time.desc()).limit(limit)
                        # On devra retrier en asc après avoir récupéré les données.
                
                result = await session.execute(stmt)
                klines_models = result.scalars().all()
                
                if not klines_models:
                    logger.debug(f"No klines found in PostgreSQL for {pair} with specified criteria.")
                    return None
                    
                data = [{
                    # Utiliser Kline.OHLCV_TIMESTAMP pour le nom de la colonne d'index dans le DataFrame
                    Kline.OHLCV_TIMESTAMP: k.kline_open_time.replace(tzinfo=timezone.utc), # Assurer UTC
                    Kline.OHLCV_OPEN: float(k.open_price),
                    Kline.OHLCV_HIGH: float(k.high_price),
                    Kline.OHLCV_LOW: float(k.low_price),
                    Kline.OHLCV_CLOSE: float(k.close_price),
                    Kline.OHLCV_VOLUME: float(k.base_asset_volume),
                    Kline.OHLCV_KLINE_CLOSE_TIME: k.kline_close_time.replace(tzinfo=timezone.utc) if k.kline_close_time else None,
                    Kline.OHLCV_QUOTE_ASSET_VOLUME: float(k.quote_asset_volume) if k.quote_asset_volume is not None else 0.0,
                    Kline.OHLCV_NUMBER_OF_TRADES: int(k.number_of_trades) if k.number_of_trades is not None else 0,
                    Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: float(k.taker_buy_base_asset_volume) if k.taker_buy_base_asset_volume is not None else 0.0,
                    Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: float(k.taker_buy_quote_asset_volume) if k.taker_buy_quote_asset_volume is not None else 0.0,
                    Kline.OHLCV_IS_KLINE_CLOSED: bool(k.is_kline_closed) if k.is_kline_closed is not None else True,
                    Kline.PAIR: k.pair
                } for k in klines_models]
                    
                df = pd.DataFrame(data)
                if df.empty: return None

                df.set_index(Kline.OHLCV_TIMESTAMP, inplace=True)
                df.sort_index(inplace=True) # Assurer l'ordre croissant si on a fait un tri desc pour le limit
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get klines from PostgreSQL for {pair}: {e}", exc_info=True)
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to get klines: {e}", original_exception=e)
            
    async def get_available_pairs(self) -> List[str]:
        self._validate_initialization()
        try:
            async with self.async_session_maker() as session: # type: ignore
                query = select(Kline1M.pair).distinct()
                result = await session.execute(query)
                pairs = [row[0] for row in result]
                return sorted(pairs)
        except Exception as e:
            logger.error(f"Failed to get available pairs from PostgreSQL: {e}", exc_info=True)
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to get available pairs: {e}", original_exception=e)
            
    async def get_data_range(
        self,
        pair: str,
        interval: str = Kline.INTERVAL_1MINUTE
    ) -> Optional[Tuple[datetime, datetime]]:
        self._validate_initialization()
        if interval != Kline.INTERVAL_1MINUTE:
            raise NotImplementedError(f"PostgreSQL storage currently only supports 1m interval for data range, got {interval}")
            
        try:
            async with self.async_session_maker() as session: # type: ignore
                query = select(
                    func.min(Kline1M.kline_open_time).label("min_time"), # type: ignore
                    func.max(Kline1M.kline_open_time).label("max_time")  # type: ignore
                ).where(Kline1M.pair == pair)
                
                result = await session.execute(query)
                row = result.first() # type: ignore
                
                if row and row.min_time is not None and row.max_time is not None:
                    min_time = row.min_time.replace(tzinfo=timezone.utc) if row.min_time.tzinfo is None else row.min_time.astimezone(timezone.utc)
                    max_time = row.max_time.replace(tzinfo=timezone.utc) if row.max_time.tzinfo is None else row.max_time.astimezone(timezone.utc)
                    return (min_time, max_time)
                return None
        except Exception as e:
            logger.error(f"Failed to get data range from PostgreSQL for {pair}: {e}", exc_info=True)
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to get data range: {e}", original_exception=e)
            
    async def delete_old_data(
        self,
        cutoff_date: datetime
    ) -> Dict[str, int]:
        self._validate_initialization()
        if cutoff_date.tzinfo is None: cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
        
        deleted_counts: Dict[str, int] = {}
        try:
            pairs = await self.get_available_pairs()
            async with self.async_session_maker() as session: # type: ignore
                for pair_item in pairs:
                    # Compter d'abord (optionnel, delete retourne le nombre de lignes affectées dans certains dialectes)
                    # Pour être sûr, on peut faire un count explicite.
                    count_stmt = select(func.count(Kline1M.id)).where( # type: ignore
                        and_(Kline1M.pair == pair_item, Kline1M.kline_open_time < cutoff_date)
                    )
                    result_count = await session.execute(count_stmt)
                    num_to_delete = result_count.scalar_one_or_none() or 0 # type: ignore

                    if num_to_delete > 0:
                        delete_stmt = delete(Kline1M).where(
                            and_(Kline1M.pair == pair_item, Kline1M.kline_open_time < cutoff_date)
                        )
                        # result_delete = await session.execute(delete_stmt) # SQLAlchemy 2.0 ne retourne plus rowcount directement pour ORM delete
                        # Pour obtenir le nombre de lignes supprimées, il faut soit un count avant, soit une fonctionnalité spécifique au dialecte.
                        # On utilise le count_stmt déjà fait.
                        await session.execute(delete_stmt) # Exécuter la suppression
                        deleted_counts[pair_item] = num_to_delete
                        logger.info(f"Deleted {num_to_delete} old klines from PostgreSQL for {pair_item} older than {cutoff_date}")
                await session.commit()
            return deleted_counts
        except Exception as e:
            logger.error(f"Failed to delete old data from PostgreSQL: {e}", exc_info=True)
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to delete old data: {e}", original_exception=e)
            
    async def get_statistics(self, pair: str, interval: str = Kline.INTERVAL_1MINUTE) -> Dict[str, Any]:
        self._validate_initialization()
        if interval != Kline.INTERVAL_1MINUTE:
            raise NotImplementedError(f"PostgreSQL storage currently only supports 1m interval for statistics, got {interval}")
            
        try:
            async with self.async_session_maker() as session: # type: ignore
                query = select(
                    func.count(Kline1M.id).label('count'), # type: ignore
                    func.min(Kline1M.kline_open_time).label('min_time'), # type: ignore
                    func.max(Kline1M.kline_open_time).label('max_time'), # type: ignore
                    func.avg(Kline1M.close_price).label('avg_price'), # type: ignore
                    func.min(Kline1M.low_price).label('min_price'), # type: ignore
                    func.max(Kline1M.high_price).label('max_price'), # type: ignore
                    func.sum(Kline1M.base_asset_volume).label('total_volume'), # type: ignore
                    func.avg(Kline1M.base_asset_volume).label('avg_volume') # type: ignore
                ).where(Kline1M.pair == pair)
                
                result = await session.execute(query)
                row = result.first() # type: ignore
                
                if not row or row.count == 0: return {"exists": False, "pair": pair, "interval": interval}
                    
                min_t = row.min_time.replace(tzinfo=timezone.utc) if row.min_time and row.min_time.tzinfo is None else (row.min_time.astimezone(timezone.utc) if row.min_time else None)
                max_t = row.max_time.replace(tzinfo=timezone.utc) if row.max_time and row.max_time.tzinfo is None else (row.max_time.astimezone(timezone.utc) if row.max_time else None)

                return {
                    "exists": True, "pair": pair, "interval": interval,
                    "count": row.count,
                    "start_date": min_t,
                    "end_date": max_t,
                    "duration_days": (max_t - min_t).days if min_t and max_t else 0,
                    "avg_price": float(row.avg_price) if row.avg_price is not None else 0.0,
                    "min_price": float(row.min_price) if row.min_price is not None else 0.0,
                    "max_price": float(row.max_price) if row.max_price is not None else 0.0,
                    "total_volume": float(row.total_volume) if row.total_volume is not None else 0.0,
                    "avg_volume": float(row.avg_volume) if row.avg_volume is not None else 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get statistics from PostgreSQL for {pair}: {e}", exc_info=True)
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to get statistics: {e}", original_exception=e)
            
    async def optimize_storage(self) -> Dict[str, Any]:
        self._validate_initialization()
        stats: Dict[str, Any] = {"tables_optimized": 0, "time_taken_seconds": 0.0, "vacuum_details": {}}
        start_time_opt = datetime.now(timezone.utc)
        
        try:
            async with self.engine.connect() as conn: # type: ignore
                # Utiliser une transaction pour les commandes VACUUM si nécessaire, bien que VACUUM soit souvent auto-commit.
                # await conn.execute(text("VACUUM FULL ANALYZE klines_1m")) # VACUUM FULL bloque la table
                await conn.execute(text("VACUUM ANALYZE klines_1m")) # Non bloquant pour les lectures/écritures concurrentes
                await conn.commit() # Assurer que la transaction est terminée
                stats["tables_optimized"] = 1
                
                # Récupérer des statistiques après optimisation
                # Note: pg_stat_user_tables n'est pas toujours mis à jour instantanément après VACUUM.
                # Les informations de taille sont plus fiables.
                result_size = await conn.execute(
                    text("SELECT pg_size_pretty(pg_total_relation_size('klines_1m')) as total_size;")
                )
                size_info = result_size.first() # type: ignore
                if size_info: stats["vacuum_details"]["klines_1m_total_size"] = size_info.total_size
                    
            stats["time_taken_seconds"] = (datetime.now(timezone.utc) - start_time_opt).total_seconds()
            logger.info(
                f"PostgreSQL optimization (VACUUM ANALYZE klines_1m) completed in {stats['time_taken_seconds']:.2f}s. "
                f"Size: {stats.get('vacuum_details', {}).get('klines_1m_total_size', 'N/A')}"
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to optimize PostgreSQL storage: {e}", exc_info=True)
            if isinstance(e, PostgresStorageError): raise
            raise PostgresStorageError(f"Failed to optimize storage: {e}", original_exception=e)

