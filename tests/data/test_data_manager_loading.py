# tests/data/test_data_manager_loading.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, Optional
import asyncio
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from src.data.data_manager import DataManager
from src.data.storage.parquet_storage import ParquetStorage 
from src.data.storage.postgres_storage import PostgresStorage 
from src.core.config import settings
from src.core.constants import Kline, StorageTypes, System # Import System
from src.core.exceptions import (
    DataError,
    StorageError,
    ConfigurationError,
    ParquetStorageError,
    PostgresStorageError
)


# Fixtures
@pytest.fixture
def mock_settings_parquet(monkeypatch, tmp_path): # tmp_path pour un chemin unique
    """Configure settings pour utiliser Parquet."""
    monkeypatch.setattr(settings.data, 'storage_type', StorageTypes.PARQUET.value)
    monkeypatch.setattr(settings.data, 'cache_enabled', False)
    monkeypatch.setattr(settings.data, 'storage_path', tmp_path / "test_parquet_storage")
    (tmp_path / "test_parquet_storage").mkdir(parents=True, exist_ok=True)
    # S'assurer que le répertoire de checkpoint existe aussi sous le storage_path
    (tmp_path / "test_parquet_storage" / ".dm_checkpoints").mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def mock_settings_postgres(monkeypatch):
    """Configure settings pour utiliser PostgreSQL."""
    monkeypatch.setattr(settings.data, 'storage_type', StorageTypes.POSTGRES.value)
    monkeypatch.setattr(settings.data, 'cache_enabled', False)
    # Assurer qu'une URL de DB est définie pour Postgres
    if not hasattr(settings.database, 'url') or not settings.database.url:
        monkeypatch.setattr(settings.database, 'url', "postgresql://user:pass@host:port/testdb")
    return settings


@pytest.fixture
def sample_raw_klines_data_for_storage() -> pd.DataFrame:
    """
    Crée des données de test brutes (similaires à ce que le stockage pourrait contenir)
    AVANT la normalisation par DataManager._normalize_storage_dataframe.
    L'index est Kline.OHLCV_TIMESTAMP et les colonnes sont 'open_price', etc.
    """
    start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Augmenter le nombre de périodes pour les tests de fréquence
    periods = 100
    dates = pd.date_range(start=start_time, periods=periods, freq='1min', tz='UTC') 
    
    base_price = 40000 
    price_variation = np.random.uniform(-0.001, 0.001, periods)
    cumulative_change = np.cumprod(1 + price_variation)
    close_prices = base_price * cumulative_change
    
    df = pd.DataFrame({
        'open_price': close_prices * np.random.uniform(0.999, 1.001, periods),
        'high_price': close_prices * np.random.uniform(1.001, 1.003, periods),
        'low_price': close_prices * np.random.uniform(0.997, 0.999, periods),
        'close_price': close_prices,
        'base_asset_volume': np.random.uniform(0.1, 1.0, periods),
        'pair': 'BTCUSDC', 
        'kline_close_time': dates + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
        'quote_asset_volume': close_prices * np.random.uniform(0.1, 1.0, periods),
        'number_of_trades': np.random.randint(10, 100, periods),
        'taker_buy_base_asset_volume': np.random.uniform(0.05, 0.5, periods),
        'taker_buy_quote_asset_volume': close_prices * np.random.uniform(0.05, 0.5, periods),
        'is_kline_closed': True
    })
    
    df['high_price'] = df[['open_price', 'high_price', 'close_price']].max(axis=1)
    df['low_price'] = df[['open_price', 'low_price', 'close_price']].min(axis=1)
    
    df.index = dates
    df.index.name = Kline.OHLCV_TIMESTAMP 
    return df

@pytest.fixture
def sample_normalized_klines_data() -> pd.DataFrame:
    """
    Crée des données de test pour les klines APRÈS normalisation par DataManager.
    L'index est Kline.OHLCV_TIMESTAMP ('timestamp') et les colonnes sont Kline.OHLCV_OPEN ('open'), etc.
    """
    start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    periods = 100
    dates = pd.date_range(start=start_time, periods=periods, freq='1min', tz='UTC')
    
    base_price = 40000
    price_variation = np.random.uniform(-0.001, 0.001, periods)
    cumulative_change = np.cumprod(1 + price_variation)
    close_prices = base_price * cumulative_change
    
    df = pd.DataFrame({
        Kline.OHLCV_OPEN: close_prices * np.random.uniform(0.999, 1.001, periods),
        Kline.OHLCV_HIGH: close_prices * np.random.uniform(1.001, 1.003, periods),
        Kline.OHLCV_LOW: close_prices * np.random.uniform(0.997, 0.999, periods),
        Kline.OHLCV_CLOSE: close_prices,
        Kline.OHLCV_VOLUME: np.random.uniform(0.1, 1.0, periods),
        Kline.PAIR: 'BTCUSDC',
        Kline.OHLCV_KLINE_CLOSE_TIME: dates + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
        Kline.OHLCV_QUOTE_ASSET_VOLUME: close_prices * np.random.uniform(0.1, 1.0, periods),
        Kline.OHLCV_NUMBER_OF_TRADES: np.random.randint(10, 100, periods).astype(np.float64), # float pour permettre NaN
        Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: np.random.uniform(0.05, 0.5, periods),
        Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: close_prices * np.random.uniform(0.05, 0.5, periods),
        Kline.OHLCV_IS_KLINE_CLOSED: True # Sera converti en 'boolean' par normalize
    })
    
    df[Kline.OHLCV_HIGH] = df[[Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_CLOSE]].max(axis=1)
    df[Kline.OHLCV_LOW] = df[[Kline.OHLCV_OPEN, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]].min(axis=1)
    
    df.index = dates
    df.index.name = Kline.OHLCV_TIMESTAMP
    df[Kline.OHLCV_IS_KLINE_CLOSED] = df[Kline.OHLCV_IS_KLINE_CLOSED].astype('boolean')
    return df


@pytest.fixture
def sample_parquet_file(tmp_path, sample_raw_klines_data_for_storage):
    """Crée un fichier Parquet de test. Les données sont préparées par ParquetStorage."""
    df_to_write = sample_raw_klines_data_for_storage.copy()
    
    # Simuler la préparation que ParquetStorage._prepare_dataframe_for_storage ferait
    df_prepared = df_to_write.reset_index() 
    
    if Kline.OHLCV_TIMESTAMP not in df_prepared.columns:
         original_index_name = df_to_write.index.name
         if original_index_name in df_prepared.columns:
            df_prepared.rename(columns={original_index_name: Kline.OHLCV_TIMESTAMP}, inplace=True)
         elif df_prepared.columns[0] == original_index_name : # Si reset_index a mis l'ancien nom en première colonne
             df_prepared.rename(columns={original_index_name : Kline.OHLCV_TIMESTAMP}, inplace=True)
         else: # Fallback si le nom d'index n'est pas clair, en supposant que la première colonne est le timestamp
            df_prepared.rename(columns={df_prepared.columns[0]: Kline.OHLCV_TIMESTAMP}, inplace=True)


    df_prepared[Kline.OHLCV_TIMESTAMP] = pd.to_datetime(df_prepared[Kline.OHLCV_TIMESTAMP], utc=True)
    df_prepared['year'] = df_prepared[Kline.OHLCV_TIMESTAMP].dt.year.astype(np.int32)
    df_prepared['month'] = df_prepared[Kline.OHLCV_TIMESTAMP].dt.month.astype(np.int32)
    
    # Renommer les colonnes pour correspondre à ce que ParquetStorage pourrait stocker
    # (c'est-à-dire, les noms bruts avant la normalisation par DataManager)
    # La fixture sample_raw_klines_data_for_storage a déjà ces noms.
    
    table_root_path = tmp_path / "BTCUSDC_1m" 
    
    table = pa.Table.from_pandas(df_prepared, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=table_root_path,
        partition_cols=['year', 'month'],
        schema=table.schema 
    )
    return tmp_path


class TestDataManagerLoading:
    """Tests pour le chargement des klines par DataManager."""
    
    @pytest.mark.asyncio
    async def test_get_klines_interface_all_params(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage) 
            mock_storage_instance.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
            
            await dm.get_klines(
                pair='BTCUSDC', interval='1m',
                start_time=start_time, end_time=end_time, limit=100
            )
            
            mock_storage_instance.get_klines.assert_called_once_with(
                pair='BTCUSDC', interval='1m',
                start_time=start_time, end_time=end_time, limit=100
            )
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_get_klines_interface_optional_params(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            await dm.get_klines(pair='BTCUSDC', interval='5m')
            
            mock_storage_instance.get_klines.assert_called_with(
                pair='BTCUSDC', interval='5m',
                start_time=None, end_time=None, limit=None
            )
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_delegation_to_parquet_storage(self, mock_settings_parquet, tmp_path):
        # mock_settings_parquet utilise déjà tmp_path via sa propre initialisation
        # settings.data.storage_path est déjà configuré par mock_settings_parquet
        # path_to_use = settings.data.storage_path # Utiliser le chemin configuré par la fixture

        with patch('src.data.data_manager.ParquetStorage', autospec=True) as MockParquetStorageClass:
            mock_storage_instance = MockParquetStorageClass.return_value
            mock_storage_instance.initialize = AsyncMock() 
            mock_storage_instance.close = AsyncMock()
            
            dm = DataManager() # Utilise settings.data.storage_path de mock_settings_parquet
            await dm.initialize()
            
            assert dm.storage is mock_storage_instance
            MockParquetStorageClass.assert_called_once_with(settings.data.storage_path)
            mock_storage_instance.initialize.assert_awaited_once()
            
            await dm.close()
            mock_storage_instance.close.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_delegation_to_postgres_storage(self, mock_settings_postgres):
        with patch('src.data.data_manager.PostgresStorage', autospec=True) as MockPostgresStorageClass:
            mock_storage_instance = MockPostgresStorageClass.return_value
            mock_storage_instance.initialize = AsyncMock()
            mock_storage_instance.close = AsyncMock()

            dm = DataManager()
            await dm.initialize()
            
            assert dm.storage is mock_storage_instance
            MockPostgresStorageClass.assert_called_once_with(str(settings.database.url))
            mock_storage_instance.initialize.assert_awaited_once()
            
            await dm.close()
            mock_storage_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_parquet_loading_logic(self, mock_settings_parquet, sample_parquet_file, sample_raw_klines_data_for_storage):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(settings.data, 'storage_path', sample_parquet_file)
        
        dm = DataManager()
        await dm.initialize()
        
        result_df = await dm.get_klines(
            pair='BTCUSDC', interval='1m',
            start_time=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2023, 1, 1, 1, 30, 0, tzinfo=timezone.utc) 
        )
        
        assert result_df is not None, "DataFrame should not be None"
        assert not result_df.empty, "DataFrame should not be empty"
        assert isinstance(result_df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        assert result_df.index.name == Kline.OHLCV_TIMESTAMP, f"Index name should be '{Kline.OHLCV_TIMESTAMP}'"
        assert result_df.index.is_monotonic_increasing, "Index should be monotonically increasing"
        
        expected_normalized_cols = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE,
            Kline.OHLCV_VOLUME, Kline.PAIR, Kline.OHLCV_KLINE_CLOSE_TIME,
            Kline.OHLCV_QUOTE_ASSET_VOLUME, Kline.OHLCV_NUMBER_OF_TRADES,
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME,
            Kline.OHLCV_IS_KLINE_CLOSED
        ]
        for col in expected_normalized_cols:
            assert col in result_df.columns, f"Normalized column '{col}' is missing"
        
        assert result_df[Kline.OHLCV_OPEN].dtype == np.float64
        assert result_df[Kline.OHLCV_HIGH].dtype == np.float64
        assert result_df[Kline.OHLCV_LOW].dtype == np.float64
        assert result_df[Kline.OHLCV_CLOSE].dtype == np.float64
        assert result_df[Kline.OHLCV_VOLUME].dtype == np.float64
        # La normalisation peut convertir en 'boolean' (nullable) ou 'bool'
        assert pd.api.types.is_bool_dtype(result_df[Kline.OHLCV_IS_KLINE_CLOSED].dtype), \
            f"Expected boolean type for {Kline.OHLCV_IS_KLINE_CLOSED}, got {result_df[Kline.OHLCV_IS_KLINE_CLOSED].dtype}"
        
        await dm.close()
    
    @pytest.mark.asyncio
    async def test_postgres_loading_logic(self, mock_settings_postgres):
        with patch('src.data.data_manager.PostgresStorage') as MockPostgresStorage:
            dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00'], utc=True) # 3 lignes
            # Simuler les données brutes du stockage (avant normalisation par DataManager)
            raw_test_data = pd.DataFrame({
                'open_price': [40000.0, 40100.0, 40050.0], 'high_price': [40200.0, 40300.0, 40150.0],
                'low_price': [39900.0, 40000.0, 39950.0], 'close_price': [40100.0, 40200.0, 40000.0],
                'base_asset_volume': [1.0, 1.1, 0.9], 'pair': 'BTCUSDC',
                'kline_close_time': [d + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1) for d in dates],
                'is_kline_closed': [True, True, True]
            }, index=pd.DatetimeIndex(dates, name=Kline.OHLCV_TIMESTAMP)) # Index correct pour le mock
            
            mock_storage_instance = AsyncMock(spec=PostgresStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=raw_test_data)
            MockPostgresStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC', interval='1m')
            
            assert result is not None
            assert len(result) == 3
            assert result.index.name == Kline.OHLCV_TIMESTAMP
            assert result.index.is_monotonic_increasing
            assert Kline.OHLCV_CLOSE in result.columns # Vérifier la colonne normalisée
            assert result[Kline.OHLCV_CLOSE].dtype == np.float64
            
            await dm.close()

    @pytest.mark.asyncio
    async def test_date_range_filtering(self, mock_settings_parquet, sample_parquet_file):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(settings.data, 'storage_path', sample_parquet_file)
        
        dm = DataManager()
        await dm.initialize()
        
        start = datetime(2023, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 0, 20, 0, tzinfo=timezone.utc)
        
        result = await dm.get_klines(
            pair='BTCUSDC', interval='1m', start_time=start, end_time=end
        )
        
        assert result is not None
        if not result.empty:
            assert result.index.min() >= start
            assert result.index.max() <= end 
        
        await dm.close()
    
    @pytest.mark.asyncio
    async def test_multiple_pairs(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            # Simuler les données brutes du stockage
            btc_data_raw = pd.DataFrame({
                'close_price': [40000.0], 'pair': 'BTCUSDC', 'open_price': [39000.0], 'high_price': [41000.0], 'low_price': [38000.0], 'base_asset_volume': [1.0]
            }, index=pd.DatetimeIndex(['2023-01-01'], name=Kline.OHLCV_TIMESTAMP, tz='UTC'))
            
            eth_data_raw = pd.DataFrame({
                'close_price': [2000.0], 'pair': 'ETHUSDC', 'open_price': [1900.0], 'high_price': [2100.0], 'low_price': [1800.0], 'base_asset_volume': [10.0]
            }, index=pd.DatetimeIndex(['2023-01-01'], name=Kline.OHLCV_TIMESTAMP, tz='UTC'))
            
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(side_effect=[btc_data_raw, eth_data_raw])
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            result_btc = await dm.get_klines(pair='BTCUSDC')
            assert result_btc[Kline.PAIR].iloc[0] == 'BTCUSDC'
            assert result_btc[Kline.OHLCV_CLOSE].iloc[0] == 40000.0 # Vérifier la colonne normalisée
            
            result_eth = await dm.get_klines(pair='ETHUSDC')
            assert result_eth[Kline.PAIR].iloc[0] == 'ETHUSDC'
            assert result_eth[Kline.OHLCV_CLOSE].iloc[0] == 2000.0 # Vérifier la colonne normalisée
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_different_timeframes(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            intervals = ['1m', '5m', '1h']
            for interval_val in intervals: # Renommer interval en interval_val
                await dm.get_klines(pair='BTCUSDC', interval=interval_val)
            
            assert mock_storage_instance.get_klines.call_count == 3
            calls = mock_storage_instance.get_klines.call_args_list
            for i, interval_val_expected in enumerate(intervals): # Renommer interval en interval_val_expected
                assert calls[i][1]['interval'] == interval_val_expected
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_dataframe_format_validation(self, mock_settings_parquet, sample_normalized_klines_data):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            # Le mock retourne des données déjà "normalisées" car on teste la sortie de DataManager
            mock_storage_instance.get_klines = AsyncMock(return_value=sample_normalized_klines_data)
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            
            assert isinstance(result, pd.DataFrame)
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result.index.name == Kline.OHLCV_TIMESTAMP # Vérifier le nom d'index normalisé
            assert result.index.is_monotonic_increasing
            assert result.index.tz is not None and result.index.tz == timezone.utc
            
            essential_cols = [
                Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, 
                Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME
            ]
            for col in essential_cols:
                assert col in result.columns, f"La colonne normalisée '{col}' est manquante"
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_storage_not_found(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(side_effect=StorageError("File not found"))
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            with pytest.raises(DataError, match="Failed to retrieve klines from storage: File not found"):
                await dm.get_klines(pair='BTCUSDC')
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_data(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=None)
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            assert result is None
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_configuration(self, monkeypatch):
        monkeypatch.setattr(settings.data, 'storage_type', 'invalid_type_storage')
        
        dm = DataManager()
        with pytest.raises(ConfigurationError, match="Unknown storage type"):
            await dm.initialize()
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_settings_parquet):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(settings.data, 'cache_enabled', False)
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            assert dm.cache_manager is None
            
            await dm.get_klines(pair='BTCUSDC')
            await dm.get_klines(pair='BTCUSDC')
            
            assert mock_storage_instance.get_klines.call_count == 2
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_empty_dataframe_handling(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert result.empty
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, mock_settings_parquet):
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            tasks = [
                dm.get_klines(pair='BTCUSDC', interval='1m'),
                dm.get_klines(pair='ETHUSDC', interval='5m'),
                dm.get_klines(pair='BTCUSDC', interval='1h')
            ]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result_item in results: 
                assert result_item is not None 
                assert isinstance(result_item, pd.DataFrame)
            
            assert mock_storage_instance.get_klines.call_count == 3
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_data_validation_after_loading(self, mock_settings_parquet, sample_raw_klines_data_for_storage, caplog):
        """Test que la validation est appelée et que les données normalisées sont utilisées."""
        invalid_raw_data = sample_raw_klines_data_for_storage.copy()
        # Introduire une incohérence dans les données brutes (qui sera normalisée en 'high' et 'low')
        invalid_raw_data.loc[invalid_raw_data.index[0], 'high_price'] = invalid_raw_data.loc[invalid_raw_data.index[0], 'low_price'] - 10 
        
        # Mock KlineProcessor.validate_klines pour vérifier son appel et simuler un échec de validation
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage, \
             patch('src.data.kline_processor.KlineProcessor.validate_klines', return_value=(False, ["Mocked validation error: OHLC inconsistent"])) as mock_validate:
            
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=invalid_raw_data)
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            # Instead of using caplog which doesn't work with Loguru,
            # we'll just verify the important behaviors
            result_df = await dm.get_klines(pair='BTCUSDC')

            # Verify validation was called once
            mock_validate.assert_called_once()
            
            # Verify result has expected properties even with validation failure
            assert result_df is not None
            assert Kline.OHLCV_HIGH in result_df.columns
            assert Kline.OHLCV_LOW in result_df.columns
            
            # Verify the data inconsistency is still present (was not fixed by normalization)
            assert result_df.iloc[0][Kline.OHLCV_HIGH] < result_df.iloc[0][Kline.OHLCV_LOW]

            assert result_df is not None
            # Vérifier que les colonnes sont normalisées même si la validation a échoué (comportement actuel)
            assert Kline.OHLCV_HIGH in result_df.columns
            assert Kline.OHLCV_LOW in result_df.columns
            # L'incohérence OHLC est toujours présente car _normalize_storage_dataframe ne la corrige pas.
            assert result_df.iloc[0][Kline.OHLCV_HIGH] < result_df.iloc[0][Kline.OHLCV_LOW]

            await dm.close()


class TestDataManagerLoadingIntegration:
    """Tests d'intégration pour le chargement des klines."""
    
    @pytest.mark.asyncio
    async def test_full_loading_pipeline_parquet(self, tmp_path, sample_raw_klines_data_for_storage):
        """Test d'intégration complet avec Parquet."""
        settings.data.storage_type = StorageTypes.PARQUET.value 
        settings.data.storage_path = tmp_path / "integration_parquet"
        settings.data.cache_enabled = False
        
        # S'assurer que le répertoire de base pour les checkpoints existe
        (settings.data.storage_path / ".dm_checkpoints").mkdir(parents=True, exist_ok=True)

        df_for_storage = sample_raw_klines_data_for_storage.copy()
        if df_for_storage.index.name != Kline.OHLCV_TIMESTAMP:
            df_for_storage.index.name = Kline.OHLCV_TIMESTAMP

        # Utiliser ParquetStorage réel pour écrire
        # Note: ParquetStorage lui-même n'a pas de __aenter__/__aexit__
        ps = ParquetStorage(settings.data.storage_path)
        await ps.initialize()
        await ps.store_klines(df_for_storage, pair="BTCUSDC", interval="1m")
        await ps.close()
            
        async with DataManager() as dm: 
            result = await dm.get_klines(
                pair='BTCUSDC', interval='1m',
                start_time=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2023, 1, 1, 1, 0, 0, tzinfo=timezone.utc) 
            )
            
            assert result is not None
            assert not result.empty
            assert len(result) == 61 # 00:00 à 01:00 inclus = 61 minutes
            
            assert result.index.is_monotonic_increasing
            assert result.index.name == Kline.OHLCV_TIMESTAMP 
            
            numeric_cols = [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME]
            for col in numeric_cols:
                assert col in result.columns, f"Colonne normalisée '{col}' manquante"
                assert pd.api.types.is_numeric_dtype(result[col])
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_retry(self, mock_settings_parquet):
        call_count = 0
        
        async def failing_then_success_get_klines(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise StorageError("Temporary storage error")
            # Retourner des données normalisées pour la deuxième tentative
            return pd.DataFrame({Kline.OHLCV_CLOSE: [40000.0], Kline.PAIR: ['BTCUSDC']}, # Ajouter les colonnes attendues
                                index=pd.DatetimeIndex(['2023-01-01'], name=Kline.OHLCV_TIMESTAMP, tz='UTC'))
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = failing_then_success_get_klines
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            with pytest.raises(DataError, match="Failed to retrieve klines from storage: Temporary storage error"):
                await dm.get_klines(pair='BTCUSDC')
            
            result = await dm.get_klines(pair='BTCUSDC') 
            assert result is not None
            assert not result.empty
            assert Kline.OHLCV_CLOSE in result.columns
            
            await dm.close()


class TestDataManagerPerformance:
    """Tests de performance pour le chargement des klines."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_loading(self, mock_settings_parquet):
        dates_large = pd.date_range('2023-01-01', periods=10000, freq='1min', tz='UTC', name=Kline.OHLCV_TIMESTAMP)
        large_data_normalized = pd.DataFrame({
            Kline.OHLCV_OPEN: np.random.uniform(39000, 41000, 10000),
            Kline.OHLCV_HIGH: np.random.uniform(39500, 41500, 10000),
            Kline.OHLCV_LOW: np.random.uniform(38500, 40500, 10000),
            Kline.OHLCV_CLOSE: np.random.uniform(39000, 41000, 10000),
            Kline.OHLCV_VOLUME: np.random.uniform(0.1, 1.0, 10000),
            Kline.PAIR: 'BTCUSDC'
        }, index=dates_large)
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage_instance = AsyncMock(spec=ParquetStorage)
            mock_storage_instance.get_klines = AsyncMock(return_value=large_data_normalized)
            MockParquetStorage.return_value = mock_storage_instance
            
            dm = DataManager()
            await dm.initialize()
            
            import time 
            start_time_perf = time.perf_counter()
            result = await dm.get_klines(pair='BTCUSDC') 
            duration = time.perf_counter() - start_time_perf
            
            assert result is not None
            assert len(result) == 10000
            assert duration < 0.1 
            
            await dm.close()
