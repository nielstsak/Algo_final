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

from src.data.data_manager import DataManager
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.postgres_storage import PostgresStorage
from src.core.config import settings
from src.core.constants import Kline, StorageTypes
from src.core.exceptions import (
    DataError,
    StorageError,
    ConfigurationError,
    ParquetStorageError,
    PostgresStorageError
)


# Fixtures
@pytest.fixture
def mock_settings_parquet(monkeypatch):
    """Configure settings pour utiliser Parquet."""
    monkeypatch.setattr(settings.data, 'storage_type', 'parquet')
    monkeypatch.setattr(settings.data, 'cache_enabled', False)
    monkeypatch.setattr(settings.data, 'storage_path', Path('/tmp/test_klines'))
    return settings


@pytest.fixture
def mock_settings_postgres(monkeypatch):
    """Configure settings pour utiliser PostgreSQL."""
    monkeypatch.setattr(settings.data, 'storage_type', 'postgres')
    monkeypatch.setattr(settings.data, 'cache_enabled', False)
    return settings


@pytest.fixture
def sample_klines_data():
    """Crée des données de test pour les klines."""
    # Créer 100 klines de 1 minute
    start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dates = pd.date_range(start=start_time, periods=100, freq='1T', tz='UTC')
    
    # Générer des prix OHLCV réalistes
    base_price = 40000  # Prix de base pour BTC
    price_variation = np.random.uniform(-0.001, 0.001, 100)  # ±0.1% variation
    cumulative_change = np.cumprod(1 + price_variation)
    close_prices = base_price * cumulative_change
    
    # Générer OHLC cohérents
    df = pd.DataFrame({
        'kline_open_time': dates,
        'open_price': close_prices * np.random.uniform(0.999, 1.001, 100),
        'high_price': close_prices * np.random.uniform(1.001, 1.003, 100),
        'low_price': close_prices * np.random.uniform(0.997, 0.999, 100),
        'close_price': close_prices,
        'base_asset_volume': np.random.uniform(0.1, 1.0, 100),
        'pair': 'BTCUSDC',
        'kline_close_time': dates + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
        'quote_asset_volume': close_prices * np.random.uniform(0.1, 1.0, 100),
        'number_of_trades': np.random.randint(10, 100, 100),
        'taker_buy_base_asset_volume': np.random.uniform(0.05, 0.5, 100),
        'taker_buy_quote_asset_volume': close_prices * np.random.uniform(0.05, 0.5, 100),
        'is_kline_closed': True
    })
    
    # Assurer la cohérence OHLC
    df['high_price'] = df[['open_price', 'high_price', 'close_price']].max(axis=1)
    df['low_price'] = df[['open_price', 'low_price', 'close_price']].min(axis=1)
    
    df.set_index('kline_open_time', inplace=True)
    return df


@pytest.fixture
def sample_parquet_file(tmp_path, sample_klines_data):
    """Crée un fichier Parquet de test."""
    # Créer la structure de répertoires
    table_path = tmp_path / "BTCUSDC_1m" / "year=2023" / "month=01"
    table_path.mkdir(parents=True)
    
    # Préparer les données pour Parquet
    df = sample_klines_data.reset_index()
    df['year'] = df['kline_open_time'].dt.year
    df['month'] = df['kline_open_time'].dt.month
    
    # Écrire le fichier Parquet
    table = pa.Table.from_pandas(df)
    parquet_file = table_path / "data.parquet"
    pq.write_table(table, parquet_file)
    
    return tmp_path


class TestDataManagerLoading:
    """Tests pour le chargement des klines par DataManager."""
    
    @pytest.mark.asyncio
    async def test_get_klines_interface_all_params(self, mock_settings_parquet):
        """Test de l'interface get_klines avec tous les paramètres."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            # Mock du storage
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage
            
            # Créer DataManager
            dm = DataManager()
            await dm.initialize()
            
            # Appeler get_klines avec tous les paramètres
            start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
            
            result = await dm.get_klines(
                pair='BTCUSDC',
                interval='1m',
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            
            # Vérifier que le storage a été appelé correctement
            mock_storage.get_klines.assert_called_once_with(
                pair='BTCUSDC',
                interval='1m',
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_get_klines_interface_optional_params(self, mock_settings_parquet):
        """Test de l'interface get_klines avec paramètres optionnels."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Test avec seulement pair et interval
            result = await dm.get_klines(pair='BTCUSDC', interval='5m')
            
            mock_storage.get_klines.assert_called_with(
                pair='BTCUSDC',
                interval='5m',
                start_time=None,
                end_time=None,
                limit=None
            )
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_delegation_to_parquet_storage(self, mock_settings_parquet):
        """Test de la délégation vers ParquetStorage."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Vérifier que ParquetStorage est utilisé
            assert isinstance(dm.storage, AsyncMock)
            MockParquetStorage.assert_called_once()
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_delegation_to_postgres_storage(self, mock_settings_postgres):
        """Test de la délégation vers PostgresStorage."""
        with patch('src.data.data_manager.PostgresStorage') as MockPostgresStorage:
            mock_storage = AsyncMock()
            MockPostgresStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Vérifier que PostgresStorage est utilisé
            assert isinstance(dm.storage, AsyncMock)
            MockPostgresStorage.assert_called_once()
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_parquet_loading_logic(self, mock_settings_parquet, sample_parquet_file, sample_klines_data):
        """Test du chargement depuis Parquet avec données réelles."""
        # Configurer le chemin de stockage
        mock_settings_parquet.data.storage_path = sample_parquet_file
        
        dm = DataManager()
        await dm.initialize()
        
        # Charger les données
        result = await dm.get_klines(
            pair='BTCUSDC',
            interval='1m',
            start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2023, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        )
        
        # Vérifications
        assert result is not None
        assert not result.empty
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == 'kline_open_time'
        assert result.index.is_monotonic_increasing
        
        # Vérifier les colonnes
        expected_columns = [
            'open_price', 'high_price', 'low_price', 'close_price',
            'base_asset_volume', 'pair', 'kline_close_time',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'is_kline_closed'
        ]
        for col in expected_columns:
            assert col in result.columns
        
        # Vérifier les types de données
        assert result['open_price'].dtype == np.float64
        assert result['high_price'].dtype == np.float64
        assert result['low_price'].dtype == np.float64
        assert result['close_price'].dtype == np.float64
        assert result['base_asset_volume'].dtype == np.float64
        assert result['is_kline_closed'].dtype == bool
        
        await dm.close()
    
    @pytest.mark.asyncio
    async def test_postgres_loading_logic(self, mock_settings_postgres):
        """Test du chargement depuis PostgreSQL avec mock."""
        with patch('src.data.data_manager.PostgresStorage') as MockPostgresStorage:
            # Créer des données de test
            test_data = pd.DataFrame({
                'open_price': [40000.0, 40100.0],
                'high_price': [40200.0, 40300.0],
                'low_price': [39900.0, 40000.0],
                'close_price': [40100.0, 40200.0],
                'base_asset_volume': [1.0, 1.1],
                'pair': 'BTCUSDC',
                'kline_close_time': pd.to_datetime(['2023-01-01 00:00:59.999', '2023-01-01 00:01:59.999']),
                'quote_asset_volume': [40100.0, 44220.0],
                'number_of_trades': [50, 55],
                'taker_buy_base_asset_volume': [0.5, 0.6],
                'taker_buy_quote_asset_volume': [20050.0, 24120.0],
                'is_kline_closed': [True, True]
            }, index=pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 00:01:00'], name='kline_open_time', tz='UTC'))
            
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=test_data)
            MockPostgresStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC', interval='1m')
            
            # Vérifications
            assert result is not None
            assert len(result) == 2
            assert result.index.name == 'kline_open_time'
            assert result.index.is_monotonic_increasing
            assert result['close_price'].dtype == np.float64
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_date_range_filtering(self, mock_settings_parquet, sample_parquet_file):
        """Test du filtrage par plage de dates."""
        mock_settings_parquet.data.storage_path = sample_parquet_file
        
        dm = DataManager()
        await dm.initialize()
        
        # Test avec start_time et end_time
        start = datetime(2023, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 0, 20, 0, tzinfo=timezone.utc)
        
        result = await dm.get_klines(
            pair='BTCUSDC',
            interval='1m',
            start_time=start,
            end_time=end
        )
        
        assert result is not None
        assert result.index.min() >= start
        assert result.index.max() <= end
        
        # Test avec start_time seulement
        result_start_only = await dm.get_klines(
            pair='BTCUSDC',
            interval='1m',
            start_time=start
        )
        assert result_start_only.index.min() >= start
        
        # Test avec end_time seulement
        result_end_only = await dm.get_klines(
            pair='BTCUSDC',
            interval='1m',
            end_time=end
        )
        assert result_end_only.index.max() <= end
        
        await dm.close()
    
    @pytest.mark.asyncio
    async def test_multiple_pairs(self, mock_settings_parquet):
        """Test avec plusieurs paires de trading."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            # Données pour BTCUSDC
            btc_data = pd.DataFrame({
                'close_price': [40000.0],
                'pair': 'BTCUSDC'
            }, index=pd.DatetimeIndex(['2023-01-01'], name='kline_open_time'))
            
            # Données pour ETHUSDC
            eth_data = pd.DataFrame({
                'close_price': [2000.0],
                'pair': 'ETHUSDC'
            }, index=pd.DatetimeIndex(['2023-01-01'], name='kline_open_time'))
            
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(side_effect=[btc_data, eth_data])
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Test BTCUSDC
            result_btc = await dm.get_klines(pair='BTCUSDC')
            assert result_btc['pair'].iloc[0] == 'BTCUSDC'
            assert result_btc['close_price'].iloc[0] == 40000.0
            
            # Test ETHUSDC
            result_eth = await dm.get_klines(pair='ETHUSDC')
            assert result_eth['pair'].iloc[0] == 'ETHUSDC'
            assert result_eth['close_price'].iloc[0] == 2000.0
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_different_timeframes(self, mock_settings_parquet):
        """Test avec différents intervalles de temps."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Test différents intervalles
            intervals = ['1m', '5m', '1h']
            for interval in intervals:
                await dm.get_klines(pair='BTCUSDC', interval=interval)
                
            # Vérifier que chaque intervalle a été appelé
            assert mock_storage.get_klines.call_count == 3
            calls = mock_storage.get_klines.call_args_list
            for i, interval in enumerate(intervals):
                assert calls[i][1]['interval'] == interval
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_dataframe_format_validation(self, mock_settings_parquet, sample_klines_data):
        """Test de la validation du format DataFrame retourné."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=sample_klines_data)
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            
            # Vérifier le format
            assert isinstance(result, pd.DataFrame)
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result.index.name == 'kline_open_time'
            assert result.index.is_monotonic_increasing
            assert result.index.tz is not None  # Doit avoir un timezone
            
            # Vérifier les colonnes essentielles
            essential_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'base_asset_volume']
            for col in essential_cols:
                assert col in result.columns
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_storage_not_found(self, mock_settings_parquet):
        """Test de gestion d'erreur - stockage introuvable."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(side_effect=StorageError("File not found"))
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            with pytest.raises(DataError) as exc_info:
                await dm.get_klines(pair='BTCUSDC')
            
            assert "Failed to retrieve klines" in str(exc_info.value)
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_data(self, mock_settings_parquet):
        """Test de gestion d'erreur - données manquantes."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=None)
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            
            # Doit retourner None sans lever d'exception
            assert result is None
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_configuration(self, monkeypatch):
        """Test de gestion d'erreur - configuration invalide."""
        # Configuration avec type de stockage invalide
        monkeypatch.setattr(settings.data, 'storage_type', 'invalid_type')
        
        dm = DataManager()
        
        with pytest.raises(ConfigurationError) as exc_info:
            await dm.initialize()
        
        assert "Unknown storage type" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_settings_parquet):
        """Test avec cache désactivé."""
        mock_settings_parquet.data.cache_enabled = False
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Vérifier que le cache manager n'est pas initialisé
            assert dm.cache_manager is None
            
            # Appeler get_klines deux fois
            await dm.get_klines(pair='BTCUSDC')
            await dm.get_klines(pair='BTCUSDC')
            
            # Sans cache, le storage doit être appelé deux fois
            assert mock_storage.get_klines.call_count == 2
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_empty_dataframe_handling(self, mock_settings_parquet):
        """Test avec DataFrame vide retourné."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert result.empty
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, mock_settings_parquet):
        """Test d'accès concurrent au DataManager."""
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=pd.DataFrame())
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Lancer plusieurs requêtes en parallèle
            tasks = [
                dm.get_klines(pair='BTCUSDC', interval='1m'),
                dm.get_klines(pair='ETHUSDC', interval='5m'),
                dm.get_klines(pair='BTCUSDC', interval='1h')
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Vérifier que toutes les requêtes ont réussi
            assert len(results) == 3
            for result in results:
                assert result is not None
            
            await dm.close()
    
    @pytest.mark.asyncio
    async def test_data_validation_after_loading(self, mock_settings_parquet, sample_klines_data):
        """Test de la validation des données après chargement."""
        # Créer des données avec des problèmes de cohérence OHLC
        invalid_data = sample_klines_data.copy()
        invalid_data.loc[invalid_data.index[0], 'high_price'] = 39000.0  # High < Low
        invalid_data.loc[invalid_data.index[1], 'low_price'] = 41000.0   # Low > High
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=invalid_data)
            MockParquetStorage.return_value = mock_storage
            
            # Le DataManager actuel ne valide pas automatiquement les données OHLC
            # mais on peut vérifier que les données sont retournées telles quelles
            dm = DataManager()
            await dm.initialize()
            
            result = await dm.get_klines(pair='BTCUSDC')
            
            # Les données invalides sont retournées sans modification
            assert result.iloc[0]['high_price'] < result.iloc[0]['low_price']
            assert result.iloc[1]['low_price'] > result.iloc[1]['high_price']
            
            await dm.close()


class TestDataManagerLoadingIntegration:
    """Tests d'intégration pour le chargement des klines."""
    
    @pytest.mark.asyncio
    async def test_full_loading_pipeline_parquet(self, tmp_path, sample_klines_data):
        """Test d'intégration complet avec Parquet."""
        # Configuration
        settings.data.storage_type = 'parquet'
        settings.data.storage_path = tmp_path
        settings.data.cache_enabled = False
        
        # Créer un fichier Parquet de test
        table_path = tmp_path / "BTCUSDC_1m"
        table_path.mkdir()
        
        df = sample_klines_data.reset_index()
        df['year'] = 2023
        df['month'] = 1
        
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path=table_path,
            partition_cols=['year', 'month']
        )
        
        # Test du pipeline complet
        async with DataManager() as dm:
            # Charger les données
            result = await dm.get_klines(
                pair='BTCUSDC',
                interval='1m',
                start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2023, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
            )
            
            # Vérifications complètes
            assert result is not None
            assert not result.empty
            assert len(result) <= 60  # Max 60 minutes
            
            # Vérifier l'ordre et l'index
            assert result.index.is_monotonic_increasing
            assert result.index.name == 'kline_open_time'
            
            # Vérifier les types
            numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'base_asset_volume']
            for col in numeric_cols:
                assert pd.api.types.is_numeric_dtype(result[col])
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_retry(self, mock_settings_parquet):
        """Test de récupération d'erreur et retry."""
        call_count = 0
        
        async def failing_then_success_get_klines(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise StorageError("Temporary error")
            return pd.DataFrame({'close_price': [40000.0]}, 
                              index=pd.DatetimeIndex(['2023-01-01'], name='kline_open_time'))
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = failing_then_success_get_klines
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            # Premier appel échoue
            with pytest.raises(DataError):
                await dm.get_klines(pair='BTCUSDC')
            
            # Deuxième appel réussit
            result = await dm.get_klines(pair='BTCUSDC')
            assert result is not None
            assert not result.empty
            
            await dm.close()


# Tests de performance
class TestDataManagerPerformance:
    """Tests de performance pour le chargement des klines."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_loading(self, mock_settings_parquet):
        """Test avec un grand volume de données."""
        # Créer 10000 klines
        large_data = pd.DataFrame({
            'open_price': np.random.uniform(39000, 41000, 10000),
            'high_price': np.random.uniform(39500, 41500, 10000),
            'low_price': np.random.uniform(38500, 40500, 10000),
            'close_price': np.random.uniform(39000, 41000, 10000),
            'base_asset_volume': np.random.uniform(0.1, 1.0, 10000),
            'pair': 'BTCUSDC'
        }, index=pd.date_range('2023-01-01', periods=10000, freq='1T', tz='UTC', name='kline_open_time'))
        
        with patch('src.data.data_manager.ParquetStorage') as MockParquetStorage:
            mock_storage = AsyncMock()
            mock_storage.get_klines = AsyncMock(return_value=large_data)
            MockParquetStorage.return_value = mock_storage
            
            dm = DataManager()
            await dm.initialize()
            
            import time
            start_time = time.time()
            result = await dm.get_klines(pair='BTCUSDC')
            duration = time.time() - start_time
            
            assert result is not None
            assert len(result) == 10000
            # Le chargement doit être rapide (< 1 seconde pour 10k lignes)
            assert duration < 1.0
            
            await dm.close()