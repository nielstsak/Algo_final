# tests/test_binance_client.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import time

from binance.exceptions import BinanceAPIException, BinanceRequestException
import aiohttp

from src.data.binance_client import (
    BinanceDataClient, 
    RateLimiter, 
    ExponentialBackoff,
    with_retry
)
from src.core.exceptions import (
    BinanceAPIError,
    BinanceRateLimitError,
    BinanceInvalidSymbolError,
    KlineValidationError
)


class TestRateLimiter:
    """Tests pour le RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_under_limit(self):
        """Test que le rate limiter permet les requêtes sous la limite."""
        limiter = RateLimiter(weight_limit=10, window_seconds=1)
        
        # Devrait permettre 10 requêtes de poids 1
        for _ in range(10):
            await limiter.acquire(weight=1)
            
        assert limiter.get_current_weight() == 10
        
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self):
        """Test que le rate limiter bloque quand on dépasse la limite."""
        limiter = RateLimiter(weight_limit=5, window_seconds=1)
        
        # Remplir la limite
        for _ in range(5):
            await limiter.acquire(weight=1)
            
        # La prochaine requête devrait attendre
        start_time = time.time()
        await limiter.acquire(weight=1)
        elapsed = time.time() - start_time
        
        # Devrait avoir attendu environ 1 seconde
        assert elapsed >= 0.9  # Petite marge pour les imprécisions
        
    def test_rate_limiter_cleans_old_requests(self):
        """Test que le rate limiter nettoie les anciennes requêtes."""
        limiter = RateLimiter(weight_limit=10, window_seconds=1)
        
        # Ajouter des requêtes anciennes
        old_time = time.time() - 2  # 2 secondes dans le passé
        limiter.requests = [(old_time, 5), (time.time(), 3)]
        
        # Vérifier le poids actuel (devrait ignorer l'ancienne requête)
        assert limiter.get_current_weight() == 3


class TestExponentialBackoff:
    """Tests pour ExponentialBackoff."""
    
    def test_exponential_backoff_increases_delay(self):
        """Test que le délai augmente exponentiellement."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(2) == 4.0
        assert backoff.get_delay(3) == 8.0
        
    def test_exponential_backoff_respects_max_delay(self):
        """Test que le délai ne dépasse pas le maximum."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Le délai devrait être plafonné à 10
        assert backoff.get_delay(10) == 10.0
        assert backoff.get_delay(100) == 10.0
        
    def test_exponential_backoff_with_jitter(self):
        """Test que le jitter ajoute de la variabilité."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Avec jitter, le délai devrait varier
        delays = [backoff.get_delay(2) for _ in range(10)]
        assert len(set(delays)) > 1  # Devrait avoir différentes valeurs


class TestRetryDecorator:
    """Tests pour le décorateur with_retry."""
    
    @pytest.mark.asyncio
    async def test_retry_on_api_exception(self):
        """Test que le retry fonctionne sur les exceptions API."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            BinanceAPIException(response=Mock(status_code=500), code=-1000),
            BinanceAPIException(response=Mock(status_code=500), code=-1000),
            "success"
        ]
        
        decorated = with_retry(max_attempts=3)(mock_func)
        result = await decorated()
        
        assert result == "success"
        assert mock_func.call_count == 3
        
    @pytest.mark.asyncio
    async def test_retry_on_network_exception(self):
        """Test que le retry fonctionne sur les erreurs réseau."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            aiohttp.ClientError("Network error"),
            "success"
        ]
        
        decorated = with_retry(max_attempts=2)(mock_func)
        result = await decorated()
        
        assert result == "success"
        assert mock_func.call_count == 2
        
    @pytest.mark.asyncio
    async def test_no_retry_on_invalid_symbol(self):
        """Test que certaines erreurs ne sont pas retentées."""
        mock_func = AsyncMock()
        exception = BinanceAPIException(response=Mock(status_code=400), code=-1121)
        exception.message = "Invalid symbol"
        mock_func.side_effect = exception
        
        decorated = with_retry(max_attempts=3)(mock_func)
        
        with pytest.raises(BinanceInvalidSymbolError):
            await decorated()
            
        # Ne devrait être appelé qu'une fois (pas de retry)
        assert mock_func.call_count == 1
        
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test que les retries s'épuisent correctement."""
        mock_func = AsyncMock()
        mock_func.side_effect = BinanceAPIException(
            response=Mock(status_code=500), 
            code=-1000
        )
        
        decorated = with_retry(max_attempts=2)(mock_func)
        
        with pytest.raises(BinanceAPIError) as exc_info:
            await decorated()
            
        assert "Failed after 2 attempts" in str(exc_info.value)
        assert mock_func.call_count == 2


class TestBinanceDataClient:
    """Tests pour BinanceDataClient."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock des settings."""
        with patch('src.data.binance_client.settings') as mock:
            mock.binance.api_key.get_secret_value.return_value = "test_key"
            mock.binance.api_secret.get_secret_value.return_value = "test_secret"
            mock.binance.testnet = False
            mock.binance.api_key_2 = None
            mock.binance.api_secret_2 = None
            mock.trading.base_currency = "USDC"
            yield mock
            
    @pytest.fixture
    async def client(self, mock_settings):
        """Crée un client pour les tests."""
        client = BinanceDataClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=False
        )
        # Mock le client Binance
        client.client = AsyncMock()
        yield client
        
    def test_validate_kline_data_valid(self, client):
        """Test la validation de klines valides."""
        valid_kline = [
            1609459200000,  # open_time
            "40000.00",     # open
            "41000.00",     # high
            "39000.00",     # low
            "40500.00",     # close
            "100.5",        # volume
            1609459260000,  # close_time
            "4050000.00",   # quote_asset_volume
            "1500",         # number_of_trades
            "50.25",        # taker_buy_base_asset_volume
            "2025000.00",   # taker_buy_quote_asset_volume
            "0"             # ignore
        ]
        
        result = client._validate_kline_data(valid_kline, "BTCUSDC")
        
        assert result['open'] == 40000.0
        assert result['high'] == 41000.0
        assert result['low'] == 39000.0
        assert result['close'] == 40500.0
        assert result['volume'] == 100.5
        assert result['number_of_trades'] == 1500
        
    def test_validate_kline_data_invalid_high_low(self, client):
        """Test la validation échoue si high < low."""
        invalid_kline = [
            1609459200000, "40000.00", "39000.00", "41000.00",  # high < low
            "40500.00", "100.5", 1609459260000, "4050000.00",
            "1500", "50.25", "2025000.00", "0"
        ]
        
        with pytest.raises(KlineValidationError) as exc_info:
            client._validate_kline_data(invalid_kline, "BTCUSDC")
            
        assert "High price" in str(exc_info.value)
        
    def test_validate_kline_data_invalid_volume(self, client):
        """Test la validation échoue si volume négatif."""
        invalid_kline = [
            1609459200000, "40000.00", "41000.00", "39000.00",
            "40500.00", "-100.5",  # volume négatif
            1609459260000, "4050000.00", "1500", "50.25", "2025000.00", "0"
        ]
        
        with pytest.raises(KlineValidationError) as exc_info:
            client._validate_kline_data(invalid_kline, "BTCUSDC")
            
        assert "Negative volume" in str(exc_info.value)
        
    def test_validate_kline_data_incomplete(self, client):
        """Test la validation échoue si données incomplètes."""
        incomplete_kline = [1609459200000, "40000.00", "41000.00"]  # Seulement 3 champs
        
        with pytest.raises(KlineValidationError) as exc_info:
            client._validate_kline_data(incomplete_kline, "BTCUSDC")
            
        assert "incomplete" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_get_exchange_info_uses_cache(self, client):
        """Test que get_exchange_info utilise le cache."""
        mock_exchange_info = {
            'symbols': [
                {'symbol': 'BTCUSDC', 'status': 'TRADING'},
                {'symbol': 'ETHUSDC', 'status': 'TRADING'}
            ]
        }
        client.client.get_exchange_info.return_value = mock_exchange_info
        
        # Premier appel
        result1 = await client.get_exchange_info()
        assert result1 == mock_exchange_info
        assert client.client.get_exchange_info.call_count == 1
        
        # Deuxième appel (devrait utiliser le cache)
        result2 = await client.get_exchange_info()
        assert result2 == mock_exchange_info
        assert client.client.get_exchange_info.call_count == 1  # Pas d'appel supplémentaire
        
    @pytest.mark.asyncio
    async def test_get_exchange_info_force_refresh(self, client):
        """Test que force_refresh ignore le cache."""
        mock_exchange_info = {'symbols': []}
        client.client.get_exchange_info.return_value = mock_exchange_info
        
        # Remplir le cache
        await client.get_exchange_info()
        
        # Forcer le rafraîchissement
        await client.get_exchange_info(force_refresh=True)
        
        assert client.client.get_exchange_info.call_count == 2
        
    @pytest.mark.asyncio
    async def test_fetch_klines_valid_symbol(self, client):
        """Test fetch_klines avec un symbole valide."""
        # Mock exchange info
        client._exchange_info_cache = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]
        }
        client._exchange_info_cache_time = time.time()
        
        # Mock klines response
        mock_klines = [
            [
                1609459200000, "40000.00", "41000.00", "39000.00",
                "40500.00", "100.5", 1609459260000, "4050000.00",
                "1500", "50.25", "2025000.00", "0"
            ]
        ]
        client.client.get_klines.return_value = mock_klines
        
        result = await client.fetch_klines("BTCUSDC", interval="1m", limit=1)
        
        assert len(result) == 1
        assert result[0]['open'] == 40000.0
        assert result[0]['close'] == 40500.0
        
    @pytest.mark.asyncio
    async def test_fetch_klines_invalid_symbol(self, client):
        """Test fetch_klines avec un symbole invalide."""
        # Mock exchange info sans le symbole
        client._exchange_info_cache = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]
        }
        client._exchange_info_cache_time = time.time()
        
        with pytest.raises(BinanceInvalidSymbolError) as exc_info:
            await client.fetch_klines("INVALID", interval="1m")
            
        assert "INVALID" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_fetch_klines_batch(self, client):
        """Test fetch_klines_batch pour téléchargement en lots."""
        # Mock exchange info
        client._exchange_info_cache = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]
        }
        client._exchange_info_cache_time = time.time()
        
        # Mock plusieurs batches de klines
        batch1 = [
            [
                1609459200000 + i * 60000, "40000.00", "41000.00", "39000.00",
                "40500.00", "100.5", 1609459200000 + i * 60000 + 59999,
                "4050000.00", "1500", "50.25", "2025000.00", "0"
            ]
            for i in range(1000)
        ]
        
        batch2 = [
            [
                1609459200000 + 1000 * 60000 + i * 60000, "40000.00", "41000.00", 
                "39000.00", "40500.00", "100.5", 
                1609459200000 + 1000 * 60000 + i * 60000 + 59999,
                "4050000.00", "1500", "50.25", "2025000.00", "0"
            ]
            for i in range(500)  # Batch partiel = fin des données
        ]
        
        client.client.get_klines.side_effect = [batch1, batch2]
        
        # Callback pour suivre la progression
        progress_calls = []
        def progress_callback(downloaded, total):
            progress_calls.append((downloaded, total))
            
        result = await client.fetch_klines_batch(
            symbol="BTCUSDC",
            interval="1m",
            start_time=1609459200000,
            end_time=1609459200000 + 2000 * 60000,
            progress_callback=progress_callback
        )
        
        assert len(result) == 1500  # 1000 + 500
        assert len(progress_calls) >= 2  # Au moins 2 appels de progression
        
    @pytest.mark.asyncio
    async def test_validate_trading_pair_valid(self, client):
        """Test validate_trading_pair avec une paire valide."""
        client._exchange_info_cache = {
            'symbols': [
                {
                    'symbol': 'BTCUSDC',
                    'status': 'TRADING',
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDC'
                }
            ]
        }
        client._exchange_info_cache_time = time.time()
        
        result = await client.validate_trading_pair("BTCUSDC")
        assert result is True
        
    @pytest.mark.asyncio
    async def test_validate_trading_pair_not_trading(self, client):
        """Test validate_trading_pair avec une paire non active."""
        client._exchange_info_cache = {
            'symbols': [
                {
                    'symbol': 'BTCUSDC',
                    'status': 'HALT',  # Pas en trading
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDC'
                }
            ]
        }
        client._exchange_info_cache_time = time.time()
        
        result = await client.validate_trading_pair("BTCUSDC")
        assert result is False
        
    @pytest.mark.asyncio
    async def test_validate_trading_pair_wrong_quote(self, client):
        """Test validate_trading_pair avec mauvaise devise de cotation."""
        client._exchange_info_cache = {
            'symbols': [
                {
                    'symbol': 'BTCUSDT',  # USDT au lieu de USDC
                    'status': 'TRADING',
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDT'
                }
            ]
        }
        client._exchange_info_cache_time = time.time()
        
        result = await client.validate_trading_pair("BTCUSDT")
        assert result is False  # Car on veut seulement USDC
        
    def test_get_interval_milliseconds(self, client):
        """Test la conversion des intervalles en millisecondes."""
        assert client._get_interval_milliseconds('1m') == 60000
        assert client._get_interval_milliseconds('5m') == 300000
        assert client._get_interval_milliseconds('1h') == 3600000
        assert client._get_interval_milliseconds('1d') == 86400000
        assert client._get_interval_milliseconds('invalid') == 60000  # Défaut


# Tests d'intégration (à exécuter avec prudence sur testnet)
@pytest.mark.integration
class TestBinanceDataClientIntegration:
    """Tests d'intégration avec l'API Binance testnet."""
    
    @pytest.mark.asyncio
    async def test_real_connection_testnet(self):
        """Test de connexion réelle au testnet."""
        # NOTE: Nécessite des clés API testnet valides dans l'environnement
        client = BinanceDataClient(testnet=True)
        
        try:
            await client.initialize()
            
            # Test simple de récupération de klines
            klines = await client.fetch_klines(
                symbol="BTCUSDT",  # USDT plus commun sur testnet
                interval="1m",
                limit=5
            )
            
            assert len(klines) <= 5
            if klines:
                assert 'open' in klines[0]
                assert 'close' in klines[0]
                
        finally:
            await client.close()