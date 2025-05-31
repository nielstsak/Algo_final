# tests/test_binance_client.py
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import time
import os # Ajout de l'import os

from binance.exceptions import BinanceAPIException, BinanceRequestException
import aiohttp

# Importer les settings réels pour les tests d'intégration si nécessaire
# from src.core.config import settings as actual_app_settings
# Pour cet exemple, on va supposer que actual_app_settings est disponible globalement si nécessaire
# ou que les fixtures de mock le gèrent correctement.
# Il est préférable de l'importer explicitement si utilisé directement dans les fixtures d'intégration.
try:
    from src.core.config import settings as actual_app_settings
except ImportError:
    actual_app_settings = None # Fallback si non trouvé, pour que le code ci-dessous ne casse pas


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
from src.core.constants import Kline, System


class TestRateLimiter:
    """Tests pour le RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_under_limit(self):
        limiter = RateLimiter(weight_limit=10, window_seconds=1)
        for i in range(10):
            await limiter.acquire(weight=1)
            assert limiter.get_current_weight() == (i + 1)
        assert limiter.get_current_weight() == 10

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self):
        limiter = RateLimiter(weight_limit=5, window_seconds=1)
        for _ in range(5):
            await limiter.acquire(weight=1)
        start_time = time.monotonic()
        acquire_task = asyncio.create_task(limiter.acquire(weight=1))
        try:
            await asyncio.wait_for(acquire_task, timeout=limiter.window_seconds + 0.5)
        except asyncio.TimeoutError:
            pytest.fail("limiter.acquire() a dépassé le temps imparti.")
        elapsed = time.monotonic() - start_time
        assert elapsed >= limiter.window_seconds * 0.8
        assert limiter.get_current_weight() <= limiter.weight_limit

    def test_rate_limiter_cleans_old_requests(self):
        limiter = RateLimiter(weight_limit=10, window_seconds=1)
        current_time = time.time()
        limiter.requests = [
            (current_time - 2, 5),
            (current_time - 0.5, 3)
        ]
        assert limiter.get_current_weight() == 3


class TestExponentialBackoff:
    """Tests pour ExponentialBackoff."""

    def test_exponential_backoff_increases_delay(self):
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False)
        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(2) == 4.0
        assert backoff.get_delay(3) == 8.0

    def test_exponential_backoff_respects_max_delay(self):
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=10.0, exponential_base=2.0, jitter=False)
        assert backoff.get_delay(3) == 8.0
        assert backoff.get_delay(4) == 10.0
        assert backoff.get_delay(10) == 10.0

    def test_exponential_backoff_with_jitter(self):
        backoff = ExponentialBackoff(initial_delay=1.0, exponential_base=2.0, jitter=True)
        base_delay_attempt_2 = 4.0
        delays = [backoff.get_delay(2) for _ in range(20)]
        all_within_bounds = all(base_delay_attempt_2 * 0.5 <= d <= base_delay_attempt_2 * 1.5 for d in delays)
        assert all_within_bounds
        assert len(set(delays)) > 1


class TestRetryDecorator:
    """Tests pour le décorateur with_retry."""

    @pytest.mark.asyncio
    async def test_retry_on_api_exception(self):
        mock_func = AsyncMock()

        mock_response1 = Mock()
        mock_response1.status_code = 500
        mock_response1.text = "Raw text for API error 1" # Utilisé si json() échoue ou ne contient pas code/msg
        # Simuler un payload JSON que BinanceAPIException essaiera de parser
        error_payload1 = {'code': -1000, 'msg': "Simulated API Error 1"}
        mock_response1.json = Mock(return_value=error_payload1)
        exc1 = BinanceAPIException(response=mock_response1, status_code=mock_response1.status_code, text=error_payload1['msg'])
        # Forcer les attributs si l'initialisation interne n'est pas fiable avec le mock
        exc1.code = -1000
        exc1.message = "Simulated API Error 1"


        mock_response2 = Mock()
        mock_response2.status_code = 500
        mock_response2.text = "Raw text for API error 2"
        error_payload2 = {'code': -1000, 'msg': "Simulated API Error 2"}
        mock_response2.json = Mock(return_value=error_payload2)
        exc2 = BinanceAPIException(response=mock_response2, status_code=mock_response2.status_code, text=error_payload2['msg'])
        exc2.code = -1000
        exc2.message = "Simulated API Error 2"


        mock_func.side_effect = [exc1, exc2, "success"]
        decorated = with_retry(max_attempts=3)(mock_func)
        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_network_exception(self):
        mock_func = AsyncMock()
        mock_func.side_effect = [aiohttp.ClientConnectionError("Network connection error"), "success"]
        decorated = with_retry(max_attempts=2)(mock_func)
        result = await decorated()
        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_invalid_symbol(self):
        mock_func = AsyncMock()
        mock_response_invalid = Mock()
        mock_response_invalid.status_code = 400
        mock_response_invalid.text = "Raw text for invalid symbol"
        # Simuler un payload JSON
        error_payload_invalid = {'code': -1121, 'msg': "Invalid symbol for test"}
        mock_response_invalid.json = Mock(return_value=error_payload_invalid)
        
        exception_to_raise = BinanceAPIException(response=mock_response_invalid, status_code=mock_response_invalid.status_code, text=error_payload_invalid['msg'])
        # Forcer les attributs pour s'assurer que le test fonctionne comme prévu
        exception_to_raise.code = -1121
        exception_to_raise.message = error_payload_invalid['msg'] # Utiliser le message du payload

        mock_func.side_effect = exception_to_raise
        decorated = with_retry(max_attempts=3)(mock_func)

        with pytest.raises(BinanceInvalidSymbolError) as exc_info:
            await decorated()
        
        assert "Invalid symbol for test" in str(exc_info.value) # Vérifier le message de BinanceInvalidSymbolError
        assert exc_info.value.binance_code == -1121 # Vérifier le binance_code sur l'exception personnalisée
        # Vérifier aussi le code de l'exception originale encapsulée
        assert isinstance(exc_info.value.original_exception, BinanceAPIException)
        assert exc_info.value.original_exception.code == -1121
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        mock_func = AsyncMock()
        mock_response_persistent = Mock()
        mock_response_persistent.status_code = 500
        mock_response_persistent.text = "Raw text for persistent error"
        error_payload_persistent = {'code': -1000, 'msg': "Persistent API Error for test"}
        mock_response_persistent.json = Mock(return_value=error_payload_persistent)

        persistent_exception = BinanceAPIException(response=mock_response_persistent, status_code=mock_response_persistent.status_code, text=error_payload_persistent['msg'])
        # Forcer les attributs
        persistent_exception.code = -1000
        persistent_exception.message = error_payload_persistent['msg']

        mock_func.side_effect = persistent_exception
        decorated = with_retry(max_attempts=2)(mock_func)

        with pytest.raises(BinanceAPIError) as exc_info:
            await decorated()

        assert f"Failed after 2 attempts" in str(exc_info.value)
        assert mock_func.call_count == 2
        assert isinstance(exc_info.value.original_exception, BinanceAPIException)
        assert exc_info.value.original_exception.code == -1000


class TestBinanceDataClient:
    @pytest.fixture
    def mock_settings(self, monkeypatch):
        class MockSecretStr:
            def __init__(self, value): self._value = value
            def get_secret_value(self): return self._value
        class MockBinanceSettings:
            api_key = MockSecretStr("test_key"); api_secret = MockSecretStr("test_secret")
            testnet = False; api_key_2 = None; api_secret_2 = None
        class MockTradingSettings:
            base_currency = "USDC"; allowed_pairs = ["BTCUSDC", "ETHUSDC"]
        class MockDataSettings:
            cache_enabled = False; cache_ttl = 3600
        class MockGlobalSettings:
            binance = MockBinanceSettings(); trading = MockTradingSettings(); data = MockDataSettings()
        monkeypatch.setattr('src.data.binance_client.settings', MockGlobalSettings())
        monkeypatch.setattr('src.data.binance_client.System', System)
        return MockGlobalSettings()

    @pytest_asyncio.fixture
    async def client(self, mock_settings):
        client_instance = BinanceDataClient(api_key="fixture_key", api_secret="fixture_secret", testnet=False)
        client_instance.client = AsyncMock() 
        client_instance.client.get_exchange_info = AsyncMock()
        client_instance.client.get_klines = AsyncMock()
        return client_instance

    @pytest.mark.asyncio
    async def test_validate_kline_data_valid(self, client: BinanceDataClient):
        valid_kline_raw = [
            1609459200000, "40000.00", "41000.00", "39000.00", "40500.00", "100.5",
            1609459259999, "4050000.00", 1500, "50.25", "2025000.00", "0"
        ]
        result = client._validate_kline_data(valid_kline_raw, "BTCUSDC")
        assert result[Kline.OHLCV_OPEN] == 40000.0
        assert result[Kline.OHLCV_HIGH] == 41000.0
        assert result[Kline.OHLCV_LOW] == 39000.0
        assert result[Kline.OHLCV_CLOSE] == 40500.0
        assert result[Kline.OHLCV_VOLUME] == 100.5

    @pytest.mark.asyncio
    async def test_validate_kline_data_invalid_high_low(self, client: BinanceDataClient):
        invalid_kline_raw = [
            1609459200000, "40000.00", "39000.00", "41000.00", "40500.00", "100.5",
            1609459259999, "4050000.00", 1500, "50.25", "2025000.00", "0"
        ]
        with pytest.raises(KlineValidationError):
            client._validate_kline_data(invalid_kline_raw, "BTCUSDC")

    @pytest.mark.asyncio
    async def test_validate_kline_data_invalid_volume(self, client: BinanceDataClient):
        invalid_kline_raw = [
            1609459200000, "40000.00", "41000.00", "39000.00", "40500.00", "-100.5",
            1609459259999, "4050000.00", 1500, "50.25", "2025000.00", "0"
        ]
        with pytest.raises(KlineValidationError):
            client._validate_kline_data(invalid_kline_raw, "BTCUSDC")

    @pytest.mark.asyncio
    async def test_validate_kline_data_incomplete(self, client: BinanceDataClient):
        incomplete_kline_raw = [1609459200000, "40000.00", "41000.00"]
        with pytest.raises(KlineValidationError):
            client._validate_kline_data(incomplete_kline_raw, "BTCUSDC")

    @pytest.mark.asyncio
    async def test_get_exchange_info_uses_cache(self, client: BinanceDataClient):
        mock_response = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]}
        client.client.get_exchange_info.return_value = mock_response
        await client.get_exchange_info()
        client.client.get_exchange_info.assert_called_once()
        await client.get_exchange_info()
        client.client.get_exchange_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_exchange_info_force_refresh(self, client: BinanceDataClient):
        mock_response1 = {'symbols': [{'symbol': 'BTCUSDC'}]}
        mock_response2 = {'symbols': [{'symbol': 'ETHUSDC'}]}
        client.client.get_exchange_info.side_effect = [mock_response1, mock_response2]
        await client.get_exchange_info()
        result_refreshed = await client.get_exchange_info(force_refresh=True)
        assert client.client.get_exchange_info.call_count == 2
        assert result_refreshed == mock_response2

    @pytest.mark.asyncio
    async def test_fetch_klines_valid_symbol(self, client: BinanceDataClient):
        client.client.get_exchange_info.return_value = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]}
        await client.get_exchange_info(force_refresh=True) 

        mock_klines_raw = [[1609459200000, "40000", "41000", "39000", "40500", "100", 1609459259999, "4000000", 10, "50", "2000000", "0"]]
        client.client.get_klines.return_value = mock_klines_raw
        
        result = await client.fetch_klines("BTCUSDC", interval="1m", limit=1)
        assert len(result) == 1
        assert result[0][Kline.OHLCV_OPEN] == 40000.0
        # Correction: La méthode fetch_klines construit les params sans startTime/endTime s'ils sont None
        client.client.get_klines.assert_called_once_with(symbol="BTCUSDC", interval="1m", limit=1)

    @pytest.mark.asyncio
    async def test_fetch_klines_invalid_symbol(self, client: BinanceDataClient):
        client.client.get_exchange_info.return_value = {'symbols': [{'symbol': 'ETHUSDC', 'status': 'TRADING'}]}
        await client.get_exchange_info(force_refresh=True)
        with pytest.raises(BinanceInvalidSymbolError):
            await client.fetch_klines("INVALIDPAIR", interval="1m")
        client.client.get_klines.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_klines_batch(self, client: BinanceDataClient):
        client.client.get_exchange_info.return_value = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]}
        await client.get_exchange_info(force_refresh=True)

        batch1_raw = [
            [1609459200000, "100", "110", "90", "105", "10", 1609459259999, "1000", 5, "5", "500", "0"],
            [1609459260000, "105", "115", "100", "110", "12", 1609459319999, "1200", 6, "6", "600", "0"],
        ]
        # Pour ce test, nous allons faire en sorte que le premier batch déclenche l'arrêt.
        client.client.get_klines.side_effect = [batch1_raw, []] 

        start_ts = 1609459200000
        # end_ts pour couvrir uniquement batch1_raw + un peu pour s'assurer que la boucle tente la suite
        end_ts = 1609459260000 + client._get_interval_milliseconds("1m") + 1 

        progress_calls = []
        def progress_callback_test(downloaded, total): progress_calls.append((downloaded, total))

        result = await client.fetch_klines_batch(
            symbol="BTCUSDC", interval="1m", start_time=start_ts, end_time=end_ts, progress_callback=progress_callback_test
        )
        assert len(result) == 2 # Devrait être 2, car len(batch1_raw) < 1000
        assert result[0]['open_time'] == 1609459200000
        assert client.client.get_klines.call_count == 1 
        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_validate_trading_pair_valid(self, client: BinanceDataClient, mock_settings):
        mock_settings.trading.allowed_pairs = ["BTCUSDC"]
        mock_settings.trading.base_currency = "USDC"
        client.client.get_exchange_info.return_value = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING', 'baseAsset': 'BTC', 'quoteAsset': 'USDC'}]
        }
        await client.get_exchange_info(force_refresh=True)
        assert await client.validate_trading_pair("BTCUSDC") is True

    @pytest.mark.asyncio
    async def test_validate_trading_pair_not_trading(self, client: BinanceDataClient, mock_settings):
        mock_settings.trading.allowed_pairs = ["BTCUSDC"]
        mock_settings.trading.base_currency = "USDC"
        client.client.get_exchange_info.return_value = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'HALTED', 'baseAsset': 'BTC', 'quoteAsset': 'USDC'}]
        }
        await client.get_exchange_info(force_refresh=True)
        assert await client.validate_trading_pair("BTCUSDC") is False

    @pytest.mark.asyncio
    async def test_validate_trading_pair_wrong_quote(self, client: BinanceDataClient, mock_settings):
        mock_settings.trading.allowed_pairs = ["BTCUSDT"]
        mock_settings.trading.base_currency = "USDT"
        client.client.get_exchange_info.return_value = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING', 'baseAsset': 'BTC', 'quoteAsset': 'USDC'}]
        }
        await client.get_exchange_info(force_refresh=True)
        assert await client.validate_trading_pair("BTCUSDC") is False

    @pytest.mark.asyncio
    async def test_get_interval_milliseconds(self, client: BinanceDataClient):
        assert client._get_interval_milliseconds('1m') == 60000
        assert client._get_interval_milliseconds('1h') == 3600000


@pytest.mark.integration
class TestBinanceDataClientIntegration:
    @pytest.fixture
    def mock_settings_for_integration(self, monkeypatch):
        class MockSecretStr:
            def __init__(self, value): self._value = value
            def get_secret_value(self): return self._value
        class MockBinanceSettings:
            api_key = MockSecretStr(os.getenv("BINANCE_TESTNET_API_KEY", "dummy_testnet_key"))
            api_secret = MockSecretStr(os.getenv("BINANCE_TESTNET_API_SECRET", "dummy_testnet_secret"))
            testnet = True; api_key_2 = None; api_secret_2 = None
        class MockTradingSettings:
            base_currency = "USDT"; allowed_pairs = ["BTCUSDT", "ETHUSDT"]
        class MockDataSettings:
            cache_enabled = False; cache_ttl = 3600
        class MockGlobalSettings:
            binance = MockBinanceSettings(); trading = MockTradingSettings(); data = MockDataSettings()
        
        monkeypatch.setattr('src.data.binance_client.settings', MockGlobalSettings())
        if actual_app_settings: # S'assurer que actual_app_settings est importé et existe
             monkeypatch.setattr(actual_app_settings, 'binance', MockBinanceSettings())
             monkeypatch.setattr(actual_app_settings, 'trading', MockTradingSettings())
             monkeypatch.setattr(actual_app_settings, 'data', MockDataSettings())

        return MockGlobalSettings()

    @pytest_asyncio.fixture
    async def live_testnet_client(self, mock_settings_for_integration): 
        # S'assurer que les settings globaux sont ceux potentiellement patchés par mock_settings_for_integration
        # Le code de BinanceDataClient utilise `from src.core.config import settings` implicitement
        # mock_settings_for_integration a déjà patché 'src.data.binance_client.settings'.
        # Il faut aussi s'assurer que 'from src.core.config import settings' dans live_testnet_client
        # pointe vers des settings qui reflètent l'état désiré pour ce test d'intégration.

        # Utilisation des settings patchés (ou réels si non patchés et variables d'env présentes)
        # La fixture mock_settings_for_integration doit s'assurer que `src.data.binance_client.settings`
        # et `actual_app_settings` (si utilisé directement ici) sont correctement configurés.
        
        # On utilise 'actual_app_settings' qui devrait être le module settings global.
        # mock_settings_for_integration devrait l'avoir patché.
        if not actual_app_settings:
             pytest.skip("actual_app_settings not available for integration test.")


        # Vérifier si les clés sont réelles ou dummy
        api_key_val = actual_app_settings.binance.api_key.get_secret_value() if actual_app_settings.binance.api_key else None
        api_secret_val = actual_app_settings.binance.api_secret.get_secret_value() if actual_app_settings.binance.api_secret else None

        if not (actual_app_settings.binance.testnet and \
                api_key_val and api_key_val != "dummy_testnet_key" and \
                api_secret_val and api_secret_val != "dummy_testnet_secret"):
            pytest.skip("Testnet API keys not configured (or are dummy keys), or testnet not enabled in settings. Skipping integration test.")

        client_instance = BinanceDataClient(
            api_key=api_key_val,
            api_secret=api_secret_val,
            testnet=True
        )
        await client_instance.initialize()
        yield client_instance
        await client_instance.close()

    @pytest.mark.asyncio
    async def test_real_connection_testnet_fetch_klines(self, live_testnet_client: BinanceDataClient):
        symbol_to_test = "BTCUSDT"
        try:
            klines = await live_testnet_client.fetch_klines(
                symbol=symbol_to_test, interval=Kline.INTERVAL_1MINUTE, limit=5
            )
            assert isinstance(klines, list)
            if klines:
                assert len(klines) <= 5
                for kline_item in klines:
                    assert Kline.OHLCV_OPEN in kline_item
                    assert kline_item['open_time'] > 0
            else:
                print(f"WARNING: No klines returned for {symbol_to_test} on testnet.")
        except BinanceInvalidSymbolError:
            pytest.skip(f"Symbol {symbol_to_test} not available on testnet. Skipping.")
        except BinanceAPIError as e:
            if "market is closed" in str(e).lower() or "service unavailable" in str(e).lower():
                pytest.skip(f"Testnet API error for {symbol_to_test}: {e}. Skipping.")
            raise
