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
from src.core.constants import Kline # Ajouté pour Kline.OHLCV_OPEN etc.


class TestRateLimiter:
    """Tests pour le RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_under_limit(self):
        """Test que le rate limiter permet les requêtes sous la limite."""
        limiter = RateLimiter(weight_limit=10, window_seconds=1)

        for i in range(10):
            await limiter.acquire(weight=1)
            assert limiter.get_current_weight() == (i + 1)
        # Après 10 requêtes, le poids devrait être 10
        assert limiter.get_current_weight() == 10


    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self):
        """Test que le rate limiter bloque quand on dépasse la limite."""
        limiter = RateLimiter(weight_limit=5, window_seconds=1)

        # Remplir la limite
        for _ in range(5):
            await limiter.acquire(weight=1)

        # La prochaine requête devrait attendre
        start_time = time.monotonic() # Utiliser monotonic pour mesurer le temps écoulé
        acquire_task = asyncio.create_task(limiter.acquire(weight=1))

        # Permettre à d'autres tâches de s'exécuter, y compris le sleep dans acquire
        # Attendre un peu plus que la fenêtre pour s'assurer que le sleep a eu lieu
        try:
            await asyncio.wait_for(acquire_task, timeout=limiter.window_seconds + 0.5)
        except asyncio.TimeoutError:
            pytest.fail("limiter.acquire() a dépassé le temps imparti, suggérant un blocage persistant.")

        elapsed = time.monotonic() - start_time

        # Devrait avoir attendu environ window_seconds (1s ici)
        # Compte tenu de la logique de wait_time (oldest_request_time + 0.1),
        # et du fait que le test remplit la limite exactement au début de la fenêtre.
        # L'attente sera proche de window_seconds.
        assert elapsed >= limiter.window_seconds * 0.8 # Marge pour l'imprécision et la logique d'attente
        assert limiter.get_current_weight() <= limiter.weight_limit # Le poids ne doit pas dépasser la limite après acquisition


    def test_rate_limiter_cleans_old_requests(self):
        """Test que le rate limiter nettoie les anciennes requêtes."""
        limiter = RateLimiter(weight_limit=10, window_seconds=1)

        # Ajouter des requêtes anciennes et récentes
        current_time = time.time()
        limiter.requests = [
            (current_time - 2, 5),  # Ancienne, devrait être ignorée
            (current_time - 0.5, 3) # Récente
        ]
        # Le nettoyage se fait au moment de l'appel à get_current_weight ou acquire
        assert limiter.get_current_weight() == 3

        # Simuler un appel à acquire pour forcer le nettoyage interne si get_current_weight ne le faisait pas
        # Dans la version corrigée, get_current_weight nettoie aussi, donc c'est redondant mais bon pour la robustesse.
        # asyncio.run(limiter.acquire(weight=0)) # Ne peut pas appeler run dans un test sync.
        # La logique de nettoyage est dans get_current_weight et au début de acquire.
        # Ici, get_current_weight suffit.


class TestExponentialBackoff:
    """Tests pour ExponentialBackoff."""

    def test_exponential_backoff_increases_delay(self):
        """Test que le délai augmente exponentiellement."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False # Important pour la prévisibilité dans ce test
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
        assert backoff.get_delay(3) == 8.0 # 2^3 = 8
        assert backoff.get_delay(4) == 10.0 # 2^4 = 16, mais plafonné à 10
        assert backoff.get_delay(10) == 10.0 # Reste plafonné

    def test_exponential_backoff_with_jitter(self):
        """Test que le jitter ajoute de la variabilité."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=True # Jitter activé
        )

        # Le délai de base pour la 2ème tentative (attempt=1) serait 1.0 * 2^1 = 2.0
        # Avec jitter, il sera entre 2.0 * 0.5 = 1.0 et 2.0 * 1.5 = 3.0
        # Pour attempt=2 (3ème tentative), base = 4.0. Avec jitter [2.0, 6.0]
        base_delay_attempt_2 = 4.0
        delays = [backoff.get_delay(2) for _ in range(20)] # Générer plusieurs délais

        all_within_bounds = all(base_delay_attempt_2 * 0.5 <= d <= base_delay_attempt_2 * 1.5 for d in delays)
        assert all_within_bounds, "Les délais avec jitter ne sont pas dans les bornes attendues"

        # Vérifier qu'il y a de la variabilité (plus d'une valeur unique)
        # C'est probabiliste, mais avec 20 essais, c'est très probable.
        assert len(set(delays)) > 1, "Le jitter n'a pas produit de valeurs de délai variées"


class TestRetryDecorator:
    """Tests pour le décorateur with_retry."""

    @pytest.mark.asyncio
    async def test_retry_on_api_exception(self):
        """Test que le retry fonctionne sur les exceptions API."""
        mock_func = AsyncMock()

        # Correction: Instancier BinanceAPIException et définir .code manuellement
        exc1 = BinanceAPIException(response=Mock(status_code=500), message="Simulated API Error 1")
        exc1.code = -1000 # Code d'erreur générique pour le test
        exc2 = BinanceAPIException(response=Mock(status_code=500), message="Simulated API Error 2")
        exc2.code = -1000

        mock_func.side_effect = [exc1, exc2, "success"]

        decorated = with_retry(max_attempts=3)(mock_func)
        result = await decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_network_exception(self):
        """Test que le retry fonctionne sur les erreurs réseau."""
        mock_func = AsyncMock()
        # aiohttp.ClientError est une classe de base pour les erreurs client aiohttp
        mock_func.side_effect = [
            aiohttp.ClientConnectionError("Network connection error"), # Exemple d'erreur réseau spécifique
            "success"
        ]

        decorated = with_retry(max_attempts=2)(mock_func)
        result = await decorated()

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_invalid_symbol(self):
        """Test que certaines erreurs (comme Invalid Symbol) ne sont pas retentées."""
        mock_func = AsyncMock()

        # Correction: Instancier BinanceAPIException et définir .code
        exception_to_raise = BinanceAPIException(response=Mock(status_code=400), message="Invalid symbol")
        exception_to_raise.code = -1121 # Code pour "Invalid symbol"

        mock_func.side_effect = exception_to_raise

        decorated = with_retry(max_attempts=3)(mock_func)

        with pytest.raises(BinanceInvalidSymbolError) as exc_info: # Le décorateur devrait convertir en BinanceInvalidSymbolError
            await decorated()

        assert "Invalid symbol" in str(exc_info.value)
        assert exc_info.value.binance_code == -1121
        assert mock_func.call_count == 1 # Ne devrait être appelé qu'une fois

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test que les retries s'épuisent correctement et que l'exception finale est levée."""
        mock_func = AsyncMock()

        # Correction: Instancier BinanceAPIException et définir .code
        persistent_exception = BinanceAPIException(response=Mock(status_code=500), message="Persistent API Error")
        persistent_exception.code = -1000 # Une erreur retryable

        mock_func.side_effect = persistent_exception # Lèvera cette exception à chaque appel

        decorated = with_retry(max_attempts=2)(mock_func)

        with pytest.raises(BinanceAPIError) as exc_info: # L'exception finale après épuisement
            await decorated()

        assert f"Failed after 2 attempts" in str(exc_info.value)
        assert mock_func.call_count == 2
        assert isinstance(exc_info.value.original_exception, BinanceAPIException)
        assert exc_info.value.original_exception.code == -1000


class TestBinanceDataClient:
    """Tests pour BinanceDataClient."""

    @pytest.fixture
    def mock_settings(self, monkeypatch): # monkeypatch pour définir les variables d'environnement si nécessaire
        """Mock des settings globaux."""
        # Utiliser monkeypatch pour s'assurer que settings.binance.api_key etc. retournent les bonnes valeurs
        # Ceci est plus robuste que de patcher 'src.data.binance_client.settings' directement
        # car cela affecte l'objet settings importé partout.

        class MockSecretStr:
            def __init__(self, value):
                self._value = value
            def get_secret_value(self):
                return self._value

        class MockBinanceSettings:
            api_key = MockSecretStr("test_key")
            api_secret = MockSecretStr("test_secret")
            testnet = False
            api_key_2 = None
            api_secret_2 = None

        class MockTradingSettings:
            base_currency = "USDC"
            allowed_pairs = ["BTCUSDC", "ETHUSDC"] # Exemple de paires autorisées

        class MockDataSettings:
            cache_enabled = False # Désactiver le cache pour simplifier certains tests
            cache_ttl = 3600


        class MockGlobalSettings:
            binance = MockBinanceSettings()
            trading = MockTradingSettings()
            data = MockDataSettings()


        monkeypatch.setattr('src.data.binance_client.settings', MockGlobalSettings())
        # Si vos constantes System sont utilisées pour les valeurs par défaut, assurez-vous qu'elles sont accessibles
        # monkeypatch.setattr('src.data.binance_client.System', System) # Si System est aussi mocké/modifié

        return MockGlobalSettings()


    @pytest.fixture
    async def client(self, mock_settings): # mock_settings est injecté par pytest
        """Crée une instance de BinanceDataClient pour les tests, avec un client interne mocké."""
        # Le client est initialisé avec des clés de test, et son client interne (AsyncClient de python-binance)
        # est remplacé par un AsyncMock.
        # Pas besoin d'appeler client_instance.initialize() si on mock toutes les interactions internes.
        client_instance = BinanceDataClient(
            api_key="fixture_key", # Peut utiliser des valeurs spécifiques ici ou celles de mock_settings
            api_secret="fixture_secret",
            testnet=False
        )
        client_instance.client = AsyncMock() # Mock l'instance interne de python-binance.AsyncClient
        yield client_instance
        # Pas besoin de client_instance.close() si initialize() n'est pas appelé
        # ou si close() ne fait que fermer le client interne réel.

    @pytest.mark.asyncio # Rendre le test async pour utiliser la fixture async 'client'
    async def test_validate_kline_data_valid(self, client: BinanceDataClient): # Type hint pour la clarté
        """Test la validation de klines valides."""
        valid_kline_raw = [
            1609459200000,  # open_time
            "40000.00",     # open
            "41000.00",     # high
            "39000.00",     # low
            "40500.00",     # close
            "100.5",        # volume
            1609459259999,  # close_time (59999ms après open_time pour 1m)
            "4050000.00",   # quote_asset_volume
            1500,           # number_of_trades (doit être int)
            "50.25",        # taker_buy_base_asset_volume
            "2025000.00",   # taker_buy_quote_asset_volume
            "0"             # ignore
        ]

        # La méthode _validate_kline_data est synchrone
        result = client._validate_kline_data(valid_kline_raw, "BTCUSDC")

        assert result[Kline.OHLCV_OPEN] == 40000.0
        assert result[Kline.OHLCV_HIGH] == 41000.0
        assert result[Kline.OHLCV_LOW] == 39000.0
        assert result[Kline.OHLCV_CLOSE] == 40500.0
        assert result[Kline.OHLCV_VOLUME] == 100.5
        assert result[Kline.OHLCV_NUMBER_OF_TRADES] == 1500
        assert result['open_time'] == 1609459200000
        assert result['close_time'] == 1609459259999


    @pytest.mark.asyncio
    async def test_validate_kline_data_invalid_high_low(self, client: BinanceDataClient):
        """Test la validation échoue si high < low."""
        invalid_kline_raw = [
            1609459200000, "40000.00", "39000.00", "41000.00",  # high (39k) < low (41k)
            "40500.00", "100.5", 1609459259999, "4050000.00",
            1500, "50.25", "2025000.00", "0"
        ]
        with pytest.raises(KlineValidationError) as exc_info:
            client._validate_kline_data(invalid_kline_raw, "BTCUSDC")
        assert "High price" in str(exc_info.value) or "less than low price" in str(exc_info.value)


    @pytest.mark.asyncio
    async def test_validate_kline_data_invalid_volume(self, client: BinanceDataClient):
        """Test la validation échoue si volume négatif."""
        invalid_kline_raw = [
            1609459200000, "40000.00", "41000.00", "39000.00",
            "40500.00", "-100.5",  # volume négatif
            1609459259999, "4050000.00", 1500, "50.25", "2025000.00", "0"
        ]
        with pytest.raises(KlineValidationError) as exc_info:
            client._validate_kline_data(invalid_kline_raw, "BTCUSDC")
        assert "volume" in str(exc_info.value).lower() and "negative" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_kline_data_incomplete(self, client: BinanceDataClient):
        """Test la validation échoue si données kline incomplètes."""
        incomplete_kline_raw = [1609459200000, "40000.00", "41000.00"] # Seulement 3 champs
        with pytest.raises(KlineValidationError) as exc_info:
            client._validate_kline_data(incomplete_kline_raw, "BTCUSDC")
        assert "incomplete" in str(exc_info.value).lower() or "expected at least 11 fields" in str(exc_info.value).lower()


    @pytest.mark.asyncio
    async def test_get_exchange_info_uses_cache(self, client: BinanceDataClient):
        """Test que get_exchange_info utilise le cache."""
        mock_exchange_info_response = {
            'symbols': [
                {'symbol': 'BTCUSDC', 'status': 'TRADING'},
                {'symbol': 'ETHUSDC', 'status': 'TRADING'}
            ]
        }
        # Configurer le mock du client interne pour retourner cette réponse
        client.client.get_exchange_info.return_value = mock_exchange_info_response

        # Premier appel - devrait appeler l'API mockée
        result1 = await client.get_exchange_info()
        assert result1 == mock_exchange_info_response
        client.client.get_exchange_info.assert_called_once() # Vérifier que le client interne a été appelé

        # Deuxième appel - devrait utiliser le cache et ne pas rappeler le client interne
        result2 = await client.get_exchange_info()
        assert result2 == mock_exchange_info_response
        client.client.get_exchange_info.assert_called_once() # Toujours une seule fois

    @pytest.mark.asyncio
    async def test_get_exchange_info_force_refresh(self, client: BinanceDataClient):
        """Test que force_refresh=True ignore le cache et rappelle l'API."""
        mock_response1 = {'symbols': [{'symbol': 'BTCUSDC'}]}
        mock_response2 = {'symbols': [{'symbol': 'ETHUSDC'}]} # Nouvelle réponse pour le deuxième appel
        client.client.get_exchange_info.side_effect = [mock_response1, mock_response2]

        # Premier appel pour remplir le cache
        await client.get_exchange_info()
        assert client.client.get_exchange_info.call_count == 1

        # Deuxième appel avec force_refresh
        result_refreshed = await client.get_exchange_info(force_refresh=True)
        assert client.client.get_exchange_info.call_count == 2
        assert result_refreshed == mock_response2 # Doit retourner la nouvelle réponse

    @pytest.mark.asyncio
    async def test_fetch_klines_valid_symbol(self, client: BinanceDataClient):
        """Test fetch_klines avec un symbole valide."""
        # Pré-remplir le cache d'exchange_info pour éviter un appel API réel ou mocké pour cela
        client._exchange_info_cache = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]}
        client._exchange_info_cache_time = time.time()

        # Mock la réponse de client.client.get_klines (le client interne de python-binance)
        mock_klines_raw_response = [
            [1609459200000, "40000", "41000", "39000", "40500", "100", 1609459259999, "4000000", 10, "50", "2000000", "0"]
        ]
        client.client.get_klines.return_value = mock_klines_raw_response

        result = await client.fetch_klines("BTCUSDC", interval="1m", limit=1)

        assert len(result) == 1
        assert result[0][Kline.OHLCV_OPEN] == 40000.0
        client.client.get_klines.assert_called_once_with(symbol="BTCUSDC", interval="1m", limit=1)


    @pytest.mark.asyncio
    async def test_fetch_klines_invalid_symbol(self, client: BinanceDataClient):
        """Test fetch_klines avec un symbole qui n'est pas dans exchange_info."""
        client._exchange_info_cache = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]}
        client._exchange_info_cache_time = time.time()

        with pytest.raises(BinanceInvalidSymbolError) as exc_info:
            await client.fetch_klines("INVALIDPAIR", interval="1m")
        assert "INVALIDPAIR" in str(exc_info.value)
        client.client.get_klines.assert_not_called() # Ne devrait pas appeler l'API si le symbole est invalide localement


    @pytest.mark.asyncio
    async def test_fetch_klines_batch(self, client: BinanceDataClient):
        """Test fetch_klines_batch pour le téléchargement en lots."""
        client._exchange_info_cache = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING'}]}
        client._exchange_info_cache_time = time.time()

        # Simuler plusieurs réponses de l'API pour les lots
        # Lot 1: 2 klines
        batch1_raw = [
            [1609459200000, "100", "110", "90", "105", "10", 1609459259999, "1000", 5, "5", "500", "0"],
            [1609459260000, "105", "115", "100", "110", "12", 1609459319999, "1200", 6, "6", "600", "0"],
        ]
        # Lot 2: 1 kline (simulant la fin des données)
        batch2_raw = [
            [1609459320000, "110", "120", "108", "118", "15", 1609459379999, "1500", 7, "7", "700", "0"],
        ]
        client.client.get_klines.side_effect = [batch1_raw, batch2_raw, []] # La dernière réponse vide pour arrêter la boucle

        start_ts = 1609459200000
        end_ts = 1609459379999 + 60000 # Un peu après la dernière kline pour s'assurer qu'on essaie de la prendre

        progress_calls = []
        def progress_callback_test(downloaded, total):
            progress_calls.append((downloaded, total))

        result = await client.fetch_klines_batch(
            symbol="BTCUSDC",
            interval="1m",
            start_time=start_ts,
            end_time=end_ts,
            progress_callback=progress_callback_test
        )

        assert len(result) == 3 # 2 du premier lot + 1 du deuxième
        assert result[0]['open_time'] == 1609459200000
        assert result[2]['open_time'] == 1609459320000
        assert client.client.get_klines.call_count >= 2 # Au moins deux appels pour les deux lots non vides
        assert len(progress_calls) > 0 # Vérifier que le callback a été appelé


    @pytest.mark.asyncio
    async def test_validate_trading_pair_valid(self, client: BinanceDataClient, mock_settings):
        """Test validate_trading_pair avec une paire valide et configurée."""
        # Assurer que mock_settings est bien configuré pour ce test
        mock_settings.trading.allowed_pairs = ["BTCUSDC"]
        mock_settings.trading.base_currency = "USDC"

        client._exchange_info_cache = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING', 'baseAsset': 'BTC', 'quoteAsset': 'USDC'}]
        }
        client._exchange_info_cache_time = time.time()

        is_valid = await client.validate_trading_pair("BTCUSDC")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_trading_pair_not_trading(self, client: BinanceDataClient, mock_settings):
        """Test avec une paire valide dans exchange_info mais pas en statut TRADING."""
        mock_settings.trading.allowed_pairs = ["BTCUSDC"]
        mock_settings.trading.base_currency = "USDC"
        client._exchange_info_cache = {
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'HALTED', 'baseAsset': 'BTC', 'quoteAsset': 'USDC'}]
        }
        client._exchange_info_cache_time = time.time()
        is_valid = await client.validate_trading_pair("BTCUSDC")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_trading_pair_wrong_quote(self, client: BinanceDataClient, mock_settings):
        """Test avec une paire dont la devise de cotation ne correspond pas à la config."""
        mock_settings.trading.allowed_pairs = ["BTCUSDT"] # La config autorise USDT
        mock_settings.trading.base_currency = "USDT"      # La config attend USDT

        client._exchange_info_cache = { # Mais l'exchange info dit que c'est une paire USDC
            'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING', 'baseAsset': 'BTC', 'quoteAsset': 'USDC'}]
        }
        client._exchange_info_cache_time = time.time()
        # On essaie de valider BTCUSDC, qui n'est pas USDT
        is_valid = await client.validate_trading_pair("BTCUSDC")
        assert is_valid is False # Devrait être faux car la config attend USDT

        # Cas inverse: la config attend USDC, mais on valide une paire USDT
        mock_settings.trading.base_currency = "USDC"
        client._exchange_info_cache = {
            'symbols': [{'symbol': 'BTCUSDT', 'status': 'TRADING', 'baseAsset': 'BTC', 'quoteAsset': 'USDT'}]
        }
        is_valid_usdt_pair = await client.validate_trading_pair("BTCUSDT")
        assert is_valid_usdt_pair is False


    @pytest.mark.asyncio # Rendre le test async
    async def test_get_interval_milliseconds(self, client: BinanceDataClient): # Utiliser la fixture client
        """Test la conversion des intervalles en millisecondes."""
        # La méthode _get_interval_milliseconds est synchrone sur l'instance client
        assert client._get_interval_milliseconds('1m') == 60000
        assert client._get_interval_milliseconds('5m') == 300000
        assert client._get_interval_milliseconds('1h') == 3600000
        assert client._get_interval_milliseconds('1d') == 86400000
        assert client._get_interval_milliseconds('1w') == 7 * 86400000
        assert client._get_interval_milliseconds('1M') == 30 * 86400000 # Approximation
        assert client._get_interval_milliseconds('invalid_interval') == 60000  # Test du défaut


# Tests d'intégration (marqués pour être potentiellement ignorés ou exécutés séparément)
@pytest.mark.integration
class TestBinanceDataClientIntegration:
    """Tests d'intégration avec l'API Binance testnet."""

    @pytest.fixture
    async def live_testnet_client(self):
        """Crée un client réel connecté au testnet Binance."""
        # Ce test nécessite que les variables d'environnement BINANCE_TESTNET_API_KEY et BINANCE_TESTNET_API_SECRET
        # soient définies, ou que settings.binance.api_key/secret pointent vers des clés testnet
        # et settings.binance.testnet = True.
        # Pour cet exemple, on suppose que les settings sont configurés pour le testnet si ce test est exécuté.
        if not (settings.binance.testnet and settings.binance.api_key and settings.binance.api_secret):
            pytest.skip("Testnet API keys not configured or testnet not enabled in settings. Skipping integration test.")

        client = BinanceDataClient(
            api_key=settings.binance.api_key.get_secret_value(),
            api_secret=settings.binance.api_secret.get_secret_value(),
            testnet=True # Forcer testnet pour ce client d'intégration
        )
        await client.initialize()
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_real_connection_testnet_fetch_klines(self, live_testnet_client: BinanceDataClient):
        """Test de récupération de klines réelles depuis le testnet."""
        # Utiliser une paire courante sur testnet, ex: BTCUSDT
        # Attention: les données sur testnet peuvent être sporadiques.
        symbol_to_test = "BTCUSDT" # Ou une autre paire active sur testnet
        try:
            klines = await live_testnet_client.fetch_klines(
                symbol=symbol_to_test,
                interval=Kline.INTERVAL_1MINUTE,
                limit=5
            )
            assert isinstance(klines, list)
            if klines: # Il se peut qu'il n'y ait pas de données récentes pour toutes les paires sur testnet
                assert len(klines) <= 5
                for kline in klines:
                    assert Kline.OHLCV_OPEN in kline
                    assert kline['open_time'] > 0
            else:
                logger.warning(f"No klines returned for {symbol_to_test} on testnet, this might be normal.")

        except BinanceInvalidSymbolError:
            pytest.skip(f"Symbol {symbol_to_test} might not be available on testnet for klines. Skipping.")
        except BinanceAPIError as e:
            # Certaines erreurs (ex: maintenance du testnet) peuvent survenir.
            # On peut choisir de les ignorer ou de faire échouer le test.
            if "market is closed" in str(e).lower() or "service unavailable" in str(e).lower():
                 pytest.skip(f"Testnet API error for {symbol_to_test}: {e}. Skipping.")
            raise # Relancer les autres erreurs API

