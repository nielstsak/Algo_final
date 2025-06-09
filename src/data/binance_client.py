# src/data/binance_client.py
import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import hmac
from urllib.parse import urlencode
from functools import wraps
from loguru import logger
import pandas as pd # Importation ajoutée pour le DataFrame dans fetch_klines_batch

from binance import AsyncClient, Client # Client est importé mais non utilisé directement ici
from binance.exceptions import (
    BinanceAPIException,
    BinanceRequestException,
    BinanceOrderException
)
import aiohttp

from src.core.config import settings
from src.core.constants import System, Kline, Trading
from src.core.exceptions import (
    BinanceAPIError,
    BinanceRateLimitError,
    BinanceInvalidSymbolError,
    DataDownloadError,
    APIError, # APIError est importé mais non utilisé directement ici
    KlineValidationError
)


class RateLimiter:
    """
    Gère les limites de taux de l'API Binance.

    Binance utilise un système de poids (weight) pour les requêtes:
    - Limite par défaut: 1200 requêtes par minute (configurable)
    - Chaque endpoint a un poids différent (1 à 40+)
    """

    def __init__(self, weight_limit: int = 1200, window_seconds: int = 60):
        self.weight_limit = weight_limit
        self.window_seconds = window_seconds
        self.requests: List[Tuple[float, int]] = []  # (timestamp, weight)
        self._lock = asyncio.Lock()
        logger.debug(f"RateLimiter initialized with limit: {weight_limit} / {window_seconds}s")

    async def acquire(self, weight: int = 1) -> None:
        """
        Acquiert le droit d'effectuer une requête avec le poids spécifié.
        Bloque si nécessaire pour respecter les limites.
        """
        while True: # Boucle pour réessayer après une attente
            async with self._lock: # Acquiert le verrou
                now = time.time()
                # Nettoyer les anciennes requêtes hors de la fenêtre
                self.requests = [
                    (ts, w) for ts, w in self.requests
                    if now - ts < self.window_seconds
                ]
                current_weight = sum(w for _, w in self.requests)

                if current_weight + weight <= self.weight_limit:
                    # La limite n'est pas dépassée, enregistrer la requête et sortir
                    self.requests.append((now, weight))
                    # logger.debug(f"Acquired weight {weight}. Current total: {current_weight + weight}/{self.weight_limit}")
                    return  # Sort de la méthode acquire avec succès

                # Si on arrive ici, la limite est dépassée.
                # Calculer le temps d'attente nécessaire *avant* de libérer le verrou.
                oldest_request_time = self.requests[0][0] if self.requests else now
                # Temps avant que la plus ancienne requête n'expire de la fenêtre
                time_until_oldest_expires = self.window_seconds - (now - oldest_request_time)
                # Attendre un peu plus pour s'assurer que la requête a expiré
                wait_time = max(0.01, time_until_oldest_expires + 0.1) # Attente minimale de 0.01s

            # Le verrou est automatiquement libéré ici à la sortie du bloc "async with self._lock"

            logger.warning(
                f"Rate limit exceeded or approached: current weight {current_weight}/{self.weight_limit}, "
                f"attempting to acquire {weight}. Waiting {wait_time:.2f}s before retrying."
            )
            await asyncio.sleep(wait_time)
            # La boucle "while True" va redémarrer et une nouvelle tentative d'acquisition du verrou sera faite.

    def get_current_weight(self) -> int:
        """Retourne le poids actuel utilisé dans la fenêtre courante."""
        # Cette méthode est appelée pour information, elle n'a pas besoin d'un verrou strict
        # car une légère imprécision due à la concurrence n'est généralement pas critique ici.
        # Cependant, pour une lecture parfaitement cohérente, un verrou pourrait être ajouté,
        # mais cela pourrait impacter les performances si appelé très fréquemment.
        now = time.time()
        # Filtrer les requêtes pour ne garder que celles dans la fenêtre actuelle
        # Il est important de ne pas modifier self.requests ici si la méthode est appelée
        # en dehors du contexte du verrou _lock.
        current_requests_in_window = [
            (ts, w) for ts, w in self.requests
            if now - ts < self.window_seconds
        ]
        return sum(w for _, w in current_requests_in_window)


class ExponentialBackoff:
    """
    Implémente un système de retry avec backoff exponentiel.
    """

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Calcule le délai pour une tentative donnée.

        Args:
            attempt: Numéro de la tentative (commence à 0)

        Returns:
            Délai en secondes
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            # Ajoute un jitter aléatoire pour éviter le "thundering herd"
            import random
            delay *= (0.5 + random.random()) # Applique le jitter (entre 0.5 et 1.5 fois le délai)

        return delay


def with_retry(
    max_attempts: int = System.MAX_BINANCE_API_RETRIES, # S'assurer que System est défini
    backoff: Optional[ExponentialBackoff] = None
):
    """
    Décorateur pour ajouter la logique de retry aux méthodes.
    """
    if backoff is None:
        backoff = ExponentialBackoff()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except BinanceAPIException as e:
                    last_exception = e
                    # Cas spéciaux qui ne doivent pas être retentés ou qui nécessitent une exception spécifique
                    if e.code in [-1121, -1100]:  # Invalid symbol, Illegal characters
                        raise BinanceInvalidSymbolError(
                            f"Invalid symbol: {e.message}",
                            binance_code=e.code,
                            original_exception=e
                        )
                    elif e.code == -1003:  # Rate limit (Too many requests)
                        # Normalement, le RateLimiter devrait gérer cela.
                        # Si cette exception est quand même levée, c'est un problème plus grave.
                        logger.error(f"BinanceRateLimitError (code -1003) encountered despite RateLimiter for {func.__name__}. This might indicate an issue with weight calculation or external rate limiting. Error: {e}")
                        raise BinanceRateLimitError(
                            "Rate limit exceeded (BinanceAPIException code -1003)",
                            status_code= getattr(e, 'status_code', 429), # Utiliser le status_code de l'exception si disponible
                            binance_code=e.code,
                            original_exception=e
                        )
                    elif e.code in [-2010, -2011]:  # New order rejected (e.g. insufficient balance), Cancel rejected
                        # Ces erreurs ne sont généralement pas transitoires et ne devraient pas être retentées.
                        logger.error(f"Non-retryable Binance API error (code {e.code}) for {func.__name__}: {e.message}")
                        raise BinanceAPIError( # Ou une exception plus spécifique si définie (ex: BinanceInsufficientFundsError)
                            f"Non-retryable API error: {e.message}",
                            status_code=getattr(e, 'status_code', None),
                            binance_code=e.code,
                            original_exception=e
                        )

                    # Pour les autres erreurs API générales potentiellement transitoires (ex: 5xx server errors)
                    if attempt < max_attempts - 1:
                        delay = backoff.get_delay(attempt)
                        logger.warning(
                            f"API error (attempt {attempt + 1}/{max_attempts}) for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"API error after {max_attempts} attempts for {func.__name__}: {e}")
                        # Relancer après épuisement des tentatives
                        raise BinanceAPIError(
                            f"Failed after {max_attempts} attempts due to API error in {func.__name__}: {e.message}",
                            original_exception=last_exception # Conserver l'exception originale
                        )

                except (BinanceRequestException, aiohttp.ClientError) as e: # Erreurs réseau/requête
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff.get_delay(attempt)
                        logger.warning(
                            f"Network/request error (attempt {attempt + 1}/{max_attempts}) for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Network/request error after {max_attempts} attempts for {func.__name__}: {e}")
                        raise BinanceAPIError( # Ou une NetworkError plus spécifique
                            f"Failed after {max_attempts} attempts due to network/request error in {func.__name__}: {e}",
                            original_exception=last_exception
                        )

                except Exception as e: # Capturer d'autres exceptions non prévues
                    logger.exception(f"Unexpected error in {func.__name__} (not retrying): {e}") # Utiliser logger.exception pour inclure le traceback
                    raise # Relancer les exceptions non prévues et non gérées pour ne pas les masquer

            # Ce point ne devrait pas être atteint si une exception est toujours levée après les tentatives
            # Mais pour être sûr, si la boucle se termine sans succès et sans exception propagée :
            if last_exception: # S'assurer qu'il y a une exception à lever
                 raise BinanceAPIError(
                    f"Function {func.__name__} failed after {max_attempts} attempts. Last error: {last_exception}",
                    original_exception=last_exception
                )
            # Fallback si last_exception n'est pas défini, bien que cela soit peu probable
            # Cela pourrait arriver si max_attempts est 0 ou négatif, ce qui est une mauvaise configuration.
            raise BinanceAPIError(f"Function {func.__name__} failed after {max_attempts} attempts with an unknown error (last_exception was None).")

        return wrapper
    return decorator


class BinanceDataClient:
    """
    Client spécialisé pour la récupération des données depuis Binance.
    Implémente DAT-001 à DAT-005 du Kanban.
    """

    # Poids des différents endpoints (peut nécessiter une mise à jour selon la doc API Binance)
    ENDPOINT_WEIGHTS = {
        'klines': 1,                # GET /api/v3/klines
        'exchangeInfo': 10,         # GET /api/v3/exchangeInfo
        'depth': {                  # GET /api/v3/depth
            'limit_1_100': 1,       # Pour limit <= 100
            'limit_500': 5,
            'limit_1000': 10,
            'limit_5000': 50,
        },
        'trades': 1,                # GET /api/v3/trades (Récent)
        'historicalTrades': 5,      # GET /api/v3/historicalTrades (Nécessite clé API)
        'ticker24hr': {'symbol': 1, 'all': 40}, # GET /api/v3/ticker/24hr
        'account': 10,              # GET /api/v3/account (Nécessite clé API)
        'myTrades': 10,             # GET /api/v3/myTrades (Nécessite clé API)
        'ping': 1,                  # GET /api/v3/ping
        'time': 1,                  # GET /api/v3/time
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        use_second_key: bool = False # Pour utiliser la deuxième paire de clés de la config
    ):
        """
        Initialise le client Binance.
        DAT-002: Gestion de l'authentification et des clés API.
        """
        resolved_api_key = api_key
        resolved_api_secret = api_secret

        # Utilisation des settings pour les clés API si non fournies directement
        if resolved_api_key is None or resolved_api_secret is None:
            active_binance_settings = settings.binance
            if use_second_key and active_binance_settings.api_key_2 and active_binance_settings.api_secret_2:
                resolved_api_key = active_binance_settings.api_key_2.get_secret_value()
                resolved_api_secret = active_binance_settings.api_secret_2.get_secret_value()
                logger.info("Using secondary Binance API key for BinanceDataClient.")
            else:
                # S'assurer que les clés primaires existent avant de les utiliser
                if not active_binance_settings.api_key or not active_binance_settings.api_secret:
                    # Si les clés ne sont pas fournies et non présentes dans les settings, c'est une erreur de configuration
                    # Cependant, certains endpoints publics n'ont pas besoin de clés.
                    # Le client python-binance peut être initialisé sans clés pour ces cas.
                    logger.warning("Primary API key/secret not found in settings. Client will operate in public mode if no keys provided.")
                    resolved_api_key = None # Explicitement None
                    resolved_api_secret = None # Explicitement None
                else:
                    resolved_api_key = active_binance_settings.api_key.get_secret_value()
                    resolved_api_secret = active_binance_settings.api_secret.get_secret_value()
                    logger.info("Using primary Binance API key for BinanceDataClient.")


        self.api_key = resolved_api_key
        self.api_secret = resolved_api_secret
        # Priorité à l'argument testnet, sinon utiliser la valeur des settings
        self.testnet = testnet if testnet is not None else settings.binance.testnet

        self.client: Optional[AsyncClient] = None
        self.rate_limiter = RateLimiter() # DAT-003: Gestion des rate limits
        # DAT-004 (Système de retry) est géré par le décorateur @with_retry

        # Cache pour les informations d'exchange
        self._exchange_info_cache: Optional[Dict] = None
        self._exchange_info_cache_time: Optional[float] = None
        # TTL pour le cache d'exchangeInfo (ex: 1 heure)
        self._exchange_info_cache_ttl: float = settings.data.cache_ttl if settings.data.cache_enabled else 3600

        logger.info(f"BinanceDataClient configured (testnet={self.testnet}). API key {'provided' if self.api_key else 'not provided'}.")

    async def initialize(self):
        """Initialise la connexion au client Binance."""
        try:
            # AsyncClient.create peut accepter api_key=None et api_secret=None
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            await self._test_connection() # Teste la connexion après initialisation
            logger.success("Binance AsyncClient initialized and connection tested successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Binance AsyncClient: {e}")
            # Rendre l'exception plus spécifique si possible
            raise BinanceAPIError("Failed to initialize Binance client", original_exception=e)

    async def _test_connection(self):
        """Teste la connexion à l'API Binance."""
        if not self.client: # Vérification au cas où initialize n'aurait pas été appelé
            raise BinanceAPIError("Client not initialized before testing connection.")
        try:
            # Utiliser un endpoint léger comme ping ou time qui a un faible poids
            await self.rate_limiter.acquire(weight=self.ENDPOINT_WEIGHTS['ping'])
            # server_time = await self.client.get_server_time() # get_server_time a un poids de 1
            await self.client.ping() # ping a un poids de 1
            logger.info(f"Successfully connected to Binance (ping successful).")
        except Exception as e:
            logger.error(f"Binance connection test (ping) failed: {e}")
            raise BinanceAPIError("Binance connection test failed", original_exception=e)

    async def close(self):
        """Ferme la connexion du client Binance."""
        if self.client:
            try:
                await self.client.close_connection()
                logger.info("Binance AsyncClient connection closed.")
            except Exception as e:
                logger.error(f"Error closing Binance AsyncClient connection: {e}")
        self.client = None # S'assurer que le client est None après la fermeture

    async def __aenter__(self):
        """Supporte l'utilisation asynchrone avec 'async with'."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage à la sortie du contexte 'async with'."""
        await self.close()

    @with_retry()
    async def get_exchange_info(self, force_refresh: bool = False) -> Dict:
        """
        Récupère les informations sur les paires de trading (symboles, filtres, etc.).
        Utilise un cache pour éviter les requêtes répétées.
        """
        now = time.time()
        if (not force_refresh and
            self._exchange_info_cache and
            self._exchange_info_cache_time and
            (now - self._exchange_info_cache_time < self._exchange_info_cache_ttl)):
            logger.debug("Returning cached exchange info.")
            return self._exchange_info_cache

        if not self.client: # Vérification supplémentaire
            raise BinanceAPIError("Client not initialized. Call initialize() first.")

        await self.rate_limiter.acquire(self.ENDPOINT_WEIGHTS['exchangeInfo'])
        logger.debug("Fetching fresh exchange info from Binance.")
        exchange_info_data = await self.client.get_exchange_info()

        # Mettre à jour le cache
        self._exchange_info_cache = exchange_info_data
        self._exchange_info_cache_time = now
        return exchange_info_data

    def _validate_kline_data(self, kline_raw: List, symbol: str) -> Dict[str, Any]:
        """
        Valide et formate une kline individuelle.
        DAT-005: Implémenter la validation des réponses API (partiellement ici pour les klines).
        """
        try:
            # Structure attendue d'une kline de l'API Binance:
            # [ open_time, open, high, low, close, volume, close_time, quote_asset_volume,
            #   number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore ]
            if len(kline_raw) < 11: # Souvent 12, mais au moins 11 champs sont critiques
                raise ValueError(f"Kline data is incomplete. Expected at least 11 fields, got {len(kline_raw)}")

            # Conversion et validation des types. Utiliser Decimal pour la précision monétaire.
            open_time_ms = int(kline_raw[0])
            open_price = Decimal(str(kline_raw[1]))
            high_price = Decimal(str(kline_raw[2]))
            low_price = Decimal(str(kline_raw[3]))
            close_price = Decimal(str(kline_raw[4]))
            base_volume = Decimal(str(kline_raw[5])) # Volume de l'actif de base
            close_time_ms = int(kline_raw[6])
            quote_volume = Decimal(str(kline_raw[7])) # Volume de l'actif de cotation
            num_trades = int(kline_raw[8])
            taker_buy_base_volume = Decimal(str(kline_raw[9]))
            taker_buy_quote_volume = Decimal(str(kline_raw[10]))
            # kline_raw[11] est 'ignore', non utilisé.

            # Validations logiques
            if high_price < low_price:
                raise ValueError(f"High price {high_price} is less than low price {low_price}")
            
            current_prices_for_ohlc = [open_price, close_price]
            # Only include high and low if they are not causing the issue initially
            if high_price >= low_price:
                current_prices_for_ohlc.extend([high_price, low_price])
            else: # high < low, which is already an issue, use only open/close for range
                logger.warning(f"Initial high {high_price} < low {low_price} for {symbol} at {open_time_ms}. OHLC will be based on O/C first.")


            # S'assurer que open/close sont dans les bornes high/low
            # Recalculer high et low pour englober open et close, et les valeurs initiales de H/L si valides
            final_high_price = max(current_prices_for_ohlc)
            final_low_price = min(current_prices_for_ohlc)

            if not (final_low_price <= open_price <= final_high_price and final_low_price <= close_price <= final_high_price):
                 logger.warning(f"OHLC prices for {symbol} at {open_time_ms} were not strictly ordered even after adjustment: "
                               f"O={open_price}, H={high_price} (orig), L={low_price} (orig), C={close_price}. "
                               f"Final H={final_high_price}, Final L={final_low_price}. Data might be unusual.")


            if base_volume < Decimal(0): # Le volume ne peut pas être négatif
                raise ValueError(f"Base asset volume {base_volume} is negative")
            if close_time_ms < open_time_ms: # close_time doit être >= open_time
                 # Pour une kline de 1m, close_time est typiquement open_time + 59999ms
                raise ValueError(f"Close time {close_time_ms} is before or same as open time {open_time_ms}")

            # Retourner un dictionnaire avec des types Python standards (float pour la plupart des usages)
            # Les noms de clés ici sont ceux que DataManager s'attend à recevoir pour son renommage.
            return {
                'open_time': open_time_ms,
                'open': float(final_low_price if open_price < final_low_price else (final_high_price if open_price > final_high_price else open_price)), # Clamped open
                'high': float(final_high_price),
                'low': float(final_low_price),
                'close': float(final_low_price if close_price < final_low_price else (final_high_price if close_price > final_high_price else close_price)), # Clamped close
                'volume': float(base_volume),
                'close_time': close_time_ms,
                'quote_asset_volume': float(quote_volume),
                'number_of_trades': num_trades,
                'taker_buy_base_asset_volume': float(taker_buy_base_volume),
                'taker_buy_quote_asset_volume': float(taker_buy_quote_volume)
            }
        except (ValueError, TypeError, IndexError) as e: # Attraper les erreurs de conversion ou d'index
            logger.error(f"Validation failed for kline data: {kline_raw} for symbol {symbol}. Error: {e}")
            # Lever une exception personnalisée pour une meilleure gestion des erreurs en amont
            raise KlineValidationError(
                f"Invalid kline data for {symbol}",
                symbol=symbol,
                timestamp=kline_raw[0] if kline_raw and len(kline_raw) > 0 else None, # Fournir le timestamp si possible
                original_exception=e
            )

    @with_retry()
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[int] = None, # Attendre un timestamp en millisecondes
        end_time: Optional[int] = None,   # Attendre un timestamp en millisecondes
        limit: int = 1000 # Limite par défaut de l'API Binance est 500, max 1000
    ) -> List[Dict[str, Any]]:
        """
        Récupère les klines pour un symbole donné.
        DAT-001: Implémenter BinanceClient avec python-binance (indirectement via AsyncClient)
        DAT-005: Implémenter la validation des réponses API (via _validate_kline_data)
        """
        if not self.client: # Vérification que le client est initialisé
            raise BinanceAPIError("Client not initialized. Call initialize() first.")

        # Valider le symbole en utilisant les informations d'exchange (mises en cache)
        exchange_info = await self.get_exchange_info() # Peut lever une exception si l'API échoue
        if not any(s['symbol'] == symbol for s in exchange_info['symbols']):
            raise BinanceInvalidSymbolError(f"Symbol '{symbol}' not found in exchange info.", symbol=symbol)

        # S'assurer que la limite ne dépasse pas le maximum autorisé par Binance
        api_max_limit = System.MAX_KLINES_PER_BINANCE_REQUEST # Ex: 1000
        if limit > api_max_limit:
            logger.warning(f"Requested kline limit {limit} exceeds API maximum {api_max_limit}. Adjusting to {api_max_limit}.")
            limit = api_max_limit

        # Préparer les paramètres pour la requête API
        params: Dict[str, Any] = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time

        # Acquérir le droit d'effectuer la requête via le RateLimiter
        await self.rate_limiter.acquire(self.ENDPOINT_WEIGHTS['klines'])
        logger.debug(f"Fetching klines for {symbol} with params: {params}")

        # Effectuer la requête API
        raw_klines = await self.client.get_klines(**params) # Peut lever BinanceAPIException, BinanceRequestException

        # Valider et formater chaque kline reçue
        validated_klines = []
        for k_raw in raw_klines:
            try:
                validated_kline = self._validate_kline_data(k_raw, symbol)
                validated_klines.append(validated_kline)
            except KlineValidationError as e:
                # Enregistrer l'erreur et continuer avec les klines valides
                # Une alternative serait de lever une exception si une seule kline est invalide,
                # mais cela pourrait interrompre le téléchargement de lots importants.
                logger.warning(f"Skipping invalid kline for {symbol} at {k_raw[0] if k_raw else 'N/A'}: {e.message}")

        logger.info(f"Fetched {len(validated_klines)} valid klines for {symbol} (out of {len(raw_klines)} raw klines received).")
        return validated_klines

    def _get_interval_milliseconds(self, interval_str: str) -> int:
        """Convertit une chaîne d'intervalle Binance (ex: '1m', '1h', '1d') en millisecondes."""
        # Multiplicateurs pour convertir les unités en secondes
        multipliers_to_seconds = {'m': 60, 'h': 60*60, 'd': 24*60*60, 'w': 7*24*60*60}
        try:
            value = int(interval_str[:-1]) # Extraire la partie numérique
            unit = interval_str[-1].lower()  # Extraire l'unité et la mettre en minuscule

            if unit == 'm' and interval_str[-1] == 'M': # Gérer 'M' pour mois (approximation)
                # Binance utilise 'M' pour mois. Approximation à 30 jours.
                # L'API Binance gère cela correctement ; ceci est pour les calculs de durée.
                return value * 30 * 24 * 60 * 60 * 1000
            elif unit in multipliers_to_seconds:
                return value * multipliers_to_seconds[unit] * 1000 # Convertir en millisecondes
            else:
                logger.error(f"Unknown interval unit: {unit} in '{interval_str}'. Defaulting to 1 minute.")
                return 60 * 1000 # Défaut à 1 minute
        except (KeyError, ValueError, TypeError) as e: # Erreurs de parsing
            logger.error(f"Invalid interval string: '{interval_str}'. Defaulting to 1 minute (60000ms). Error: {e}")
            return 60 * 1000 # Défaut à 1 minute en cas d'erreur

    async def fetch_klines_batch(
        self,
        symbol: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[int] = None, # Attendre un timestamp en millisecondes
        end_time: Optional[int] = None,   # Attendre un timestamp en millisecondes
        progress_callback: Optional[Callable[[int, int, Optional[int]], None]] = None # (klines_telechargees_batch, klines_estimees_total_pour_paire, last_kline_ts_in_batch)
    ) -> List[Dict[str, Any]]:
        """
        Récupère un grand nombre de klines en effectuant plusieurs requêtes, gérant la pagination.
        DAT-007: Implémenter la pagination des résultats.
        DAT-008: Gérer les données manquantes (implicitement, si l'API ne retourne rien pour une période).
        DAT-009: Créer le système de reprise après échec (géré par le décorateur @with_retry pour les erreurs API individuelles,
                 mais une reprise de lot plus large nécessiterait un checkpointing externe, non implémenté ici).
        """
        if not self.client: # S'assurer que le client est initialisé
            raise BinanceAPIError("Client not initialized. Call initialize() first.")

        # Définir les bornes temporelles si non fournies
        current_time_ms = int(datetime.now().timestamp() * 1000)
        if end_time is None:
            end_time = current_time_ms # Heure actuelle
        if start_time is None: # Par défaut, récupérer les 30 derniers jours si start_time n'est pas fourni
            start_time = end_time - (30 * 24 * 60 * 60 * 1000) # 30 jours en millisecondes
        
        # S'assurer que start_time n'est pas dans le futur et end_time n'est pas avant start_time
        if start_time >= end_time:
            logger.warning(f"Start time {start_time} is after or equal to end time {end_time} for {symbol}. No klines will be fetched.")
            return []
        if start_time > current_time_ms : # Ne pas essayer de fetcher des données du futur
            logger.warning(f"Start time {start_time} is in the future for {symbol}. Adjusting to current time.")
            start_time = current_time_ms
            if start_time >= end_time: return []


        all_klines: List[Dict[str, Any]] = []
        current_fetch_start_time = start_time
        interval_ms = self._get_interval_milliseconds(interval)
        if interval_ms == 0: # Eviter division par zéro
            raise ValueError(f"Interval {interval} resulted in 0 milliseconds duration.")
            
        limit_per_request = System.MAX_KLINES_PER_BINANCE_REQUEST # Limite de klines par requête API

        # Estimer le nombre total de klines pour le callback de progression
        total_estimated_klines_for_period = (end_time - start_time) // interval_ms
        
        # Variable pour suivre le total de klines téléchargées pour cette paire dans cette session de fetch_klines_batch
        total_klines_downloaded_for_pair_session = 0

        logger.info(f"Starting batch kline download for {symbol} (interval: {interval}) "
                    f"from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}."
                    f" Estimated klines for period: {total_estimated_klines_for_period if total_estimated_klines_for_period > 0 else 'N/A'}")

        while current_fetch_start_time < end_time:
            try:
                batch_potential_end_time = current_fetch_start_time + (limit_per_request * interval_ms) - interval_ms 
                current_batch_end_time = min(batch_potential_end_time, end_time -1) 

                if current_fetch_start_time > current_batch_end_time :
                     logger.debug(f"Calculated current_fetch_start_time ({datetime.fromtimestamp(current_fetch_start_time/1000)}) "
                                  f"is > current_batch_end_time ({datetime.fromtimestamp(current_batch_end_time/1000)}). Ending batch download.")
                     break

                klines_in_current_api_call = await self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_fetch_start_time,
                    end_time=current_batch_end_time, 
                    limit=limit_per_request
                )

                if not klines_in_current_api_call: 
                    logger.info(f"No more klines returned for {symbol} starting {datetime.fromtimestamp(current_fetch_start_time/1000)}. "
                                f"Batch download likely complete for this period.")
                    break
                
                # Filtrer pour s'assurer que les klines sont dans la plage [start_time, end_time)
                # et éviter les doublons si les API se chevauchent légèrement.
                # La déduplication finale gérera les doublons plus robustement.
                new_klines_to_add_this_batch = [k for k in klines_in_current_api_call if k['open_time'] < end_time]


                if new_klines_to_add_this_batch:
                    all_klines.extend(new_klines_to_add_this_batch)
                    total_klines_downloaded_for_pair_session += len(new_klines_to_add_this_batch)
                    
                    last_kline_ts_in_batch = new_klines_to_add_this_batch[-1]['open_time']

                    if progress_callback: 
                        # Le callback attend (klines_telechargees_CE_BATCH, klines_estimees_TOTAL_POUR_PAIRE, last_kline_ts_in_batch)
                        # Ici, on fournit le nombre de klines dans ce batch spécifique, et l'estimation totale pour la période demandée.
                        progress_callback(len(new_klines_to_add_this_batch), total_estimated_klines_for_period, last_kline_ts_in_batch)
                else: # No new klines to add, probably means we fetched beyond end_time or got empty list
                    if progress_callback: # Informer qu'aucun kline n'a été ajouté dans ce batch
                         progress_callback(0, total_estimated_klines_for_period, None)


                # Mettre à jour current_fetch_start_time pour la prochaine itération.
                last_processed_kline_close_time_ms = klines_in_current_api_call[-1]['close_time']
                next_start_time = last_processed_kline_close_time_ms + 1

                if next_start_time <= current_fetch_start_time :
                    logger.warning(f"No progress in kline batch download for {symbol}. "
                                   f"Last kline time: {datetime.fromtimestamp(klines_in_current_api_call[-1]['open_time']/1000)}. Breaking loop.")
                    break
                current_fetch_start_time = next_start_time

                if len(klines_in_current_api_call) < limit_per_request:
                    logger.info(f"Batch for {symbol} returned {len(klines_in_current_api_call)} klines (less than limit {limit_per_request}), "
                                f"assuming end of available data for the period.")
                    break

                await asyncio.sleep(0.1) 

            except BinanceInvalidSymbolError as e: 
                logger.error(f"Invalid symbol {symbol} encountered during batch kline fetch: {e}")
                raise 
            except BinanceRateLimitError as e: 
                logger.error(f"Rate limit hit during batch kline fetch for {symbol}: {e}. ")
                raise 
            except Exception as e: 
                logger.error(f"Error during kline batch download for {symbol} "
                               f"at start_time {datetime.fromtimestamp(current_fetch_start_time/1000)}: {e}", exc_info=True)
                raise DataDownloadError(
                    message=f"Failed to download kline batch for {symbol}",
                    source="Binance",
                    symbol=symbol,
                    original_exception=e
                )

        # Déduplication finale
        if all_klines:
            try:
                # Convertir en DataFrame pour une déduplication et un tri faciles
                df_temp = pd.DataFrame(all_klines)
                # S'assurer que 'open_time' est unique, en gardant la première occurrence
                df_temp.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
                # Trier par 'open_time' pour s'assurer de l'ordre chronologique
                df_temp.sort_values(by='open_time', inplace=True)
                # Reconvertir en liste de dictionnaires
                all_klines = df_temp.to_dict('records')
            except Exception as e_df:
                logger.error(f"Error during final deduplication/sorting of klines for {symbol}: {e_df}. Proceeding with potentially non-unique or unsorted data.")


        logger.success(f"Completed batch kline download for {symbol}. Total unique klines fetched: {len(all_klines)}")
        return all_klines

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations détaillées pour un symbole spécifique (filtres de prix, de quantité, etc.).
        """
        try:
            exchange_info = await self.get_exchange_info() # Utilise le cache interne
            for sym_info in exchange_info.get('symbols', []): # S'assurer que 'symbols' existe
                if sym_info.get('symbol') == symbol:
                    return sym_info
            logger.warning(f"Symbol {symbol} not found in exchange information.")
            return None # Retourner None explicitement si non trouvé
        except BinanceAPIError as e: # Si get_exchange_info échoue
            logger.error(f"API error while trying to get symbol info for {symbol}: {e}")
            raise # Relancer l'erreur API
        except Exception as e: # Autres erreurs inattendues
            logger.error(f"Unexpected error fetching symbol info for {symbol}: {e}")
            # Envelopper dans une BinanceAPIError ou une exception personnalisée plus générique
            raise BinanceAPIError(f"Could not get symbol info for {symbol} due to an unexpected error.", original_exception=e)

    async def validate_trading_pair(self, symbol: str) -> bool:
        """
        Vérifie si une paire de trading est valide, active pour le trading,
        et correspond aux critères de configuration (ex: devise de cotation).
        """
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info: # Si get_symbol_info retourne None
                logger.warning(f"Validation failed for {symbol}: Symbol not found in exchange information.")
                return False

            # Vérifier le statut du symbole (doit être 'TRADING')
            if symbol_info.get('status') != 'TRADING':
                logger.warning(f"Validation failed for {symbol}: Status is '{symbol_info.get('status')}', not 'TRADING'.")
                return False

            # Vérifier si la devise de cotation correspond à celle configurée dans settings.trading.base_currency
            # Note: Dans la terminologie de Binance, pour "BTCUSDC", "USDC" est 'quoteAsset' et "BTC" est 'baseAsset'.
            # settings.trading.base_currency se réfère à l'actif dans lequel on veut évaluer les profits, donc la 'quoteAsset'.
            expected_quote_asset = settings.trading.base_currency
            if symbol_info.get('quoteAsset') != expected_quote_asset:
                logger.warning(f"Validation failed for {symbol}: Quote asset is '{symbol_info.get('quoteAsset')}', "
                               f"but expected '{expected_quote_asset}' based on configuration (settings.trading.base_currency).")
                return False

            # Vérifier si la paire est dans la liste des paires autorisées de la configuration
            if settings.trading.allowed_pairs and symbol not in settings.trading.allowed_pairs:
                logger.warning(f"Validation failed for {symbol}: Not in 'allowed_pairs' list in configuration.")
                return False

            # D'autres validations pourraient être ajoutées ici (ex: vérifier les permissions de trading pour cette paire)
            logger.debug(f"Trading pair {symbol} validated successfully against exchange info and configuration.")
            return True

        except BinanceAPIError as e: # Si get_symbol_info ou get_exchange_info lève une erreur API
            logger.error(f"API error during validation of trading pair {symbol}: {e}")
            return False # Échec de la validation en cas d'erreur API
        except Exception as e: # Gérer toute autre exception inattendue
            logger.error(f"Unexpected error during validation of trading pair {symbol}: {e}")
            return False # Échec de la validation en cas d'erreur inattendue

# Exemple d'utilisation (peut être commenté ou retiré pour la production)
async def example_usage():
    """Exemple d'utilisation du BinanceDataClient."""
    logger.info("Starting BinanceDataClient example usage...")
    # Utilisation avec context manager pour s'assurer que initialize et close sont appelés
    try:
        async with BinanceDataClient(testnet=True) as client: # Mettre testnet=True pour les tests
            # Vérifier une paire
            is_valid_btcusdc = await client.validate_trading_pair("BTCUSDT") # USDT est plus commun sur testnet
            print(f"BTCUSDT (testnet) is valid for trading based on config: {is_valid_btcusdc}")

            is_valid_ethusdc = await client.validate_trading_pair("ETHUSDT")
            print(f"ETHUSDT (testnet) is valid for trading based on config: {is_valid_ethusdc}")

            # Récupérer quelques klines récentes pour une paire valide du testnet
            if is_valid_btcusdc : # ou une autre paire valide sur testnet
                target_symbol = "BTCUSDT"
                print(f"\nFetching recent klines for {target_symbol}...")
                klines = await client.fetch_klines(
                    symbol=target_symbol,
                    interval=Kline.INTERVAL_1MINUTE,
                    limit=5 # Demander un petit nombre pour l'exemple
                )
                print(f"Fetched {len(klines)} klines for {target_symbol}.")
                if klines:
                    for k in klines:
                        print(f"  - Time: {datetime.fromtimestamp(k['open_time']/1000)}, Close: {k['close']}")
            else:
                print(f"\nSkipping kline fetch as BTCUSDT (testnet) is not considered valid by current config.")


            # Récupérer un historique plus large pour une paire valide
            if is_valid_btcusdc: # ou une autre paire valide sur testnet
                target_symbol_batch = "BTCUSDT"
                print(f"\nFetching historical klines for {target_symbol_batch} (last 1 hour)...")
                end_time_ms = int(datetime.now().timestamp() * 1000)
                start_time_ms = end_time_ms - (1 * 60 * 60 * 1000)  # Les 60 dernières minutes

                def progress_update(klines_in_batch, total_expected_for_pair, last_ts): # Adapté aux nouveaux params
                    percentage = (klines_in_batch / total_expected_for_pair * 100) if total_expected_for_pair > 0 and klines_in_batch <= total_expected_for_pair else 0 # Approximation
                    # Note: klines_in_batch est pour le batch actuel, total_expected_for_pair est pour toute la période.
                    # Un meilleur suivi de la progression globale nécessiterait de cumuler klines_in_batch.
                    print(f"  Batch Progress: Received {klines_in_batch} klines. (Total Period Est: {total_expected_for_pair}) Last TS in batch: {last_ts}")


                historical_klines = await client.fetch_klines_batch(
                    symbol=target_symbol_batch,
                    interval=Kline.INTERVAL_1MINUTE,
                    start_time=start_time_ms,
                    end_time=end_time_ms,
                    progress_callback=progress_update
                )
                print(f"Downloaded {len(historical_klines)} historical klines for {target_symbol_batch}.")
                if len(historical_klines) > 0:
                     print(f"  First kline: Time: {datetime.fromtimestamp(historical_klines[0]['open_time']/1000)}, Close: {historical_klines[0]['close']}")
                     print(f"  Last kline:  Time: {datetime.fromtimestamp(historical_klines[-1]['open_time']/1000)}, Close: {historical_klines[-1]['close']}")


            # Afficher les limites de taux actuelles (approximatif)
            current_api_weight = client.rate_limiter.get_current_weight()
            print(f"\nCurrent API rate limit weight usage (approx): {current_api_weight}/{client.rate_limiter.weight_limit}")

    except BinanceAPIError as e:
        print(f"Binance API Error during example: {e}")
    except DataDownloadError as e:
        print(f"Data Download Error during example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during example: {e}")

    logger.info("BinanceDataClient example usage finished.")


if __name__ == "__main__":
    # Configuration du logging pour voir les messages de log de l'exemple
    # Ceci est un setup minimal, votre application aura une configuration plus complète.
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info("Running BinanceDataClient example directly.")

    # Pour exécuter cet exemple, assurez-vous que votre configuration (settings) est accessible.
    # Si settings.binance.api_key n'est pas défini, le client fonctionnera en mode public.
    # Pour les tests sur testnet, assurez-vous que settings.binance.testnet = True ou passez testnet=True au constructeur.
    # Et que les paires comme 'BTCUSDT' sont dans settings.trading.allowed_pairs et que settings.trading.base_currency est 'USDT' si vous testez avec USDT.
    asyncio.run(example_usage())
