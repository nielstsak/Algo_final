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

from binance import AsyncClient, Client
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
    APIError,
    KlineValidationError
)


class RateLimiter:
    """
    Gère les limites de taux de l'API Binance.
    
    Binance utilise un système de poids (weight) pour les requêtes:
    - Limite par défaut: 1200 requêtes par minute
    - Chaque endpoint a un poids différent (1 à 40+)
    """
    
    def __init__(self, weight_limit: int = 1200, window_seconds: int = 60):
        self.weight_limit = weight_limit
        self.window_seconds = window_seconds
        self.requests: List[Tuple[float, int]] = []  # (timestamp, weight)
        self._lock = asyncio.Lock()
        
    async def acquire(self, weight: int = 1) -> None:
        """
        Acquiert le droit d'effectuer une requête avec le poids spécifié.
        Bloque si nécessaire pour respecter les limites.
        """
        async with self._lock:
            now = time.time()
            
            # Nettoyer les anciennes requêtes hors de la fenêtre
            self.requests = [
                (ts, w) for ts, w in self.requests 
                if now - ts < self.window_seconds
            ]
            
            # Calculer le poids actuel
            current_weight = sum(w for _, w in self.requests)
            
            # Si on dépasse la limite, attendre
            if current_weight + weight > self.weight_limit:
                # Calculer le temps d'attente nécessaire
                oldest_request_time = self.requests[0][0] if self.requests else now
                wait_time = max(0, self.window_seconds - (now - oldest_request_time) + 0.1)
                
                logger.warning(
                    f"Rate limit approaching: current weight {current_weight}/{self.weight_limit}. "
                    f"Waiting {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                
                # Réessayer récursivement après l'attente
                await self.acquire(weight)
            else:
                # Enregistrer cette requête
                self.requests.append((now, weight))
                
    def get_current_weight(self) -> int:
        """Retourne le poids actuel utilisé dans la fenêtre courante."""
        now = time.time()
        self.requests = [
            (ts, w) for ts, w in self.requests 
            if now - ts < self.window_seconds
        ]
        return sum(w for _, w in self.requests)


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
            delay *= (0.5 + random.random())
            
        return delay


def with_retry(
    max_attempts: int = System.MAX_BINANCE_API_RETRIES,
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
                    
                    # Cas spéciaux qui ne doivent pas être retentés
                    if e.code in [-1121, -1100]:  # Invalid symbol, Illegal characters
                        raise BinanceInvalidSymbolError(
                            f"Invalid symbol: {e.message}",
                            binance_code=e.code,
                            original_exception=e
                        )
                    elif e.code == -1003:  # Rate limit
                        raise BinanceRateLimitError(
                            "Rate limit exceeded",
                            status_code=429,
                            binance_code=e.code,
                            original_exception=e
                        )
                    elif e.code in [-2010, -2011]:  # Insufficient funds
                        raise BinanceAPIError(
                            f"Insufficient funds: {e.message}",
                            status_code=e.status_code,
                            binance_code=e.code,
                            original_exception=e
                        )
                    
                    # Pour les autres erreurs, attendre avant de réessayer
                    if attempt < max_attempts - 1:
                        delay = backoff.get_delay(attempt)
                        logger.warning(
                            f"API error (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"API error after {max_attempts} attempts: {e}")
                        
                except (BinanceRequestException, aiohttp.ClientError) as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = backoff.get_delay(attempt)
                        logger.warning(
                            f"Network error (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Network error after {max_attempts} attempts: {e}")
                        
                except Exception as e:
                    # Erreurs inattendues ne sont pas retentées
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
                    
            # Si on arrive ici, toutes les tentatives ont échoué
            raise BinanceAPIError(
                f"Failed after {max_attempts} attempts",
                original_exception=last_exception
            )
            
        return wrapper
    return decorator


class BinanceDataClient:
    """
    Client spécialisé pour la récupération des données depuis Binance.
    Implémente DAT-001 à DAT-005.
    """
    
    # Poids des différents endpoints
    ENDPOINT_WEIGHTS = {
        'klines': 1,
        'exchangeInfo': 10,
        'depth': {'limit_5': 1, 'limit_10': 1, 'limit_20': 1, 'limit_50': 1, 
                  'limit_100': 1, 'limit_500': 5, 'limit_1000': 10},
        'trades': 1,
        'historicalTrades': 5,
        'ticker24hr': {'symbol': 1, 'all': 40},
        'account': 10,
        'myTrades': 10
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        use_second_key: bool = False
    ):
        """
        Initialise le client Binance.
        
        Args:
            api_key: Clé API (si None, utilise les settings)
            api_secret: Secret API (si None, utilise les settings)
            testnet: Utiliser le testnet Binance
            use_second_key: Utiliser la deuxième paire de clés
        """
        # Gestion de l'authentification (DAT-002)
        if api_key is None or api_secret is None:
            if use_second_key and settings.binance.api_key_2 and settings.binance.api_secret_2:
                api_key = settings.binance.api_key_2.get_secret_value()
                api_secret = settings.binance.api_secret_2.get_secret_value()
                logger.info("Using secondary API keys")
            else:
                api_key = settings.binance.api_key.get_secret_value()
                api_secret = settings.binance.api_secret.get_secret_value()
                logger.info("Using primary API keys")
                
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet or settings.binance.testnet
        
        # Client asynchrone Binance
        self.client: Optional[AsyncClient] = None
        
        # Gestion des rate limits (DAT-003)
        self.rate_limiter = RateLimiter()
        
        # Système de retry (DAT-004)
        self.backoff = ExponentialBackoff()
        
        # Cache pour les informations d'exchange
        self._exchange_info_cache: Optional[Dict] = None
        self._exchange_info_cache_time: Optional[float] = None
        self._exchange_info_cache_ttl: float = 3600  # 1 heure
        
        logger.info(f"BinanceDataClient initialized (testnet={self.testnet})")
        
    async def __aenter__(self):
        """Contexte manager pour l'initialisation asynchrone."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Contexte manager pour le nettoyage."""
        await self.close()
        
    async def initialize(self):
        """Initialise la connexion au client Binance."""
        try:
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            logger.success("Binance client initialized successfully")
            
            # Tester la connexion
            await self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise BinanceAPIError(
                "Failed to initialize Binance client",
                original_exception=e
            )
            
    async def close(self):
        """Ferme proprement les connexions."""
        if self.client:
            await self.client.close_connection()
            logger.info("Binance client connection closed")
            
    async def _test_connection(self):
        """Teste la connexion à l'API."""
        try:
            await self.client.ping()
            server_time = await self.client.get_server_time()
            logger.info(f"Connected to Binance. Server time: {server_time['serverTime']}")
        except Exception as e:
            raise BinanceAPIError(
                "Connection test failed",
                original_exception=e
            )
            
    @with_retry()
    async def get_exchange_info(self, force_refresh: bool = False) -> Dict:
        """
        Récupère les informations sur les paires de trading.
        Utilise un cache pour éviter les requêtes inutiles.
        """
        now = time.time()
        
        # Vérifier le cache
        if (not force_refresh and 
            self._exchange_info_cache is not None and 
            self._exchange_info_cache_time is not None and
            now - self._exchange_info_cache_time < self._exchange_info_cache_ttl):
            return self._exchange_info_cache
            
        # Acquérir le droit de faire la requête
        await self.rate_limiter.acquire(self.ENDPOINT_WEIGHTS['exchangeInfo'])
        
        # Faire la requête
        exchange_info = await self.client.get_exchange_info()
        
        # Mettre à jour le cache
        self._exchange_info_cache = exchange_info
        self._exchange_info_cache_time = now
        
        return exchange_info
        
    def _validate_kline_data(self, kline: List, symbol: str) -> Dict[str, Any]:
        """
        Valide et formate une kline (DAT-005).
        
        Args:
            kline: Données brutes de la kline depuis l'API
            symbol: Symbole de trading
            
        Returns:
            Dict avec les données OHLCV validées
            
        Raises:
            KlineValidationError: Si les données sont invalides
        """
        try:
            # Format attendu de Binance:
            # [
            #   [0] open_time,
            #   [1] open,
            #   [2] high,
            #   [3] low,
            #   [4] close,
            #   [5] volume,
            #   [6] close_time,
            #   [7] quote_asset_volume,
            #   [8] number_of_trades,
            #   [9] taker_buy_base_asset_volume,
            #   [10] taker_buy_quote_asset_volume,
            #   [11] ignore
            # ]
            
            if len(kline) < 11:
                raise ValueError(f"Kline data incomplete: expected 11+ fields, got {len(kline)}")
                
            # Convertir et valider les valeurs numériques
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            
            # Validations OHLCV
            if high_price < low_price:
                raise ValueError(f"High price ({high_price}) < Low price ({low_price})")
                
            if high_price < max(open_price, close_price):
                raise ValueError(f"High price ({high_price}) < max(open, close)")
                
            if low_price > min(open_price, close_price):
                raise ValueError(f"Low price ({low_price}) > min(open, close)")
                
            if volume < 0:
                raise ValueError(f"Negative volume: {volume}")
                
            # Valider les timestamps
            open_time = int(kline[0])
            close_time = int(kline[6])
            
            if close_time <= open_time:
                raise ValueError(f"Close time ({close_time}) <= Open time ({open_time})")
                
            return {
                'open_time': open_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'close_time': close_time,
                'quote_asset_volume': float(kline[7]),
                'number_of_trades': int(kline[8]),
                'taker_buy_base_asset_volume': float(kline[9]),
                'taker_buy_quote_asset_volume': float(kline[10])
            }
            
        except (ValueError, IndexError, TypeError) as e:
            raise KlineValidationError(
                f"Invalid kline data for {symbol}",
                symbol=symbol,
                timestamp=kline[0] if kline else None,
                original_exception=e
            )
            
    @with_retry()
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Récupère les klines pour un symbole donné.
        
        Args:
            symbol: Paire de trading (ex: "BTCUSDC")
            interval: Intervalle des klines (défaut: 1m)
            start_time: Timestamp de début en millisecondes
            end_time: Timestamp de fin en millisecondes
            limit: Nombre maximum de klines (max: 1000)
            
        Returns:
            Liste de klines validées
        """
        # Vérifier que le symbole est valide
        exchange_info = await self.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols']}
        
        if symbol not in valid_symbols:
            raise BinanceInvalidSymbolError(
                f"Symbol '{symbol}' not found in exchange",
                symbol=symbol
            )
            
        # Limiter à 1000 klines maximum (limite API)
        limit = min(limit, System.MAX_KLINES_PER_BINANCE_REQUEST)
        
        # Préparer les paramètres
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
            
        # Acquérir le droit de faire la requête
        await self.rate_limiter.acquire(self.ENDPOINT_WEIGHTS['klines'])
        
        logger.debug(f"Fetching klines for {symbol} with params: {params}")
        
        # Faire la requête
        raw_klines = await self.client.get_klines(**params)
        
        # Valider et formater chaque kline
        validated_klines = []
        for kline in raw_klines:
            try:
                validated_kline = self._validate_kline_data(kline, symbol)
                validated_klines.append(validated_kline)
            except KlineValidationError as e:
                logger.warning(f"Skipping invalid kline: {e}")
                # Décider si on veut continuer ou échouer complètement
                # Pour l'instant, on continue avec les klines valides
                
        logger.info(
            f"Fetched {len(validated_klines)} valid klines for {symbol} "
            f"(out of {len(raw_klines)} total)"
        )
        
        return validated_klines
        
    async def fetch_klines_batch(
        self,
        symbol: str,
        interval: str = Kline.INTERVAL_1MINUTE,
        start_time: int = None,
        end_time: int = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère un grand nombre de klines en plusieurs requêtes.
        Gère automatiquement la pagination.
        
        Args:
            symbol: Paire de trading
            interval: Intervalle des klines
            start_time: Timestamp de début en millisecondes
            end_time: Timestamp de fin en millisecondes (défaut: maintenant)
            progress_callback: Fonction appelée avec (klines_downloaded, total_estimated)
            
        Returns:
            Liste complète des klines
        """
        if end_time is None:
            end_time = int(time.time() * 1000)
            
        if start_time is None:
            # Par défaut, récupérer les 30 derniers jours
            start_time = end_time - (30 * 24 * 60 * 60 * 1000)
            
        all_klines = []
        current_start = start_time
        
        # Calculer le nombre approximatif de klines à télécharger
        interval_ms = self._get_interval_milliseconds(interval)
        total_estimated = (end_time - start_time) // interval_ms
        
        logger.info(
            f"Starting batch download for {symbol} from "
            f"{datetime.fromtimestamp(start_time/1000)} to "
            f"{datetime.fromtimestamp(end_time/1000)} "
            f"(~{total_estimated} klines)"
        )
        
        while current_start < end_time:
            try:
                # Récupérer un batch
                batch = await self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_time,
                    limit=1000
                )
                
                if not batch:
                    logger.warning(f"Empty batch received for {symbol} at {current_start}")
                    break
                    
                all_klines.extend(batch)
                
                # Callback de progression
                if progress_callback:
                    progress_callback(len(all_klines), total_estimated)
                    
                # Préparer pour le prochain batch
                # Utiliser le close_time de la dernière kline + 1ms
                last_close_time = batch[-1]['close_time']
                
                # Vérifier qu'on progresse
                if last_close_time <= current_start:
                    logger.error(f"No progress in batch download: last_close_time={last_close_time}, current_start={current_start}")
                    break
                    
                current_start = last_close_time + 1
                
                # Si on a récupéré moins de klines que demandé, on a atteint la fin
                if len(batch) < 1000:
                    logger.info(f"Reached end of available data (batch size: {len(batch)})")
                    break
                    
                # Petite pause pour ne pas surcharger l'API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch download at timestamp {current_start}: {e}")
                raise DataDownloadError(
                    f"Failed to download klines batch for {symbol}",
                    source="Binance",
                    symbol=symbol,
                    original_exception=e
                )
                
        logger.success(
            f"Batch download completed: {len(all_klines)} klines for {symbol}"
        )
        
        return all_klines
        
    def _get_interval_milliseconds(self, interval: str) -> int:
        """Convertit un intervalle en millisecondes."""
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000  # Approximation
        }
        
        return interval_map.get(interval, 60 * 1000)  # Défaut: 1 minute
        
    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Récupère les informations détaillées pour un symbole.
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Informations du symbole (précisions, limites, etc.)
        """
        exchange_info = await self.get_exchange_info()
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
                
        raise BinanceInvalidSymbolError(
            f"Symbol '{symbol}' not found",
            symbol=symbol
        )
        
    async def validate_trading_pair(self, symbol: str) -> bool:
        """
        Vérifie qu'une paire de trading est valide et active.
        
        Args:
            symbol: Paire à vérifier
            
        Returns:
            True si la paire est valide et active
        """
        try:
            symbol_info = await self.get_symbol_info(symbol)
            
            # Vérifier que le symbole est actif
            if symbol_info.get('status') != 'TRADING':
                logger.warning(f"Symbol {symbol} is not in TRADING status: {symbol_info.get('status')}")
                return False
                
            # Vérifier que c'est bien une paire USDC si configuré ainsi
            if settings.trading.base_currency == "USDC":
                if not symbol.endswith("USDC"):
                    logger.warning(f"Symbol {symbol} does not end with USDC")
                    return False
                    
            return True
            
        except BinanceInvalidSymbolError:
            return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False


# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du BinanceDataClient."""
    
    # Utilisation avec context manager
    async with BinanceDataClient() as client:
        # Vérifier une paire
        is_valid = await client.validate_trading_pair("BTCUSDC")
        print(f"BTCUSDC is valid: {is_valid}")
        
        # Récupérer quelques klines récentes
        klines = await client.fetch_klines(
            symbol="BTCUSDC",
            interval="1m",
            limit=10
        )
        
        print(f"Fetched {len(klines)} klines")
        if klines:
            print(f"Latest kline: {klines[-1]}")
            
        # Récupérer un historique plus large
        end_time = int(time.time() * 1000)
        start_time = end_time - (7 * 24 * 60 * 60 * 1000)  # 7 jours
        
        def progress_callback(downloaded, total):
            percentage = (downloaded / total * 100) if total > 0 else 0
            print(f"Progress: {downloaded}/{total} ({percentage:.1f}%)")
            
        historical_klines = await client.fetch_klines_batch(
            symbol="BTCUSDC",
            interval="1m",
            start_time=start_time,
            end_time=end_time,
            progress_callback=progress_callback
        )
        
        print(f"Downloaded {len(historical_klines)} historical klines")
        
        # Afficher les limites de taux actuelles
        current_weight = client.rate_limiter.get_current_weight()
        print(f"Current rate limit usage: {current_weight}/1200")


if __name__ == "__main__":
    # Pour tester le client
    asyncio.run(example_usage())