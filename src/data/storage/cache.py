# src/data/storage/cache.py
import asyncio
import pickle # Using pickle for simplicity, consider msgpack for performance/security with DataFrames
from typing import Any, Optional, Union
from datetime import timedelta
import redis.asyncio as aioredis # type: ignore
from loguru import logger

from src.core.config import settings
from src.core.exceptions import CacheError, ConfigurationError

class CacheManager:
    """
    Manages caching operations, primarily using Redis.
    Handles serialization/deserialization of cached objects.
    """

    def __init__(self):
        """Initializes the CacheManager."""
        self.redis_url: Optional[str] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self._initialized: bool = False

        if settings.data.cache_enabled:
            if not settings.redis or not settings.redis.host:
                raise ConfigurationError("Redis is enabled but connection details (host) are not configured.")

            # Construct Redis URL if not fully provided in settings
            # Example: redis://[[username]:[password]]@[hostname]:[port]/[db_number]
            # For simplicity, assuming host and port are main concerns from settings.
            # Password handling should be secure.
            password_part = ""
            if settings.redis.password:
                password_value = settings.redis.password.get_secret_value()
                if password_value: # Ensure password is not None or empty
                    password_part = f":{password_value}@"

            self.redis_url = f"redis://{password_part}{settings.redis.host}:{settings.redis.port}/0"
            logger.info(f"CacheManager initialized for Redis at {settings.redis.host}:{settings.redis.port}")
        else:
            logger.info("CacheManager initialized with caching disabled.")


    async def initialize(self):
        """Initializes the Redis client connection if caching is enabled."""
        if not settings.data.cache_enabled:
            logger.debug("Caching is disabled, skipping Redis client initialization.")
            self._initialized = True # Mark as initialized even if disabled
            return

        if self._initialized and self.redis_client:
            logger.debug("Redis client already initialized.")
            return

        if not self.redis_url:
             raise ConfigurationError("Redis URL not set, cannot initialize client.")

        try:
            # pool = aioredis.ConnectionPool.from_url(self.redis_url, max_connections=10)
            # self.redis_client = aioredis.Redis(connection_pool=pool)
            # Using from_url directly creates a client with a connection pool
            self.redis_client = aioredis.Redis.from_url(self.redis_url)
            await self.redis_client.ping() # Verify connection
            self._initialized = True
            logger.success("Redis client initialized and connection verified.")
        except Exception as e:
            self.redis_client = None # Ensure client is None if init fails
            logger.error(f"Failed to initialize Redis client: {e}")
            # Optionally, allow the app to run without cache, or raise CacheError
            # For now, we'll log and proceed, cache operations will then fail gracefully.
            # raise CacheError(f"Redis client initialization failed: {e}", original_exception=e)


    async def close(self):
        """Closes the Redis client connection."""
        if self.redis_client:
            try:
                await self.redis_client.close()
                # await self.redis_client.connection_pool.disconnect() # If using explicit pool
                logger.info("Redis client connection closed.")
            except Exception as e:
                logger.error(f"Error closing Redis client connection: {e}")
            finally:
                self.redis_client = None
                self._initialized = False
        else:
            logger.debug("Redis client already closed or not initialized.")


    def generate_key(self, *args: Any) -> str:
        """
        Generates a consistent cache key from the given arguments.
        Simple string concatenation for this example.
        """
        return ":".join(map(str, args))

    async def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache."""
        if not self.redis_client or not self._initialized:
            logger.warning("Cache not available (not initialized or disabled). Cannot get item.")
            return None
        try:
            cached_value = await self.redis_client.get(key)
            if cached_value:
                logger.debug(f"Cache hit for key: {key}")
                # Deserialize the value (e.g., using pickle)
                return pickle.loads(cached_value)
            logger.debug(f"Cache miss for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting item from cache (key: {key}): {e}")
            # Optionally, raise CacheError or return None
            # raise CacheError(f"Failed to get item for key '{key}': {e}", original_exception=e)
            return None # Fail gracefully

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Stores an item in the cache with an optional Time-To-Live (TTL).
        """
        if not self.redis_client or not self._initialized:
            logger.warning("Cache not available (not initialized or disabled). Cannot set item.")
            return

        if ttl_seconds is None:
            ttl_seconds = settings.data.cache_ttl # Default TTL from settings

        try:
            # Serialize the value (e.g., using pickle)
            serialized_value = pickle.dumps(value)
            await self.redis_client.set(key, serialized_value, ex=ttl_seconds)
            logger.debug(f"Item stored in cache (key: {key}, ttl: {ttl_seconds}s)")
        except Exception as e:
            logger.error(f"Error setting item in cache (key: {key}): {e}")
            # Optionally, raise CacheError
            # raise CacheError(f"Failed to set item for key '{key}': {e}", original_exception=e)


    async def delete(self, key: str) -> bool:
        """Deletes an item from the cache. Returns True if deleted, False otherwise."""
        if not self.redis_client or not self._initialized:
            logger.warning("Cache not available (not initialized or disabled). Cannot delete item.")
            return False
        try:
            result = await self.redis_client.delete(key)
            if result > 0:
                logger.debug(f"Item deleted from cache (key: {key})")
                return True
            logger.debug(f"Item not found in cache for deletion (key: {key})")
            return False
        except Exception as e:
            logger.error(f"Error deleting item from cache (key: {key}): {e}")
            # raise CacheError(f"Failed to delete item for key '{key}': {e}", original_exception=e)
            return False

    async def exists(self, key: str) -> bool:
        """Checks if a key exists in the cache."""
        if not self.redis_client or not self._initialized:
            logger.warning("Cache not available (not initialized or disabled). Cannot check if item exists.")
            return False
        try:
            return await self.redis_client.exists(key) > 0 # type: ignore
        except Exception as e:
            logger.error(f"Error checking if key exists in cache (key: {key}): {e}")
            # raise CacheError(f"Failed to check existence for key '{key}': {e}", original_exception=e)
            return False

    async def clear(self) -> bool:
        """Clears the entire cache (flushes the current DB). Use with caution."""
        if not self.redis_client or not self._initialized:
            logger.warning("Cache not available (not initialized or disabled). Cannot clear cache.")
            return False
        try:
            await self.redis_client.flushdb()
            logger.info("Cache cleared (current Redis DB flushed).")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            # raise CacheError(f"Failed to clear cache: {e}", original_exception=e)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidates (deletes) all keys matching a given pattern.
        Warning: SCAN can be slow on large Redis instances if not used carefully.
        Ensure patterns are specific enough.

        Args:
            pattern: The pattern to match keys against (e.g., "klines:BTCUSDC:*")

        Returns:
            The number of keys deleted.
        """
        if not self.redis_client or not self._initialized:
            logger.warning("Cache not available (not initialized or disabled). Cannot invalidate pattern.")
            return 0

        deleted_count = 0
        try:
            async for key in self.redis_client.scan_iter(match=pattern, count=100): # count for batching
                await self.redis_client.delete(key)
                deleted_count += 1
            if deleted_count > 0:
                logger.info(f"Invalidated {deleted_count} keys matching pattern: {pattern}")
            else:
                logger.debug(f"No keys found matching pattern for invalidation: {pattern}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error invalidating cache pattern '{pattern}': {e}")
            # raise CacheError(f"Failed to invalidate pattern '{pattern}': {e}", original_exception=e)
            return 0

# Example usage (typically not directly in this file but in DataManager or services)
async def example_cache_usage():
    # This requires Redis to be running and configured in .env or settings.
    # For this example, we assume settings are loaded.
    if not settings.data.cache_enabled:
        print("Caching is disabled in settings. Skipping example.")
        return

    cache_manager = CacheManager()
    await cache_manager.initialize()

    if not cache_manager.redis_client:
        print("Failed to initialize cache manager. Skipping example.")
        return

    my_key = cache_manager.generate_key("test_data", "user123")
    my_data = {"name": "Test User", "value": 42, "items": [1, 2, 3]}

    # Set data
    await cache_manager.set(my_key, my_data, ttl_seconds=60)
    print(f"Set data for key: {my_key}")

    # Get data
    retrieved_data = await cache_manager.get(my_key)
    if retrieved_data:
        print(f"Retrieved data: {retrieved_data}")
        assert retrieved_data == my_data
    else:
        print("Data not found in cache (or expired).")

    # Check existence
    exists = await cache_manager.exists(my_key)
    print(f"Key '{my_key}' exists: {exists}")
    assert exists

    # Invalidate pattern (example)
    await cache_manager.set("klines:BTCUSDC:1m", "some_kline_data_btc", 60)
    await cache_manager.set("klines:ETHUSDC:1m", "some_kline_data_eth", 60)
    deleted_num = await cache_manager.invalidate_pattern("klines:BTCUSDC:*")
    print(f"Deleted {deleted_num} keys matching 'klines:BTCUSDC:*'")
    assert await cache_manager.exists("klines:BTCUSDC:1m") is False
    assert await cache_manager.exists("klines:ETHUSDC:1m") is True


    # Delete data
    deleted = await cache_manager.delete(my_key)
    print(f"Key '{my_key}' deleted: {deleted}")
    assert deleted
    assert await cache_manager.get(my_key) is None


    await cache_manager.close()

if __name__ == "__main__":
    # To run this example:
    # 1. Ensure you have a Redis server running.
    # 2. Configure Redis in your settings (e.g., through .env file for src.core.config.settings)
    #    settings.data.cache_enabled = True
    #    settings.redis.host = "localhost"
    #    settings.redis.port = 6379
    try:
        asyncio.run(example_cache_usage())
    except ConfigurationError as e:
        print(f"Configuration Error: {e}. Please ensure Redis is configured and cache is enabled in settings.")
    except CacheError as e:
        print(f"Cache Error: {e}. Is Redis server running and accessible?")
    except ConnectionRefusedError:
        print("Connection refused. Is Redis server running on the configured host/port?")