# src/data/storage/cache.py
import pickle
import hashlib
import asyncio
from typing import Any, Optional, Union, List, Dict # Added Dict for get_info return type
from datetime import datetime, timedelta
import pandas as pd
import redis.asyncio as redis
from loguru import logger

from src.core.config import settings
# from src.core.exceptions import CacheError # CacheError n'est pas utilisé dans ce snippet

class CacheManager:
    """
    Gestionnaire de cache utilisant Redis pour stocker temporairement les données.
    Améliore les performances en évitant les requêtes répétitives.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de cache."""
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = settings.data.cache_ttl
        self._initialized = False
        
        # Configuration Redis
        self.redis_config = {
            'host': settings.redis.host if hasattr(settings, 'redis') else 'localhost',
            'port': settings.redis.port if hasattr(settings, 'redis') else 6379,
            'decode_responses': False,  # Pour gérer les données binaires
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        # Ajouter le mot de passe si configuré
        if hasattr(settings, 'redis') and settings.redis.password:
            self.redis_config['password'] = settings.redis.password.get_secret_value()
            
        logger.info("CacheManager initialized")
        
    async def initialize(self) -> None:
        """Initialise la connexion Redis."""
        try:
            # Créer le client Redis asynchrone
            self.redis_client = redis.Redis(**self.redis_config)
            
            # Tester la connexion
            await self.redis_client.ping()
            
            self._initialized = True
            logger.success("CacheManager connected to Redis")
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
            self._initialized = False
            
        except Exception as e:
            # Correction pour l'erreur à la ligne 63 originale:
            # La ligne `logger.error(f"Error getting cache info: {e}")` a été supprimée car elle semblait déplacée.
            # La ligne `return {"available": False, "error": str(e)} as e:` contenait une erreur de syntaxe (`as e:`)
            # et le `return` n'est pas approprié pour la méthode `initialize` qui retourne `None`.
            # Le message d'erreur ci-dessous est plus générique pour une initialisation échouée.
            logger.error(f"Unexpected error initializing cache: {e}")
            self.redis_client = None
            self._initialized = False
            
    async def close(self) -> None:
        """Ferme la connexion Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("CacheManager connection closed")
            
    def is_available(self) -> bool:
        """Vérifie si le cache est disponible."""
        return self._initialized and self.redis_client is not None
        
    def generate_key(self, *args) -> str:
        """
        Génère une clé de cache unique basée sur les arguments.
        
        Args:
            *args: Arguments utilisés pour générer la clé
            
        Returns:
            Clé de cache unique
        """
        # Créer une représentation string des arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, (datetime, pd.Timestamp)):
                key_parts.append(arg.isoformat())
            elif isinstance(arg, pd.DataFrame):
                # Pour les DataFrames, utiliser une représentation simplifiée
                key_parts.append(f"df_{len(arg)}_{arg.index.min()}_{arg.index.max()}")
            else:
                key_parts.append(str(arg))
                
        key_string = ":".join(key_parts)
        
        # Limiter la longueur de la clé
        if len(key_string) > 200:
            # Utiliser un hash pour les clés très longues
            hash_value = hashlib.md5(key_string.encode()).hexdigest()
            key_string = f"{key_parts[0]}:{hash_value}"
            
        return key_string
        
    async def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur stockée ou None si non trouvée/expirée
        """
        if not self.is_available():
            return None
            
        try:
            # Récupérer la valeur
            value_bytes = await self.redis_client.get(key)
            
            if value_bytes is None:
                return None
                
            # Désérialiser
            value = pickle.loads(value_bytes)
            
            # Si c'est un DataFrame, recréer avec les types corrects
            if isinstance(value, dict) and '_dataframe_' in value:
                df_data = value['_dataframe_']
                df = pd.DataFrame(df_data['data'])
                
                # Restaurer l'index
                if df_data.get('index_name'):
                    df.set_index(df_data['index_name'], inplace=True)
                    
                # Restaurer les types datetime
                for col in df_data.get('datetime_columns', []):
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                    elif col == df.index.name: # Check if the index itself is a datetime column
                        df.index = pd.to_datetime(df.index)
                        
                return df
                
            return value
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error getting cache key {key}: {e}")
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Time To Live en secondes (None = utiliser le TTL par défaut)
            
        Returns:
            True si stocké avec succès, False sinon
        """
        if not self.is_available():
            return False
            
        try:
            # Utiliser le TTL par défaut si non spécifié
            if ttl is None:
                ttl = self.default_ttl
                
            # Préparer la valeur pour la sérialisation
            if isinstance(value, pd.DataFrame):
                # Convertir le DataFrame en format sérialisable
                df_data = {
                    'data': value.reset_index().to_dict('records'), # reset_index to handle index serialization
                    'index_name': value.index.name,
                    'datetime_columns': []
                }
                
                # Identifier les colonnes datetime
                for col in value.columns:
                    if pd.api.types.is_datetime64_any_dtype(value[col]):
                        df_data['datetime_columns'].append(col)
                        
                if pd.api.types.is_datetime64_any_dtype(value.index):
                    # Ensure index name is a string, use 'index' if None
                    df_data['datetime_columns'].append(value.index.name if value.index.name is not None else 'index') 
                    
                value_to_store = {'_dataframe_': df_data}
            else:
                value_to_store = value
                
            # Sérialiser
            value_bytes = pickle.dumps(value_to_store)
            
            # Stocker avec TTL
            await self.redis_client.setex(key, ttl, value_bytes)
            
            logger.debug(f"Cached key {key} with TTL {ttl}s")
            return True
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error setting cache key {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """
        Supprime une clé du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si supprimé, False sinon
        """
        if not self.is_available():
            return False
            
        try:
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """
        Vérifie si une clé existe dans le cache.
        
        Args:
            key: Clé à vérifier
            
        Returns:
            True si la clé existe, False sinon
        """
        if not self.is_available():
            return False
            
        try:
            return await self.redis_client.exists(key) > 0
            
        except Exception: # Consider logging this exception too
            return False
            
    async def clear(self) -> bool:
        """
        Vide complètement le cache.
        
        Returns:
            True si vidé avec succès, False sinon
        """
        if not self.is_available():
            return False
            
        try:
            await self.redis_client.flushdb()
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
            
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalide toutes les clés correspondant à un pattern.
        
        Args:
            pattern: Pattern de clé (ex: "klines:BTCUSDC:*")
            
        Returns:
            Nombre de clés supprimées
        """
        if not self.is_available():
            return 0
            
        try:
            # Rechercher les clés correspondantes
            keys = []
            async for key_bytes in self.redis_client.scan_iter(match=pattern): # redis-py returns bytes for keys
                keys.append(key_bytes) 
                
            # Supprimer par batch
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.debug(f"Invalidated {deleted} keys matching pattern {pattern}")
                return deleted
                
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0
            
    async def get_info(self) -> Dict[str, Any]:
        """
        Récupère des informations sur l'état du cache.
        
        Returns:
            Dictionnaire avec les informations du cache
        """
        if not self.is_available():
            return {"available": False}
            
        try:
            info = await self.redis_client.info()
            
            keyspace_hits = info.get("keyspace_hits", 0)
            keyspace_misses = info.get("keyspace_misses", 0)
            total_lookups = keyspace_hits + keyspace_misses
            if total_lookups == 0: # Avoid division by zero if no lookups yet
                 total_lookups = 1 # Set to 1 to avoid division by zero, hit rate will be 0

            return {
                "available": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "used_memory_peak": info.get("used_memory_peak_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": keyspace_hits,
                "keyspace_misses": keyspace_misses,
                "hit_rate": (keyspace_hits / total_lookups) * 100 if keyspace_hits > 0 else 0.0

            }
            
        # Correction pour l'erreur à la ligne 341 originale:
        # `except Exception` a été complété par `: as e:` et un corps pour la gestion d'erreur.
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"available": False, "error": str(e)}