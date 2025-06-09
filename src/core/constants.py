# src/core/constants.py
from enum import Enum
from typing import List

class Kline:
    # Noms de colonnes standardisés pour les DataFrames de klines
    OHLCV_TIMESTAMP = "timestamp"  # Nom de la colonne pour l'index DatetimeIndex (kline_open_time)
    OHLCV_OPEN = "open"
    OHLCV_HIGH = "high"
    OHLCV_LOW = "low"
    OHLCV_CLOSE = "close"
    OHLCV_VOLUME = "volume" # Correspond à base_asset_volume de Binance
    
    # Colonnes additionnelles souvent présentes dans les données Binance
    OHLCV_KLINE_CLOSE_TIME = "kline_close_time"
    OHLCV_QUOTE_ASSET_VOLUME = "quote_asset_volume"
    OHLCV_NUMBER_OF_TRADES = "number_of_trades"
    OHLCV_TAKER_BUY_BASE_ASSET_VOLUME = "taker_buy_base_asset_volume"
    OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME = "taker_buy_quote_asset_volume"
    OHLCV_IS_KLINE_CLOSED = "is_kline_closed" # Booléen indiquant si la kline est finalisée
    PAIR = "pair" # Colonne pour stocker le symbole de la paire

    # Intervalles de klines standards (peuvent être étendus)
    INTERVAL_1MINUTE = "1m"
    INTERVAL_3MINUTE = "3m"
    INTERVAL_5MINUTE = "5m"
    INTERVAL_15MINUTE = "15m"
    INTERVAL_30MINUTE = "30m"
    INTERVAL_1HOUR = "1h"
    INTERVAL_2HOUR = "2h"
    INTERVAL_4HOUR = "4h"
    INTERVAL_6HOUR = "6h"
    INTERVAL_8HOUR = "8h"
    INTERVAL_12HOUR = "12h"
    INTERVAL_1DAY = "1d"
    INTERVAL_3DAY = "3d"
    INTERVAL_1WEEK = "1w"
    INTERVAL_1MONTH = "1M"

    # Préfixe pour les colonnes de klines agrégées/roulantes
    ROLLING_KLINE_PREFIX = "K"

    # Intervalles cibles pour les klines roulantes
    TARGET_ROLLING_INTERVALS: List[str] = [
        INTERVAL_3MINUTE, INTERVAL_5MINUTE, INTERVAL_15MINUTE, INTERVAL_30MINUTE,
        INTERVAL_1HOUR, INTERVAL_2HOUR, INTERVAL_4HOUR, INTERVAL_6HOUR, INTERVAL_8HOUR
    ]
    
    # Colonnes de base qui seront préfixées pour les klines roulantes et 1m dans le format enrichi
    # Kline.OHLCV_TIMESTAMP (l'index) et Kline.PAIR ne sont généralement pas préfixés de cette manière.
    BASE_COLS_FOR_PREFIXING: List[str] = [
        OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME,
        OHLCV_KLINE_CLOSE_TIME, OHLCV_QUOTE_ASSET_VOLUME, OHLCV_NUMBER_OF_TRADES,
        OHLCV_TAKER_BUY_BASE_ASSET_VOLUME, OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME,
        OHLCV_IS_KLINE_CLOSED
    ]


    # Règles d'agrégation par défaut pour le resampling standard
    # Utilisées par KlineProcessor pour les klines NON ROULANTES
    AGGREGATION_RULES_STANDARD = {
        OHLCV_OPEN: "first",
        OHLCV_HIGH: "max",
        OHLCV_LOW: "min",
        OHLCV_CLOSE: "last",
        OHLCV_VOLUME: "sum",
        OHLCV_QUOTE_ASSET_VOLUME: "sum",
        OHLCV_NUMBER_OF_TRADES: "sum",
        OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: "sum",
        OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: "sum",
        OHLCV_KLINE_CLOSE_TIME: "last", # Pour les klines standard, le close time est celui de la dernière kline source
        OHLCV_IS_KLINE_CLOSED: "all" # Une kline standard est fermée si toutes ses klines sources l'étaient
        # PAIR: "first", # Si la colonne pair est présente et doit être propagée
    }

    @staticmethod
    def get_prefixed_col_name(interval: str, base_col_name: str) -> str:
        """
        Génère un nom de colonne préfixé pour les klines roulantes ou 1m.
        Exemple: get_prefixed_col_name("5m", "open") -> "K_5m_open"
        Exemple: get_prefixed_col_name("1m", "close") -> "K_1m_close"
        """
        if not isinstance(interval, str) or not interval:
            raise ValueError("Interval must be a non-empty string.")
        if not isinstance(base_col_name, str) or not base_col_name:
            raise ValueError("Base column name must be a non-empty string.")
            
        # Nettoyer le base_col_name s'il contient déjà des points (ex: kline.OHLCV_CLOSE_TIME)
        # pour éviter des doubles points ou des noms de colonnes invalides.
        # La convention est d'utiliser les constantes de base comme OHLCV_CLOSE, etc.
        # Si base_col_name est "kline.OHLCV_CLOSE_TIME", on veut "K_5m_kline.OHLCV_CLOSE_TIME"
        # Si base_col_name est "close", on veut "K_5m_close"
        # Le préfixe est toujours K_<interval>_
        
        return f"{Kline.ROLLING_KLINE_PREFIX}_{interval}_{base_col_name}"


class StorageTypes(Enum):
    """Types de stockage supportés pour les données historiques."""
    PARQUET = "parquet"
    POSTGRES = "postgres"
    # SQLITE = "sqlite" # Exemple si vous ajoutez SQLite plus tard


class Trading:
    """Constantes liées au trading."""
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"

    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    ORDER_TYPE_TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

    TIME_IN_FORCE_GTC = "GTC"  # Good Til Canceled
    TIME_IN_FORCE_IOC = "IOC"  # Immediate Or Cancel
    TIME_IN_FORCE_FOK = "FOK"  # Fill Or Kill

    # Types de signaux (exemples)
    SIGNAL_TYPE_LONG = "LONG"
    SIGNAL_TYPE_SHORT = "SHORT"
    SIGNAL_TYPE_NEUTRAL = "NEUTRAL" # ou "HOLD", "EXIT"


class System:
    """Constantes liées à la configuration système et aux valeurs par défaut."""
    # Valeur par défaut pour le nombre de workers dans ThreadPoolExecutor
    # Utilisé par DataManager si non spécifié dans la configuration (settings.system.thread_pool_workers)
    THREAD_POOL_WORKERS_DEFAULT = 4 
    
    # Valeur par défaut pour le nombre de workers pour les opérations de stockage
    # Utilisé par ParquetStorage si non spécifié dans settings.system.thread_pool_workers_storage
    THREAD_POOL_WORKERS_STORAGE_DEFAULT = 4

    # Nombre maximum de tentatives pour les appels API Binance en cas d'erreur récupérable
    MAX_BINANCE_API_RETRIES = 5

    # Nombre maximum de klines par requête à l'API Binance (généralement 1000)
    MAX_KLINES_PER_BINANCE_REQUEST = 1000


class DataFrameCols(Enum):
    """
    Noms de colonnes standardisés utilisés dans les DataFrames à travers l'application,
    notamment pour les signaux et les indicateurs dans les stratégies.
    """
    # Colonnes de base OHLCV (normalisées en minuscules pour la logique interne)
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TIMESTAMP = Kline.OHLCV_TIMESTAMP # "timestamp" ou "kline_open_time"

    # Colonnes pour les signaux de trading
    SIGNAL = "signal"               # Valeur numérique: 1 (long), -1 (short), 0 (neutre/hold)
    ENTRY_LONG = "entry_long"       # Booléen: True si signal d'entrée long
    EXIT_LONG = "exit_long"         # Booléen: True si signal de sortie long
    ENTRY_SHORT = "entry_short"     # Booléen: True si signal d'entrée short
    EXIT_SHORT = "exit_short"       # Booléen: True si signal de sortie short
    
    ENTRY_PRICE = "entry_price"     # Prix d'entrée suggéré/effectif
    STOP_LOSS = "sl"                # Niveau de Stop Loss (prix)
    TAKE_PROFIT = "tp"              # Niveau de Take Profit (prix)
    
    CONFIDENCE = "confidence"       # Score de confiance du signal [0-1] (optionnel)
    SIZE = "size"                   # Taille de la position (peut être en % capital, montant fixe, etc.)

    # Exemple de noms de colonnes pour des indicateurs communs (la stratégie définira les noms exacts)
    # Ces noms peuvent être construits dynamiquement par les stratégies en fonction de leurs paramètres.
    # SMA_FAST = "sma_fast"
    # SMA_SLOW = "sma_slow"
    # RSI = "rsi"
    # BB_UPPER = "bb_upper"
    # BB_MIDDLE = "bb_middle"
    # BB_LOWER = "bb_lower"
    # ATR = "atr"

    # Vous pouvez ajouter d'autres constantes de noms de colonnes ici au besoin.
