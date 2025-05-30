# src/core/constants.py
from typing import Dict, Any

# --- Constantes de Trading ---
class Trading:
    SIDE_BUY: str = "BUY"
    SIDE_SELL: str = "SELL"

    ORDER_TYPE_LIMIT: str = "LIMIT"
    ORDER_TYPE_MARKET: str = "MARKET"
    ORDER_TYPE_STOP_LOSS: str = "STOP_LOSS" # Souvent utilisé comme trigger pour STOP_LOSS_LIMIT
    ORDER_TYPE_STOP_LOSS_LIMIT: str = "STOP_LOSS_LIMIT"
    ORDER_TYPE_TAKE_PROFIT: str = "TAKE_PROFIT" # Souvent utilisé comme trigger pour TAKE_PROFIT_LIMIT
    ORDER_TYPE_TAKE_PROFIT_LIMIT: str = "TAKE_PROFIT_LIMIT"
    ORDER_TYPE_LIMIT_MAKER: str = "LIMIT_MAKER"

    TIME_IN_FORCE_GTC: str = "GTC"  # Good Til Cancelled
    TIME_IN_FORCE_IOC: str = "IOC"  # Immediate Or Cancel
    TIME_IN_FORCE_FOK: str = "FOK"  # Fill Or Kill

    SIGNAL_TYPE_LONG: str = "LONG"
    SIGNAL_TYPE_SHORT: str = "SHORT"
    SIGNAL_TYPE_NEUTRAL: str = "NEUTRAL" # Pourrait être utile pour sortir d'une position sans en prendre une nouvelle

    # Effets secondaires pour les ordres sur marge croisée Binance
    BINANCE_SIDE_EFFECT_TYPE_MARGIN_BUY: str = "MARGIN_BUY"
    BINANCE_SIDE_EFFECT_TYPE_AUTO_REPAY: str = "AUTO_REPAY"
    BINANCE_SIDE_EFFECT_TYPE_NO_SIDE_EFFECT: str = "NO_SIDE_EFFECT"

    # Statuts d'ordre courants de Binance
    ORDER_STATUS_NEW: str = "NEW"
    ORDER_STATUS_PARTIALLY_FILLED: str = "PARTIALLY_FILLED"
    ORDER_STATUS_FILLED: str = "FILLED"
    ORDER_STATUS_CANCELED: str = "CANCELED"
    ORDER_STATUS_PENDING_CANCEL: str = "PENDING_CANCEL" # (currently unused)
    ORDER_STATUS_REJECTED: str = "REJECTED"
    ORDER_STATUS_EXPIRED: str = "EXPIRED"
    ORDER_STATUS_EXPIRED_IN_MATCH: str = "EXPIRED_IN_MATCH" # Pour les ordres de type FOK/IOC non exécutés

# --- Constantes de Données (Kline) ---
class Kline:
    INTERVAL_1MINUTE: str = "1m"
    INTERVAL_3MINUTE: str = "3m"
    INTERVAL_5MINUTE: str = "5m"
    INTERVAL_15MINUTE: str = "15m"
    INTERVAL_30MINUTE: str = "30m"
    INTERVAL_1HOUR: str = "1h"
    INTERVAL_2HOUR: str = "2h"
    INTERVAL_4HOUR: str = "4h"
    INTERVAL_6HOUR: str = "6h"
    INTERVAL_8HOUR: str = "8h"
    INTERVAL_12HOUR: str = "12h"
    INTERVAL_1DAY: str = "1d"
    INTERVAL_3DAY: str = "3d"
    INTERVAL_1WEEK: str = "1w"
    INTERVAL_1MONTH: str = "1M" # Note: Binance utilise 'M' majuscule pour mois

    # Noms des colonnes OHLCV standard
    OHLCV_OPEN: str = "open"
    OHLCV_HIGH: str = "high"
    OHLCV_LOW: str = "low"
    OHLCV_CLOSE: str = "close"
    OHLCV_VOLUME: str = "volume"
    OHLCV_TIMESTAMP: str = "timestamp" # Souvent utilisé comme index ou colonne
    OHLCV_QUOTE_ASSET_VOLUME: str = "quote_asset_volume"
    OHLCV_NUMBER_OF_TRADES: str = "number_of_trades"
    OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: str = "taker_buy_base_asset_volume"
    OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: str = "taker_buy_quote_asset_volume"

    # Règles d'agrégation standard pour le resampling des klines
    AGGREGATION_RULES_STANDARD: Dict[str, str] = {
        OHLCV_OPEN: "first",
        OHLCV_HIGH: "max",
        OHLCV_LOW: "min",
        OHLCV_CLOSE: "last",
        OHLCV_VOLUME: "sum",
        OHLCV_QUOTE_ASSET_VOLUME: "sum", # Si disponible
        OHLCV_NUMBER_OF_TRADES: "sum"    # Si disponible
    }

# --- Constantes Numériques et Précision ---
class Precision:
    DEFAULT_FLOAT_PRECISION: int = 8  # Pour les prix et quantités crypto
    SATOSHI_PRECISION: int = 8        # Pour BTC et autres cryptos similaires
    SMALL_FLOAT_VALUE: float = 1e-9   # Pour éviter les divisions par zéro ou pour comparaisons de flottants
    PERCENTAGE_MULTIPLIER: float = 100.0

# --- Constantes de Configuration/Système ---
class System:
    MAX_BINANCE_API_RETRIES: int = 5
    DEFAULT_API_TIMEOUT_SECONDS: int = 10 # Timeout pour les requêtes API
    MAX_KLINES_PER_BINANCE_REQUEST: int = 1000 # Limite de l'API Binance pour les klines
    WEBSOCKET_MAX_RECONNECT_ATTEMPTS: int = 10
    WEBSOCKET_RECONNECT_DELAY_SECONDS: int = 5

# --- Constantes de Noms de Colonnes DataFrame (Exemples) ---
# Utilisées pour la cohérence dans les DataFrames manipulés par les stratégies ou backtests
class DataFrameCols:
    SIGNAL: str = "signal"                # Colonne indiquant le signal (1 pour long, -1 pour short, 0 pour neutre)
    ENTRY_PRICE: str = "entry_price"      # Prix d'entrée d'une position
    EXIT_PRICE: str = "exit_price"        # Prix de sortie d'une position
    STOP_LOSS_PRICE: str = "stop_loss"    # Niveau de Stop Loss
    TAKE_PROFIT_PRICE: str = "take_profit" # Niveau de Take Profit
    POSITION_SIZE: str = "position_size"  # Taille de la position
    PNL: str = "pnl"                      # Profit and Loss
    EQUITY: str = "equity"                # Courbe de capital

# --- Constantes pour les types de stockage ---
class StorageTypes:
    PARQUET: str = "parquet"
    POSTGRES: str = "postgres"

# --- Constantes pour les environnements ---
class Environments:
    DEVELOPMENT: str = "development"
    PRODUCTION: str = "production"
    TEST: str = "test"

# --- Constantes pour les modes de trading ---
class TradingModes:
    CROSS_MARGIN: str = "cross_margin"
    # ISOLATED_MARGIN: str = "ISOLATED_MARGIN" # Si supporté ultérieurement
    # SPOT: str = "SPOT" # Si supporté ultérieurement

# --- Constantes pour les WebSockets Binance ---
class BinanceWebsockets:
    BASE_STREAM_URL_SPOT: str = "wss://stream.binance.com:9443/ws"
    BASE_STREAM_URL_FUTURES: str = "wss://fstream.binance.com/ws"
    BASE_STREAM_URL_TESTNET_SPOT: str = "wss://testnet.binance.vision/ws"
    BASE_STREAM_URL_TESTNET_FUTURES: str = "wss://stream.binancefuture.com/ws" # Vérifier l'URL exacte du testnet futures

    KLINE_STREAM_SUFFIX: str = "@kline_" # ex: btcusdt@kline_1m
    DEPTH_STREAM_SUFFIX: str = "@depth" # ex: btcusdt@depth ou btcusdt@depth5@100ms
    USER_DATA_STREAM: str = "userDataStream" # Pour les mises à jour de compte/ordres


# Il est possible d'importer les classes directement:
# from src.core.constants import Trading, Kline
# print(Trading.SIDE_BUY)
# print(Kline.INTERVAL_5MINUTE)

# Ou d'importer des constantes spécifiques si les classes ne sont pas utilisées comme namespaces:
# # Définir les constantes directement sans classes si préféré
# SIDE_BUY = "BUY"
# KLINE_INTERVAL_1MINUTE = "1m"
# from src.core.constants import SIDE_BUY
