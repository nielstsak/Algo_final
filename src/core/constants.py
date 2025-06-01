# src/core/constants.py
from typing import Dict, Any

# --- Constantes de Trading ---
class Trading:
    """Constantes relatives aux opérations de trading."""
    SIDE_BUY: str = "BUY"  # Action d'achat
    SIDE_SELL: str = "SELL" # Action de vente

    ORDER_TYPE_LIMIT: str = "LIMIT"  # Ordre à cours limité
    ORDER_TYPE_MARKET: str = "MARKET" # Ordre au marché
    ORDER_TYPE_STOP_LOSS: str = "STOP_LOSS" # Type d'ordre pour déclencher un Stop Loss (souvent utilisé avec STOP_LOSS_LIMIT)
    ORDER_TYPE_STOP_LOSS_LIMIT: str = "STOP_LOSS_LIMIT" # Ordre Stop Loss à cours limité
    ORDER_TYPE_TAKE_PROFIT: str = "TAKE_PROFIT" # Type d'ordre pour déclencher un Take Profit (souvent utilisé avec TAKE_PROFIT_LIMIT)
    ORDER_TYPE_TAKE_PROFIT_LIMIT: str = "TAKE_PROFIT_LIMIT" # Ordre Take Profit à cours limité
    ORDER_TYPE_LIMIT_MAKER: str = "LIMIT_MAKER" # Ordre Limit qui ne s'exécute que s'il est Post-Only (Maker)

    TIME_IN_FORCE_GTC: str = "GTC"  # Good Til Cancelled: L'ordre reste actif jusqu'à ce qu'il soit exécuté ou annulé.
    TIME_IN_FORCE_IOC: str = "IOC"  # Immediate Or Cancel: L'ordre doit être exécuté immédiatement en totalité ou en partie, ce qui n'est pas exécuté est annulé.
    TIME_IN_FORCE_FOK: str = "FOK"  # Fill Or Kill: L'ordre doit être exécuté immédiatement et en totalité, sinon il est annulé.

    SIGNAL_TYPE_LONG: str = "LONG"    # Signal pour une position d'achat (longue)
    SIGNAL_TYPE_SHORT: str = "SHORT"  # Signal pour une position de vente (courte)
    SIGNAL_TYPE_NEUTRAL: str = "NEUTRAL" # Signal pour indiquer une absence de position ou la clôture d'une position existante sans en ouvrir une nouvelle.

    # Effets secondaires pour les ordres sur marge croisée Binance (SideEffectType)
    BINANCE_SIDE_EFFECT_TYPE_MARGIN_BUY: str = "MARGIN_BUY" # Pour les ordres d'achat sur marge qui peuvent emprunter automatiquement.
    BINANCE_SIDE_EFFECT_TYPE_AUTO_REPAY: str = "AUTO_REPAY" # Pour les ordres de vente sur marge qui remboursent automatiquement les dettes.
    BINANCE_SIDE_EFFECT_TYPE_NO_SIDE_EFFECT: str = "NO_SIDE_EFFECT" # L'ordre n'a pas d'effet secondaire sur la marge (comportement par défaut pour spot).

    # Statuts d'ordre courants de Binance
    ORDER_STATUS_NEW: str = "NEW" # L'ordre a été accepté par le moteur mais pas encore exécuté.
    ORDER_STATUS_PARTIALLY_FILLED: str = "PARTIALLY_FILLED" # L'ordre a été partiellement exécuté.
    ORDER_STATUS_FILLED: str = "FILLED" # L'ordre a été complètement exécuté.
    ORDER_STATUS_CANCELED: str = "CANCELED" # L'ordre a été annulé par l'utilisateur.
    ORDER_STATUS_PENDING_CANCEL: str = "PENDING_CANCEL" # Annulation en cours (actuellement non utilisé par Binance).
    ORDER_STATUS_REJECTED: str = "REJECTED" # L'ordre a été rejeté par le moteur.
    ORDER_STATUS_EXPIRED: str = "EXPIRED" # L'ordre a expiré (ex: ordres avec timeInForce spécifique).
    ORDER_STATUS_EXPIRED_IN_MATCH: str = "EXPIRED_IN_MATCH" # Pour les ordres FOK/IOC qui n'ont pas pu être exécutés et ont expiré.

# --- Constantes de Données (Kline) ---
class Kline:
    """Constantes relatives aux données klines (bougies)."""
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
    INTERVAL_1MONTH: str = "1M" # Note: Binance utilise 'M' majuscule pour le mois.

    # Noms des colonnes OHLCV standardisés pour l'application
    OHLCV_TIMESTAMP: str = "timestamp" # Colonne pour le timestamp d'ouverture de la kline (souvent l'index du DataFrame).
    OHLCV_OPEN: str = "open"        # Prix d'ouverture
    OHLCV_HIGH: str = "high"        # Prix le plus haut
    OHLCV_LOW: str = "low"         # Prix le plus bas
    OHLCV_CLOSE: str = "close"       # Prix de clôture
    OHLCV_VOLUME: str = "volume"      # Volume de l'actif de base

    # Noms de colonnes additionnelles provenant de l'API Binance
    OHLCV_KLINE_CLOSE_TIME: str = "kline_close_time" # Timestamp de fermeture de la kline
    OHLCV_QUOTE_ASSET_VOLUME: str = "quote_asset_volume" # Volume de l'actif de cotation
    OHLCV_NUMBER_OF_TRADES: str = "number_of_trades" # Nombre de transactions
    OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: str = "taker_buy_base_asset_volume" # Volume de l'actif de base acheté par les takers
    OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: str = "taker_buy_quote_asset_volume" # Volume de l'actif de cotation acheté par les takers
    OHLCV_IS_KLINE_CLOSED: str = "is_kline_closed" # Booléen indiquant si la kline est clôturée

    # Règles d'agrégation standard pour le resampling des klines
    AGGREGATION_RULES_STANDARD: Dict[str, str] = {
        OHLCV_OPEN: "first",
        OHLCV_HIGH: "max",
        OHLCV_LOW: "min",
        OHLCV_CLOSE: "last",
        OHLCV_VOLUME: "sum",
        OHLCV_QUOTE_ASSET_VOLUME: "sum", # Si disponible et pertinent
        OHLCV_NUMBER_OF_TRADES: "sum"    # Si disponible et pertinent
    }

# --- Constantes Numériques et Précision ---
class Precision:
    """Constantes relatives à la précision numérique."""
    DEFAULT_FLOAT_PRECISION: int = 8  # Précision décimale par défaut pour les prix et quantités de nombreuses cryptomonnaies.
    SATOSHI_PRECISION: int = 8        # Précision spécifique pour le Bitcoin (nombre de décimales d'un satoshi).
    SMALL_FLOAT_VALUE: float = 1e-9   # Très petite valeur flottante utilisée pour éviter les erreurs de division par zéro ou pour des comparaisons de flottants.
    PERCENTAGE_MULTIPLIER: float = 100.0 # Multiplicateur pour convertir les fractions en pourcentages.

# --- Constantes de Configuration/Système ---
class System:
    """Constantes générales du système et des interactions externes."""
    MAX_BINANCE_API_RETRIES: int = 5 # Nombre maximum de tentatives pour les requêtes API Binance en cas d'erreur récupérable.
    DEFAULT_API_TIMEOUT_SECONDS: int = 10 # Timeout par défaut en secondes pour les requêtes API externes.
    MAX_KLINES_PER_BINANCE_REQUEST: int = 1000 # Limite de l'API Binance pour le nombre de klines retournées par requête.
    WEBSOCKET_MAX_RECONNECT_ATTEMPTS: int = 10 # Nombre maximum de tentatives de reconnexion WebSocket.
    WEBSOCKET_RECONNECT_DELAY_SECONDS: int = 5 # Délai en secondes entre les tentatives de reconnexion WebSocket.

# --- Constantes de Noms de Colonnes DataFrame (pour analyse et backtesting) ---
class DataFrameCols:
    """Noms de colonnes standardisés utilisés dans les DataFrames pour l'analyse, les signaux et les résultats de backtesting."""
    SIGNAL: str = "signal"                # Colonne indiquant le signal de trading (ex: 1 pour long, -1 pour short, 0 pour neutre).
    ENTRY_PRICE: str = "entry_price"      # Colonne pour le prix d'entrée d'une position.
    EXIT_PRICE: str = "exit_price"        # Colonne pour le prix de sortie d'une position.
    STOP_LOSS_PRICE: str = "stop_loss"    # Colonne pour le niveau de Stop Loss calculé.
    TAKE_PROFIT_PRICE: str = "take_profit" # Colonne pour le niveau de Take Profit calculé.
    POSITION_SIZE: str = "position_size"  # Colonne pour la taille de la position (en actif de base ou en devise de cotation).
    PNL: str = "pnl"                      # Colonne pour le Profit and Loss d'un trade ou d'une période.
    EQUITY: str = "equity"                # Colonne représentant l'évolution du capital (courbe de capital).
    ATR: str = "atr"                      # Colonne pour l'Average True Range, souvent utilisé pour SL/TP.

# --- Constantes pour les types de stockage ---
class StorageTypes:
    """Types de systèmes de stockage de données supportés."""
    PARQUET: str = "parquet"    # Stockage basé sur des fichiers Parquet.
    POSTGRES: str = "postgres"  # Stockage basé sur une base de données PostgreSQL.

# --- Constantes pour les environnements ---
class Environments:
    """Environnements d'exécution de l'application."""
    DEVELOPMENT: str = "development"
    PRODUCTION: str = "production"
    TEST: str = "test"

# --- Constantes pour les modes de trading ---
class TradingModes:
    """Modes de trading supportés (en lien avec les types de comptes Binance)."""
    CROSS_MARGIN: str = "cross_margin" # Trading sur marge croisée.
    # ISOLATED_MARGIN: str = "ISOLATED_MARGIN" # Si supporté ultérieurement
    # SPOT: str = "SPOT" # Si supporté ultérieurement

# --- Constantes pour les WebSockets Binance ---
class BinanceWebsockets:
    """URL de base et suffixes pour les flux WebSocket de Binance."""
    BASE_STREAM_URL_SPOT: str = "wss://stream.binance.com:9443/ws" # URL pour les flux Spot principaux.
    BASE_STREAM_URL_FUTURES: str = "wss://fstream.binance.com/ws" # URL pour les flux Futures principaux.
    BASE_STREAM_URL_TESTNET_SPOT: str = "wss://testnet.binance.vision/ws" # URL pour les flux Spot Testnet.
    BASE_STREAM_URL_TESTNET_FUTURES: str = "wss://stream.binancefuture.com/ws" # URL pour les flux Futures Testnet.

    KLINE_STREAM_SUFFIX: str = "@kline_" # Suffixe pour les flux kline (ex: btcusdt@kline_1m).
    DEPTH_STREAM_SUFFIX: str = "@depth" # Suffixe pour les flux de profondeur du carnet d'ordres (ex: btcusdt@depth ou btcusdt@depth5@100ms).
    USER_DATA_STREAM: str = "userDataStream" # Nom du flux pour les mises à jour de compte et d'ordres (nécessite une listenKey).

