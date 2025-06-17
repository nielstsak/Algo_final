# src/core/exceptions.py

from typing import Optional, Any

class AlgoBotException(Exception):
    """Classe de base pour toutes les erreurs spécifiques à l'application."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception
        self.context = kwargs # Pour stocker des informations contextuelles supplémentaires

    def __str__(self) -> str:
        base_str = super().__str__()
        if self.original_exception:
            base_str += f" (Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)})"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_str += f" [{context_str}]"
        return base_str

# --- Configuration Exceptions ---
class ConfigurationError(AlgoBotException):
    """Erreur générale liée à la configuration de l'application."""
    pass

class InvalidConfigurationValueError(ConfigurationError):
    """Une valeur de configuration spécifique est invalide."""
    def __init__(self, message: str, parameter: Optional[str] = None, value: Any = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, parameter=parameter, value=value, **kwargs)
        self.parameter = parameter
        self.value = value

class MissingConfigurationError(ConfigurationError):
    """Un fichier, une section ou une clé de configuration est manquant."""
    def __init__(self, message: str, item: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, item=item, **kwargs)
        self.item = item

# --- API Interaction Exceptions ---
class APIError(AlgoBotException):
    """Erreur générique lors de l'interaction avec une API externe."""
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, api_name=api_name, status_code=status_code, **kwargs)
        self.api_name = api_name
        self.status_code = status_code

class BinanceAPIError(APIError):
    """Erreur spécifique à l'API Binance."""
    def __init__(self, message: str, status_code: Optional[int] = None, binance_code: Optional[int] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, api_name="Binance", status_code=status_code, original_exception=original_exception, binance_code=binance_code, **kwargs)
        self.binance_code = binance_code # Code d'erreur spécifique de Binance

class BinanceRateLimitError(BinanceAPIError):
    """Limite de taux de l'API Binance atteinte."""
    pass

class BinanceInvalidSymbolError(BinanceAPIError):
    """Symbole de trading invalide pour l'API Binance."""
    def __init__(self, message: str, symbol: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception=original_exception, symbol=symbol, **kwargs)
        self.symbol = symbol

class BinanceInsufficientFundsError(BinanceAPIError):
    """Fonds insuffisants sur le compte Binance."""
    pass

# --- Data Management Exceptions ---
class DataError(AlgoBotException):
    """Erreur générale liée à la gestion des données."""
    pass

class DataDownloadError(DataError):
    """Échec du téléchargement des données."""
    def __init__(self, message: str, source: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, source=source, **kwargs)
        self.source = source

class KlineValidationError(DataError):
    """Les données kline (OHLCV) ne sont pas valides."""
    def __init__(self, message: str, symbol: Optional[str] = None, timestamp: Optional[Any] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, symbol=symbol, timestamp=timestamp, **kwargs)
        self.symbol = symbol
        self.timestamp = timestamp

class UnsupportedDataFormatError(DataError):
    """Format de données non supporté."""
    def __init__(self, message: str, format_name: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, format_name=format_name, **kwargs)
        self.format_name = format_name

class StorageError(DataError):
    """Problème général avec le stockage des données (lecture/écriture)."""
    pass

class ParquetStorageError(StorageError):
    """Erreur spécifique au stockage Parquet."""
    pass

class PostgresStorageError(StorageError):
    """Erreur spécifique au stockage PostgreSQL."""
    pass

class CacheError(DataError):
    """Erreur liée au système de cache."""
    pass

# --- Strategy Exceptions ---
class StrategyError(AlgoBotException):
    """Erreur générale liée aux stratégies de trading."""
    def __init__(self, message: str, strategy_name: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, strategy_name=strategy_name, **kwargs)
        self.strategy_name = strategy_name

class InvalidStrategyParamsError(StrategyError):
    """Paramètres fournis pour une stratégie sont invalides."""
    def __init__(self, strategy_name: str, parameter_name: str, details: str, original_exception: Optional[Exception] = None, **kwargs: Any):
        message = f"Invalid parameter '{parameter_name}' for strategy '{strategy_name}': {details}"
        super().__init__(message, strategy_name=strategy_name, original_exception=original_exception, parameter_name=parameter_name, details=details, **kwargs)
        self.parameter_name = parameter_name
        self.details = details

class SignalGenerationError(StrategyError):
    """Erreur lors de la génération de signaux par une stratégie."""
    pass

class IndicatorCalculationError(StrategyError):
    """Erreur dans le calcul d'un indicateur technique."""
    def __init__(self, message: str, indicator_name: Optional[str] = None, strategy_name: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, strategy_name=strategy_name, original_exception=original_exception, indicator_name=indicator_name, **kwargs)
        self.indicator_name = indicator_name

class StrategyLoadError(StrategyError):
    """Échec du chargement dynamique d'une stratégie."""
    pass

# --- Backtesting Exceptions ---
# --- CORRECTION : Les classes suivantes étaient manquantes ou mal définies ---
class BacktestError(AlgoBotException):
    """Erreur générale liée au processus de backtesting."""
    pass

class BacktestFailureError(BacktestError):
    """Erreur levée quand un backtest ne peut pas être complété (ex: aucun trade)."""
    pass

class BacktestSetupError(BacktestError):
    """Erreur dans la configuration ou la préparation d'un backtest."""
    pass

class InsufficientDataForBacktestError(BacktestError):
    """Pas assez de données historiques disponibles pour effectuer le backtest."""
    def __init__(self, message: str, required_period: Optional[str] = None, available_period: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, required_period=required_period, available_period=available_period, **kwargs)
        self.required_period = required_period
        self.available_period = available_period
# --- FIN DE LA CORRECTION ---

# --- Optimization Exceptions ---
class OptimizationError(AlgoBotException):
    """Erreur générale liée au processus d'optimisation."""
    pass

class OptunaStudyError(OptimizationError):
    """Problème avec une étude Optuna (création, chargement, exécution)."""
    def __init__(self, message: str, study_name: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, study_name=study_name, **kwargs)
        self.study_name = study_name

class WFOSplitError(OptimizationError):
    """Erreur dans la génération ou la gestion des splits Walk-Forward Optimization."""
    pass

# --- Live Trading Exceptions ---
class LiveTradingError(AlgoBotException):
    """Erreur générale liée au trading en temps réel."""
    pass

class OrderPlacementError(LiveTradingError):
    """Échec du placement d'un ordre sur l'exchange."""
    def __init__(self, message: str, symbol: Optional[str] = None, order_details: Optional[dict] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, symbol=symbol, order_details=order_details, **kwargs)
        self.symbol = symbol
        self.order_details = order_details

class OrderExecutionError(LiveTradingError):
    """L'ordre n'a pas été exécuté comme prévu ou a été rejeté après placement."""
    def __init__(self, message: str, order_id: Optional[str] = None, symbol: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, order_id=order_id, symbol=symbol, **kwargs)
        self.order_id = order_id
        self.symbol = symbol

class RiskManagementViolationError(LiveTradingError):
    """Une règle de gestion des risques a été enfreinte, empêchant une action."""
    def __init__(self, message: str, rule_violated: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, rule_violated=rule_violated, **kwargs)
        self.rule_violated = rule_violated

class PositionManagementError(LiveTradingError):
    """Erreur dans la gestion des positions ouvertes."""
    pass

class WebSocketConnectionError(LiveTradingError):
    """Problème avec la connexion WebSocket (déconnexion, erreur de message)."""
    pass

# --- Core/Utility Exceptions ---
class SerializationError(AlgoBotException):
    """Problème de sérialisation ou désérialisation de données."""
    def __init__(self, message: str, data_type: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, data_type=data_type, **kwargs)
        self.data_type = data_type

class InitializationError(AlgoBotException):
    """Erreur lors de l'initialisation d'un composant ou service."""
    def __init__(self, message: str, component_name: Optional[str] = None, original_exception: Optional[Exception] = None, **kwargs: Any):
        super().__init__(message, original_exception, component_name=component_name, **kwargs)
        self.component_name = component_name