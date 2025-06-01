# src/core/config.py
import os
import yaml
from pathlib import Path
from typing import List, Literal, Optional, Any, Dict, Union

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    HttpUrl,
    validator,
    root_validator,
    ValidationError,
    PostgresDsn
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core import PydanticCustomError

# Importation de loguru pour le logging à la fin du chargement
from loguru import logger

from src.core.exceptions import (
    ConfigurationError,
    InvalidConfigurationValueError,
    MissingConfigurationError
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

PLACEHOLDER_WEBHOOK_URL = "YOUR_ALERT_WEBHOOK_URL_HERE"

# --- Modèles Pydantic pour la configuration ---

class AppConfig(BaseModel):
    """Application configuration settings."""
    name: str = "Algo Trading Bot"
    version: str = "1.0.0"
    environment: Literal['development', 'production', 'test'] = "development"

class BinanceConfigModel(BaseModel):
    """Binance API configuration settings."""
    api_key: SecretStr
    api_secret: SecretStr
    testnet: bool = False
    api_key_2: Optional[SecretStr] = None
    api_secret_2: Optional[SecretStr] = None

class DatabaseConfigModel(BaseModel):
    """Database connection and pooling settings."""
    db_type: Literal["postgresql", "sqlite"] = "postgresql"
    url: Optional[Union[PostgresDsn, str]] = None # Priorité si fourni

    # Paramètres pour construire l'URL si non fournie directement
    pg_user: Optional[str] = None
    pg_password: Optional[SecretStr] = None
    pg_host: Optional[str] = "localhost"
    pg_port: Optional[Union[int, str]] = 5434 # Peut être string dans YAML, converti par Pydantic
    pg_db: Optional[str] = "algobot_db"
    sqlite_path: Optional[str] = None # ex: "data/algobot.db" ou ":memory:"

    # Options de pooling
    pool_size: int = Field(default=5, gt=0)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout_seconds: int = Field(default=30, gt=0)
    pool_recycle_seconds: int = Field(default=3600, gt=0) # -1 pour désactiver
    pool_pre_ping_enabled: bool = True
    connect_max_retries: int = Field(default=5, ge=0)
    connect_retry_min_wait_seconds: int = Field(default=1, gt=0)
    connect_retry_max_wait_seconds: int = Field(default=10, gt=0)

    @root_validator(pre=True)
    def assemble_db_url_if_not_provided(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs the database URL from individual parameters if 'url' is not explicitly provided.
        This allows for easier configuration via separate environment variables or YAML fields.
        Priority is given to an explicitly provided 'url'.
        """
        if values.get("url"): # Si une URL complète est déjà fournie, on l'utilise
            return values

        db_type = values.get("db_type", "postgresql") # Default to postgresql if not specified

        if db_type == "postgresql":
            pg_user = values.get("pg_user")
            pg_password_secret = values.get("pg_password")
            pg_password = pg_password_secret.get_secret_value() if isinstance(pg_password_secret, SecretStr) else pg_password_secret
            pg_host = values.get("pg_host", "localhost")
            pg_port_val = values.get("pg_port", 5434) # Default port for PostgreSQL
            pg_port = str(pg_port_val) if pg_port_val is not None else "5434"
            pg_db = values.get("pg_db", "algobot_db")

            # Tous les composants PostgreSQL doivent être présents pour construire l'URL
            if all(val is not None for val in [pg_user, pg_password, pg_host, pg_db]):
                # Utilisation de psycopg2 comme driver par défaut pour SQLAlchemy avec PostgreSQL
                values["url"] = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
            # Si des composants manquent, la validation de 'url' plus bas échouera si elle est requise.
        
        elif db_type == "sqlite":
            sqlite_path = values.get("sqlite_path")
            if sqlite_path:
                if sqlite_path == ":memory:":
                    values["url"] = "sqlite:///:memory:"
                else:
                    # Assurer que le chemin est absolu ou relatif à la racine du projet
                    path_obj = Path(sqlite_path)
                    if not path_obj.is_absolute():
                        # PROJECT_ROOT est défini globalement dans ce module
                        path_obj = PROJECT_ROOT / path_obj 
                    values["url"] = f"sqlite:///{path_obj.resolve()}"
            # Si sqlite_path est None, la validation de 'url' échouera si elle est requise.
        return values

    @validator('url', always=True) # S'exécute après `assemble_db_url_if_not_provided`
    def check_db_url_final_state(cls, v: Optional[Union[PostgresDsn, str]], values: Dict[str, Any]) -> Optional[Union[PostgresDsn, str]]:
        """
        Validates the final state of the database URL.
        Ensures that a URL is present if the db_type requires it (e.g., PostgreSQL or file-based SQLite).
        """
        db_type = values.get('db_type')
        # Une URL est requise si db_type est postgresql, ou si c'est sqlite avec un chemin de fichier (pas :memory:)
        is_persistent_sqlite = db_type == 'sqlite' and values.get('sqlite_path') and values.get('sqlite_path') != ":memory:"
        
        if not v and (db_type == 'postgresql' or is_persistent_sqlite):
             # This error should ideally be caught by Pydantic if 'url' was non-optional and no default_factory.
             # However, since 'url' is Optional and we try to build it, this explicit check is useful.
             raise ValueError(f"{db_type.capitalize()} database URL is required but could not be constructed and was not provided.")

        if v: # Si une URL a été fournie ou construite
            if db_type == 'postgresql':
                # La validation PostgresDsn est gérée par Pydantic si 'v' est de ce type.
                # Si c'est une chaîne, on vérifie le préfixe.
                if isinstance(v, str) and not v.startswith("postgresql+psycopg2://"):
                    # Pydantic lèvera une erreur si la conversion en PostgresDsn échoue.
                    # On peut ajouter une vérification de préfixe pour plus de clarté si c'est une chaîne.
                    pass # La validation de Pydantic pour PostgresDsn est suffisante.
            elif db_type == 'sqlite':
                if isinstance(v, str) and not v.startswith("sqlite:///"):
                    raise ValueError("Invalid SQLite URL format. Expected 'sqlite:///...'")
        return v


class RedisConfigModel(BaseModel):
    """Redis connection settings."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[SecretStr] = None

class DataConfig(BaseModel):
    """Data storage and processing settings."""
    storage_type: Literal['parquet', 'postgres'] = "parquet"
    storage_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "historical")
    parquet_partition_cols: List[str] = ['pair', 'year', 'month']
    cache_enabled: bool = True
    cache_ttl: int = Field(default=3600, gt=0) # en secondes
    parquet_export_enabled: bool = False # Feature pour exporter des données vers Parquet
    parquet_export_path: Optional[Path] = None # Chemin pour les exports Parquet

    @validator('storage_path', pre=True, always=True)
    def ensure_storage_path_is_absolute(cls, v: Union[str, Path]) -> Path:
        """Ensures that storage_path is an absolute path, resolving relative to PROJECT_ROOT."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return v.resolve()

    @validator('parquet_export_path', pre=True, always=True)
    def check_export_path_if_enabled(cls, v: Optional[Union[str, Path]], values: Dict[str, Any]) -> Optional[Path]:
        """Validates parquet_export_path if parquet_export_enabled is True."""
        if values.get('parquet_export_enabled') and v is None:
            # Utilisation de PydanticCustomError pour un message plus structuré si nécessaire,
            # mais InvalidConfigurationValueError est déjà assez descriptif.
            raise InvalidConfigurationValueError(
                message="parquet_export_path must be set if parquet_export_enabled is True",
                parameter="parquet_export_path"
            )
        if v:
            if isinstance(v, str):
                v = Path(v)
            if not v.is_absolute():
                return (PROJECT_ROOT / v).resolve()
            return v.resolve() if v else None
        return None

class TradingConfig(BaseModel):
    """Trading parameters and strategy settings."""
    mode: Literal['cross_margin'] = "cross_margin" # Actuellement, seul cross_margin est supporté
    base_currency: Literal['USDC', 'USDT', 'BUSD', 'BTC', 'ETH'] = "USDC"
    allowed_pairs: List[str] # Liste des paires de trading autorisées, ex: ["BTCUSDC", "ETHUSDC"]

    @validator('allowed_pairs', each_item=True)
    def check_pair_format(cls, v: str) -> str:
        """Validates the format of each trading pair."""
        if not v.isupper() or not v.isalnum() or len(v) < 6: # Simple validation
            raise InvalidConfigurationValueError(
                message=f"Trading pair '{v}' format is invalid. Expected uppercase alphanumeric, e.g., 'BTCUSDC'.",
                parameter="allowed_pairs",
                value=v
            )
        return v

class RiskConfig(BaseModel):
    """Risk management parameters."""
    max_position_pct: float = Field(0.1, gt=0, le=1, description="Maximum percentage of capital per position.")
    max_drawdown_pct: float = Field(0.15, gt=0, le=1, description="Maximum allowable portfolio drawdown percentage.")
    daily_loss_limit_pct: float = Field(0.05, gt=0, le=1, description="Maximum daily loss percentage before stopping trading.")

class MonitoringConfig(BaseModel):
    """Monitoring and alerting settings."""
    prometheus_enabled: bool = False
    prometheus_port: int = Field(9090, gt=1023, lt=65536)
    alert_webhook_url: Optional[HttpUrl] = None # URL pour les alertes (ex: Slack, Discord)
    log_level_console: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "INFO"
    log_level_file: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "DEBUG"
    log_to_file_enabled: bool = True
    log_rotation: str = "1 day" # Format Loguru: "100 MB", "1 week", "00:00"
    log_retention: str = "7 days" # Format Loguru: "1 month"
    log_compression: Optional[Literal["gz", "bz2", "zip", "xz", "lzma", "tar", "tar.gz", "tar.bz2", "tar.xz"]] = "zip"

    @validator('alert_webhook_url', pre=True)
    def handle_placeholder_webhook_url(cls, v: Any) -> Optional[Any]:
        """Converts a placeholder string for webhook URL to None."""
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

class Settings(BaseSettings):
    """
    Main settings class, aggregates all configurations.
    Loads from YAML, .env file, and environment variables.
    """
    # Variables d'environnement directes pour pydantic-settings
    # Ces variables peuvent surcharger les valeurs du YAML ou du .env
    # Leurs noms doivent correspondre exactement aux variables d'environnement.
    BINANCE_API_KEY: Optional[SecretStr] = None
    BINANCE_API_SECRET: Optional[SecretStr] = None
    BINANCE_TESTNET: Optional[bool] = None # Laisser Pydantic gérer le type booléen depuis str
    BINANCE_API_KEY_2: Optional[SecretStr] = None
    BINANCE_API_SECRET_2: Optional[SecretStr] = None

    DB_TYPE: Optional[Literal["postgresql", "sqlite"]] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[SecretStr] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[Union[int, str]] = None # Peut être string dans .env, Pydantic le convertira
    POSTGRES_DB: Optional[str] = None
    SQLITE_PATH: Optional[str] = None
    DATABASE_URL: Optional[Union[PostgresDsn, str]] = None # URL complète peut être fournie via .env

    DB_POOL_SIZE: Optional[int] = None
    DB_MAX_OVERFLOW: Optional[int] = None
    DB_POOL_TIMEOUT_SECONDS: Optional[int] = None
    DB_POOL_RECYCLE_SECONDS: Optional[int] = None
    DB_POOL_PRE_PING_ENABLED: Optional[bool] = None
    DB_CONNECT_MAX_RETRIES: Optional[int] = None
    DB_CONNECT_RETRY_MIN_WAIT_SECONDS: Optional[int] = None
    DB_CONNECT_RETRY_MAX_WAIT_SECONDS: Optional[int] = None

    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    REDIS_PASSWORD: Optional[SecretStr] = None
    
    ALERT_WEBHOOK_URL: Optional[HttpUrl] = None # Variable d'env de haut niveau pour le webhook

    # Sections qui seront principalement peuplées par le fichier YAML,
    # mais peuvent aussi avoir des valeurs par défaut via Field(default_factory=...)
    # ou être influencées par des variables d'environnement si leurs champs correspondent.
    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    # 'trading' est requis par le YAML (ou doit être entièrement défini par des env vars si Pydantic le supportait directement pour les sous-modèles complexes).
    # Sa présence est validée dans perform_cross_model_validation.
    trading: Optional[TradingConfig] = None # Rendu Optional ici, validé plus tard
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: Optional[MonitoringConfig] = Field(default_factory=MonitoringConfig) # Rendu Optional, car on le prépare dans prepare_nested
    
    # Champs pour les modèles imbriqués. Pydantic essaiera de les construire
    # à partir des dictionnaires préparés par prepare_nested_model_inputs.
    # Ces champs sont marqués Optional car leur existence dépend de la présence des
    # clés correspondantes ('binance', 'database', 'redis') dans les données d'entrée (YAML + env).
    # Leur nécessité est ensuite validée dans perform_cross_model_validation.
    binance: Optional[BinanceConfigModel] = None
    database: Optional[DatabaseConfigModel] = None
    redis: Optional[RedisConfigModel] = Field(default_factory=RedisConfigModel) # Peut avoir des défauts

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_prefix='', # Pas de préfixe pour les variables d'env de haut niveau
        extra='ignore', # Ignorer les variables d'env supplémentaires non mappées
        populate_by_name=True # Permet d'utiliser des alias (non utilisé ici mais bonne pratique)
    )

    @validator('ALERT_WEBHOOK_URL', pre=True)
    def handle_placeholder_top_level_webhook_url(cls, v: Any) -> Optional[Any]:
        """Converts a placeholder string for top-level ALERT_WEBHOOK_URL to None."""
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

    @root_validator(pre=True) # S'exécute AVANT la validation des champs individuels de Settings
    def prepare_nested_model_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares input dictionaries for nested Pydantic models (binance, database, redis, monitoring).
        It merges data from the YAML (already in `values` if `Settings(**yaml_data)` was called)
        with direct environment variables (also already in `values` due to BaseSettings).
        The resulting keys ('binance', 'database', 'redis', 'monitoring') in `values` will be dictionaries
        that Pydantic will then use to instantiate BinanceConfigModel, DatabaseConfigModel, etc.
        Environment variables take precedence over YAML values for the same logical field.
        """
        
        # Pour Binance:
        # `values` contient déjà les clés BINANCE_API_KEY, etc. de l'env et la section `binance` du YAML.
        binance_yaml_data = values.pop('binance', {}) # Récupère la section YAML 'binance'
        if not isinstance(binance_yaml_data, dict): binance_yaml_data = {}
        
        values['binance'] = { # Prépare le dict pour le champ Settings.binance
            "api_key": values.get('BINANCE_API_KEY', binance_yaml_data.get('api_key')),
            "api_secret": values.get('BINANCE_API_SECRET', binance_yaml_data.get('api_secret')),
            "testnet": values.get('BINANCE_TESTNET', binance_yaml_data.get('testnet', False)), # False est le défaut Pydantic
            "api_key_2": values.get('BINANCE_API_KEY_2', binance_yaml_data.get('api_key_2')),
            "api_secret_2": values.get('BINANCE_API_SECRET_2', binance_yaml_data.get('api_secret_2')),
        }

        # Pour Database:
        database_yaml_data = values.pop('database', {})
        if not isinstance(database_yaml_data, dict): database_yaml_data = {}
        values['database'] = {
            "db_type": values.get('DB_TYPE', database_yaml_data.get('db_type', "postgresql")),
            "pg_user": values.get('POSTGRES_USER', database_yaml_data.get('pg_user')),
            "pg_password": values.get('POSTGRES_PASSWORD', database_yaml_data.get('pg_password')),
            "pg_host": values.get('POSTGRES_HOST', database_yaml_data.get('pg_host', "localhost")),
            "pg_port": values.get('POSTGRES_PORT', database_yaml_data.get('pg_port', 5434)),
            "pg_db": values.get('POSTGRES_DB', database_yaml_data.get('pg_db', "algobot_db")),
            "sqlite_path": values.get('SQLITE_PATH', database_yaml_data.get('sqlite_path')),
            "url": values.get('DATABASE_URL', database_yaml_data.get('url')),
            "pool_size": values.get('DB_POOL_SIZE', database_yaml_data.get('pool_size', 5)),
            "max_overflow": values.get('DB_MAX_OVERFLOW', database_yaml_data.get('max_overflow', 10)),
            "pool_timeout_seconds": values.get('DB_POOL_TIMEOUT_SECONDS', database_yaml_data.get('pool_timeout_seconds', 30)),
            "pool_recycle_seconds": values.get('DB_POOL_RECYCLE_SECONDS', database_yaml_data.get('pool_recycle_seconds', 3600)),
            "pool_pre_ping_enabled": values.get('DB_POOL_PRE_PING_ENABLED', database_yaml_data.get('pool_pre_ping_enabled', True)),
            "connect_max_retries": values.get('DB_CONNECT_MAX_RETRIES', database_yaml_data.get('connect_max_retries', 5)),
            "connect_retry_min_wait_seconds": values.get('DB_CONNECT_RETRY_MIN_WAIT_SECONDS', database_yaml_data.get('connect_retry_min_wait_seconds', 1)),
            "connect_retry_max_wait_seconds": values.get('DB_CONNECT_RETRY_MAX_WAIT_SECONDS', database_yaml_data.get('connect_retry_max_wait_seconds', 10)),
        }
        
        # Pour Redis:
        redis_yaml_data = values.pop('redis', {})
        if not isinstance(redis_yaml_data, dict): redis_yaml_data = {}
        values['redis'] = {
            "host": values.get('REDIS_HOST', redis_yaml_data.get('host', "localhost")),
            "port": values.get('REDIS_PORT', redis_yaml_data.get('port', 6379)),
            "password": values.get('REDIS_PASSWORD', redis_yaml_data.get('password')),
        }

        # Pour Monitoring:
        # Le champ `monitoring` de `Settings` est `Optional[MonitoringConfig]`.
        # On prend les données YAML pour `monitoring` (si elles existent) et on les fusionne
        # avec la variable d'environnement `ALERT_WEBHOOK_URL` de haut niveau.
        monitoring_data_from_yaml = values.pop('monitoring', {}) # Récupère section YAML 'monitoring'
        if not isinstance(monitoring_data_from_yaml, dict): monitoring_data_from_yaml = {}
        
        # La variable d'environnement ALERT_WEBHOOK_URL (déjà dans `values`) a priorité.
        # Si elle est définie (et non placeholder), elle écrase `alert_webhook_url` du YAML.
        # Si elle est None (après validation du placeholder), la valeur du YAML sera utilisée.
        # Le validateur de MonitoringConfig.alert_webhook_url gère le placeholder s'il vient du YAML.
        if 'ALERT_WEBHOOK_URL' in values and values['ALERT_WEBHOOK_URL'] is not None:
            monitoring_data_from_yaml['alert_webhook_url'] = values['ALERT_WEBHOOK_URL']
        
        values['monitoring'] = monitoring_data_from_yaml # Assure que 'monitoring' est un dict pour Pydantic

        # Les autres sections comme 'app', 'data', 'trading', 'risk' sont lues directement
        # par Pydantic à partir de `values` si elles sont présentes dans le YAML.
        # Leurs modèles respectifs (AppConfig, DataConfig, etc.) ont des valeurs par défaut
        # ou des `default_factory` si les sections sont absentes du YAML et qu'aucun env var ne les définit.
        return values

    @root_validator(pre=False, skip_on_failure=True) # S'exécute APRES la validation des champs individuels
    def perform_cross_model_validation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs cross-model validation after individual fields and nested models are validated.
        Ensures consistency between different configuration sections.
        """
        # À ce stade, values['binance'], values['database'], values['redis'], values['trading'], etc.
        # devraient être des instances de leurs modèles respectifs (ou None si Optional et non fourni/construit).
        
        binance_cfg: Optional[BinanceConfigModel] = values.get('binance')
        if not binance_cfg or not isinstance(binance_cfg, BinanceConfigModel):
            raise MissingConfigurationError(message="Binance configuration (API keys) is required and could not be loaded.", item="binance")
        # Vérifier si les clés API sont présentes (Pydantic le fait déjà si non Optional)
        if not binance_cfg.api_key or not binance_cfg.api_secret:
             raise MissingConfigurationError(message="Binance api_key and api_secret are required.", item="binance.api_key/api_secret")


        db_config: Optional[DatabaseConfigModel] = values.get('database')
        if not db_config or not isinstance(db_config, DatabaseConfigModel):
             raise MissingConfigurationError(message="Database configuration is required and could not be loaded.", item="database")
        # La validation de db_config.url est déjà faite dans DatabaseConfigModel.
        # On s'assure juste que l'objet db_config lui-même est là.

        # Valider la présence de la section 'trading' car elle est essentielle
        trading_config: Optional[TradingConfig] = values.get('trading')
        if not trading_config or not isinstance(trading_config, TradingConfig):
            raise MissingConfigurationError(message="Trading configuration section ('trading:') is missing or invalid in YAML config file.", item="trading")
        if not trading_config.allowed_pairs:
            raise InvalidConfigurationValueError(message="Trading configuration must include at least one pair in 'allowed_pairs'.", parameter="trading.allowed_pairs")


        # Valider la cohérence entre data.storage_type et la configuration de la base de données
        data_config: Optional[DataConfig] = values.get('data')
        if data_config and data_config.storage_type == 'postgres':
            if not db_config or db_config.db_type != 'postgresql':
                raise InvalidConfigurationValueError(
                    message="If data.storage_type is 'postgres', then database.db_type must be 'postgresql'.",
                    parameter="data.storage_type/database.db_type"
                )
            if not db_config.url or not (isinstance(db_config.url, PostgresDsn) or (isinstance(db_config.url, str) and db_config.url.startswith("postgresql"))):
                 raise InvalidConfigurationValueError(
                    message="A valid PostgreSQL URL (database.url) is required when data.storage_type is 'postgres'.",
                    parameter="database.url"
                )
        return values


def load_raw_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Loads raw configuration data from a YAML file.
    Returns an empty dict if the file does not exist.
    """
    if not config_path.exists():
        # logger.warning(f"Configuration file {config_path} not found. Proceeding with defaults/env vars.")
        # Le logging n'est pas encore configuré à ce stade très précoce.
        # Un print peut être utile pour le débogage initial, mais peut être bruyant.
        # print(f"INFO: Configuration file {config_path} not found. Using defaults/env vars.")
        return {} 
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        if raw_config is None: # Fichier YAML vide
             # print(f"INFO: Configuration file {config_path} is empty. Using defaults/env vars.")
             return {} 
        if not isinstance(raw_config, dict):
            raise InvalidConfigurationValueError(
                message=f"Configuration file {config_path} content is not a valid YAML dictionary.",
                parameter="config_file_format"
            )
        return raw_config
    except yaml.YAMLError as e:
        # Erreur de parsing YAML
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}", original_exception=e)
    except Exception as e: # Autres erreurs de lecture de fichier
        raise ConfigurationError(f"Could not load configuration from {config_path}: {e}", original_exception=e)

def load_settings() -> Settings:
    """
    Loads application settings from YAML, .env file, and environment variables.
    Pydantic-Settings handles the merging and type validation.
    """
    try:
        # Charger les données brutes du YAML.
        # Pydantic-Settings (BaseSettings) chargera ensuite les variables d'environnement et .env.
        # Les données du YAML sont passées comme **kwargs à l'initialiseur de Settings.
        yaml_data = load_raw_config_from_yaml(CONFIG_FILE_PATH)
        
        # L'ordre de priorité de pydantic-settings est typiquement:
        # 1. Arguments d'initialisation (ici, `**yaml_data`)
        # 2. Variables d'environnement
        # 3. Variables du fichier .env
        # 4. Valeurs par défaut du modèle
        # Nos validateurs `prepare_nested_model_inputs` et `perform_cross_model_validation`
        # s'exécuteront sur ces données fusionnées.
        settings_instance = Settings(**yaml_data)
        return settings_instance

    except ValidationError as e: # Erreur de validation Pydantic (y compris celles de nos validateurs)
        error_details = []
        # Utiliser include_input=False pour ne pas afficher les valeurs sensibles dans les logs/sorties
        # Pour le débogage local, on peut mettre include_input=True temporairement.
        for error in e.errors(include_url=False, include_input=False): 
            loc_str = " -> ".join(map(str, error['loc']))
            msg = error['msg']
            # inp_val_debug = error.get('input') # Pour le débogage, ne pas mettre en production
            # if isinstance(inp_val_debug, dict) and len(inp_val_debug) > 5: inp_repr_debug = f"{{...dict with {len(inp_val_debug)} keys...}}"
            # elif isinstance(inp_val_debug, SecretStr): inp_repr_debug = "[SECRET]"
            # else: inp_repr_debug = repr(inp_val_debug)
            # error_details.append(f"  - Location: '{loc_str}', Message: '{msg}', Input: {inp_repr_debug}")
            error_details.append(f"  - Location: '{loc_str}', Message: '{msg}'")


        error_messages_str = "\n".join(error_details)
        # Lever notre exception personnalisée pour une meilleure gestion en amont
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages_str}", original_exception=e)
    except ConfigurationError: # Si nos fonctions lèvent déjà une ConfigurationError
        raise 
    except Exception as e: # Autres erreurs inattendues
        # print(f"UNEXPECTED ERROR TYPE: {type(e).__name__} - {e}") # Pour débogage
        raise ConfigurationError(f"An unexpected error occurred while loading settings: {type(e).__name__} - {e}", original_exception=e)

# --- Instance Globale des Paramètres ---
# Cette instance sera chargée lors de la première importation de ce module.
# Le logging sera configuré après le chargement réussi de `settings`.

try:
    settings: Settings = load_settings()
    
    # Importation locale pour éviter une dépendance circulaire au niveau du module
    # si logging_config importe lui-même des éléments de config (ce qui n'est pas le cas ici directement pour settings).
    # Cet import est sûr ici car `settings` est maintenant chargé.
    from src.core.logging_config import setup_logging 
    
    # Initialiser le logging avec la configuration fraîchement chargée
    setup_logging(settings) 
    
    # Log de succès après l'initialisation du logging
    logger.info(f"Configuration loaded successfully. Environment: {settings.app.environment}, App Version: {settings.app.version}")
    # Optionnel: logger les settings complets en mode DEBUG (attention aux secrets si non masqués)
    if settings.monitoring and settings.monitoring.log_level_console == "DEBUG":
         # model_dump_json masque les SecretStr par défaut.
        logger.debug(f"Full settings loaded (sensitive fields may be masked):\n{settings.model_dump_json(indent=2)}")

except ConfigurationError as e:
    # Ce print est crucial car Loguru n'est pas (ou ne peut pas être) configuré si load_settings() échoue.
    print(f"CRITICAL CONFIGURATION ERROR during settings load: {e}")
    # Il est important de relancer l'exception pour arrêter l'application si la config est mauvaise.
    raise
except Exception as e: # Attraper d'autres erreurs potentielles durant l'init du logging lui-même
    print(f"CRITICAL ERROR during logging setup after successful settings load: {e}")
    # Si le logging lui-même échoue, c'est aussi un problème critique.
    raise
