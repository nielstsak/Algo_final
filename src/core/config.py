# src/core/config.py
import os
import yaml
import json # Added for parsing JSON strings from env vars
from pathlib import Path
from typing import List, Literal, Optional, Any, Dict, Union

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    HttpUrl,
    validator, # field_validator is preferred in Pydantic V2
    root_validator, # model_validator is preferred in Pydantic V2
    ValidationError,
    PostgresDsn,
    field_validator # Added for Pydantic V2 style
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
# --- CORRECTION ---
# Le chemin pointe maintenant vers le dossier /configs/
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
    url: Optional[Union[PostgresDsn, str]] = None

    pg_user: Optional[str] = "algobot_user" 
    pg_password: Optional[SecretStr] = "PeLtj20h" 
    pg_host: Optional[str] = "localhost"
    pg_port: Optional[Union[int, str]] = 5434
    pg_db: Optional[str] = "algobot_db" 
    sqlite_path: Optional[str] = "data/algobot_main.db" 

    pool_size: int = Field(default=5, gt=0)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout_seconds: int = Field(default=30, gt=0)
    pool_recycle_seconds: int = Field(default=3600, gt=0)
    pool_pre_ping_enabled: bool = True
    connect_max_retries: int = Field(default=5, ge=0)
    connect_retry_min_wait_seconds: int = Field(default=1, gt=0)
    connect_retry_max_wait_seconds: int = Field(default=10, gt=0)

    @field_validator('pg_port', mode='before') 
    @classmethod
    def coerce_pg_port_to_int(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"pg_port must be a valid integer or integer string, got {v}")
        if isinstance(v, int):
            return v
        raise ValueError(f"Invalid type for pg_port: {type(v)}")


    @model_validator(mode='before')
    @classmethod
    def assemble_db_url_if_not_provided(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("url"):
            url_str = str(values["url"])
            if url_str.startswith("postgresql"):
                values["db_type"] = "postgresql"
            elif url_str.startswith("sqlite"):
                values["db_type"] = "sqlite"
            return values

        db_type = values.get("db_type", cls.model_fields['db_type'].default) 

        if db_type == "postgresql":
            pg_user = values.get("pg_user", cls.model_fields['pg_user'].default)
            pg_password_secret = values.get("pg_password", cls.model_fields['pg_password'].default)
            pg_password = pg_password_secret.get_secret_value() if isinstance(pg_password_secret, SecretStr) else pg_password_secret
            pg_host = values.get("pg_host", cls.model_fields['pg_host'].default)
            pg_port_val = values.get("pg_port", cls.model_fields['pg_port'].default)
            pg_port = str(pg_port_val) 
            pg_db = values.get("pg_db", cls.model_fields['pg_db'].default)

            if all(val is not None for val in [pg_user, pg_password, pg_host, pg_db, pg_port]):
                values["url"] = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        
        elif db_type == "sqlite":
            sqlite_path = values.get("sqlite_path", cls.model_fields['sqlite_path'].default)
            if sqlite_path:
                if sqlite_path == ":memory:":
                    values["url"] = "sqlite:///:memory:"
                else:
                    path_obj = Path(sqlite_path)
                    if not path_obj.is_absolute():
                        path_obj = PROJECT_ROOT / path_obj
                    values["url"] = f"sqlite:///{path_obj.resolve().as_posix()}"
        return values

    @field_validator('url', mode='after')
    @classmethod
    def check_db_url_final_state(cls, v: Optional[Union[PostgresDsn, str]], info) -> Optional[Union[PostgresDsn, str]]:
        values = info.data
        db_type = values.get('db_type')
        
        if v and isinstance(v, str):
            if v.startswith("postgresql"):
                db_type = "postgresql"
            elif v.startswith("sqlite"):
                db_type = "sqlite"
        
        is_persistent_sqlite = db_type == 'sqlite' and values.get('sqlite_path') and values.get('sqlite_path') != ":memory:"
        
        if not v and (db_type == 'postgresql' or is_persistent_sqlite):
             raise ValueError(f"{str(db_type).capitalize()} database URL is required but could not be constructed and was not provided.")

        if v:
            if db_type == 'postgresql':
                if isinstance(v, str) and not v.startswith("postgresql"): 
                    try:
                        return PostgresDsn(v) 
                    except Exception as e:
                        raise ValueError(f"Invalid PostgreSQL URL format: {v}. Error: {e}")
            elif db_type == 'sqlite':
                if isinstance(v, str) and not v.startswith("sqlite:///"):
                    raise ValueError("Invalid SQLite URL format. Expected 'sqlite:///...'")
        return v


class RedisConfigModel(BaseModel):
    host: str = "localhost"
    port: int = 6379
    password: Optional[SecretStr] = None

class DataConfig(BaseModel):
    storage_type: Literal['parquet', 'postgres'] = "parquet"
    storage_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "historical")
    parquet_partition_cols: List[str] = ['pair', 'year', 'month']
    cache_enabled: bool = True
    cache_ttl: int = Field(default=3600, gt=0)
    parquet_export_enabled: bool = False
    parquet_export_path: Optional[Path] = None

    @field_validator('storage_path', 'parquet_export_path', mode='before') 
    @classmethod
    def ensure_path_is_absolute(cls, v: Any, info: Any) -> Optional[Path]: 
        if v is None and info.field_name == 'parquet_export_path': 
            return None
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return v.resolve()

    @model_validator(mode='after')
    @classmethod
    def check_export_path_if_enabled(cls, model) -> 'DataConfig':
        if model.parquet_export_enabled and model.parquet_export_path is None:
            raise InvalidConfigurationValueError(
                message="parquet_export_path must be set if parquet_export_enabled is True",
                parameter="parquet_export_path"
            )
        return model

class TradingConfig(BaseModel):
    mode: Literal['cross_margin'] = "cross_margin"
    base_currency: Literal['USDC', 'USDT', 'BUSD', 'BTC', 'ETH'] = "USDC"
    allowed_pairs: List[str] = Field(default_factory=list) 

    @field_validator('allowed_pairs', mode='after')
    @classmethod
    def check_pair_format(cls, v: str) -> str:
        # This check is too strict for pairs with "/"
        # A better check would be to see if it matches a regex or just ensure it's a string.
        # For now, let's relax it slightly. The main validation is that the list is not empty.
        if not isinstance(v, str) or len(v) < 6:
             raise InvalidConfigurationValueError(
                 message=f"Trading pair '{v}' format is invalid. Expected a string like 'BTC/USDT'.",
                 parameter="allowed_pairs", value=v
             )
        return v.replace("/", "") # Standardize to a format without slashes internally if needed

class RiskConfig(BaseModel):
    max_position_pct: float = Field(0.1, gt=0, le=1)
    max_drawdown_pct: float = Field(0.15, gt=0, le=1)
    daily_loss_limit_pct: float = Field(0.05, gt=0, le=1)

class MonitoringConfig(BaseModel):
    prometheus_enabled: bool = False
    prometheus_port: int = Field(9090, gt=1023, lt=65536)
    alert_webhook_url: Optional[HttpUrl] = None
    log_level_console: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "INFO"
    log_level_file: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "DEBUG"
    log_to_file_enabled: bool = True
    log_rotation: str = "1 day"
    log_retention: str = "7 days"
    log_compression: Optional[Literal["gz", "bz2", "zip", "xz", "lzma", "tar", "tar.gz", "tar.bz2", "tar.xz"]] = "zip"

    @field_validator('alert_webhook_url')
    @classmethod
    def handle_placeholder_webhook_url(cls, v: Any) -> Optional[Any]:
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

class SystemConfig(BaseModel): # NEW MODEL
    """System-level configurations."""
    thread_pool_workers: int = Field(default=4, gt=0, description="Number of workers for thread pools.")
    # Add other system-wide settings here if needed

class Settings(BaseSettings):
    APP_NAME: Optional[str] = None
    APP_VERSION: Optional[str] = None
    APP_ENV: Optional[Literal['development', 'production', 'test']] = None

    BINANCE_API_KEY: Optional[SecretStr] = None
    BINANCE_API_SECRET: Optional[SecretStr] = None
    BINANCE_TESTNET: Optional[bool] = None
    BINANCE_API_KEY_2: Optional[SecretStr] = None
    BINANCE_API_SECRET_2: Optional[SecretStr] = None

    DB_TYPE: Optional[Literal["postgresql", "sqlite"]] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[SecretStr] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[Union[int,str]] = None 
    POSTGRES_DB: Optional[str] = None
    SQLITE_PATH: Optional[str] = None
    DATABASE_URL: Optional[Union[PostgresDsn, str]] = None
    DB_POOL_SIZE: Optional[int] = None
    
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    REDIS_PASSWORD: Optional[SecretStr] = None
    
    ALERT_WEBHOOK_URL: Optional[HttpUrl] = None 

    TRADING_MODE: Optional[Literal['cross_margin']] = None
    TRADING_BASE_CURRENCY: Optional[Literal['USDC', 'USDT', 'BUSD', 'BTC', 'ETH']] = None
    TRADING_ALLOWED_PAIRS: Optional[str] = None 

    # Nested models
    app: AppConfig = Field(default_factory=AppConfig)
    binance: Optional[BinanceConfigModel] = None
    database: Optional[DatabaseConfigModel] = None
    redis: Optional[RedisConfigModel] = Field(default_factory=RedisConfigModel)
    data: DataConfig = Field(default_factory=DataConfig)
    trading: Optional[TradingConfig] = None
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    system: SystemConfig = Field(default_factory=SystemConfig) # ADDED SYSTEM CONFIG

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_prefix='', 
        extra='ignore',
        populate_by_name=True 
    )

    @field_validator('ALERT_WEBHOOK_URL')
    @classmethod
    def handle_placeholder_top_level_webhook_url(cls, v: Any) -> Optional[Any]:
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

    @model_validator(mode='before')
    @classmethod
    def prepare_nested_model_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # --- App Config ---
        app_yaml_data = values.pop('app', {})
        if not isinstance(app_yaml_data, dict): app_yaml_data = {}
        values['app'] = {
            "name": values.get('APP_NAME', app_yaml_data.get('name', AppConfig.model_fields['name'].default)),
            "version": values.get('APP_VERSION', app_yaml_data.get('version', AppConfig.model_fields['version'].default)),
            "environment": values.get('APP_ENV', app_yaml_data.get('environment', AppConfig.model_fields['environment'].default)),
        }
        
        # --- Binance Config ---
        binance_yaml_data = values.pop('binance', {})
        if not isinstance(binance_yaml_data, dict): binance_yaml_data = {}
        values['binance'] = {
            "api_key": values.get('BINANCE_API_KEY', binance_yaml_data.get('api_key')),
            "api_secret": values.get('BINANCE_API_SECRET', binance_yaml_data.get('api_secret')),
            "testnet": values.get('BINANCE_TESTNET', binance_yaml_data.get('testnet', BinanceConfigModel.model_fields['testnet'].default)),
            "api_key_2": values.get('BINANCE_API_KEY_2', binance_yaml_data.get('api_key_2')),
            "api_secret_2": values.get('BINANCE_API_SECRET_2', binance_yaml_data.get('api_secret_2')),
        }

        # --- Database Config ---
        db_yaml_data = values.pop('database', {})
        if not isinstance(db_yaml_data, dict): db_yaml_data = {}
        values['database'] = {
            "db_type": values.get('DB_TYPE', db_yaml_data.get('db_type', DatabaseConfigModel.model_fields['db_type'].default)),
            "pg_user": values.get('POSTGRES_USER', db_yaml_data.get('pg_user', DatabaseConfigModel.model_fields['pg_user'].default)),
            "pg_password": values.get('POSTGRES_PASSWORD', db_yaml_data.get('pg_password', DatabaseConfigModel.model_fields['pg_password'].default)),
            "pg_host": values.get('POSTGRES_HOST', db_yaml_data.get('pg_host', DatabaseConfigModel.model_fields['pg_host'].default)),
            "pg_port": values.get('POSTGRES_PORT', db_yaml_data.get('pg_port', DatabaseConfigModel.model_fields['pg_port'].default)),
            "pg_db": values.get('POSTGRES_DB', db_yaml_data.get('pg_db', DatabaseConfigModel.model_fields['pg_db'].default)),
            "sqlite_path": values.get('SQLITE_PATH', db_yaml_data.get('sqlite_path', DatabaseConfigModel.model_fields['sqlite_path'].default)),
            "url": values.get('DATABASE_URL', db_yaml_data.get('url')), 
            "pool_size": values.get('DB_POOL_SIZE', db_yaml_data.get('pool_size', DatabaseConfigModel.model_fields['pool_size'].default)),
            # Propagate other DB pool settings if they can be set by env vars too
            "max_overflow": db_yaml_data.get('max_overflow', DatabaseConfigModel.model_fields['max_overflow'].default),
            "pool_timeout_seconds": db_yaml_data.get('pool_timeout_seconds', DatabaseConfigModel.model_fields['pool_timeout_seconds'].default),
            "pool_recycle_seconds": db_yaml_data.get('pool_recycle_seconds', DatabaseConfigModel.model_fields['pool_recycle_seconds'].default),
            "pool_pre_ping_enabled": db_yaml_data.get('pool_pre_ping_enabled', DatabaseConfigModel.model_fields['pool_pre_ping_enabled'].default),
        }
        
        # --- Redis Config ---
        redis_yaml_data = values.pop('redis', {})
        if not isinstance(redis_yaml_data, dict): redis_yaml_data = {}
        values['redis'] = {
            "host": values.get('REDIS_HOST', redis_yaml_data.get('host', RedisConfigModel.model_fields['host'].default)),
            "port": values.get('REDIS_PORT', redis_yaml_data.get('port', RedisConfigModel.model_fields['port'].default)),
            "password": values.get('REDIS_PASSWORD', redis_yaml_data.get('password')),
        }

        # --- Trading Config ---
        trading_yaml_data = values.pop('trading', {})
        if not isinstance(trading_yaml_data, dict): trading_yaml_data = {}
        
        allowed_pairs_final = trading_yaml_data.get('allowed_pairs') 
        if values.get('TRADING_ALLOWED_PAIRS') is not None: 
            env_pairs_str = values['TRADING_ALLOWED_PAIRS']
            if isinstance(env_pairs_str, str):
                try:
                    # Attempt to parse as JSON list first
                    parsed_env_pairs = json.loads(env_pairs_str)
                    if isinstance(parsed_env_pairs, list):
                        allowed_pairs_final = parsed_env_pairs
                    else: # If not a list, treat as comma-separated string
                        allowed_pairs_final = [p.strip().upper() for p in env_pairs_str.split(',') if p.strip()]
                except json.JSONDecodeError: # If not JSON, treat as comma-separated string
                    allowed_pairs_final = [p.strip().upper() for p in env_pairs_str.split(',') if p.strip()]
            elif isinstance(env_pairs_str, list): # Already a list (e.g. from default_factory if env var not set)
                 allowed_pairs_final = env_pairs_str

        elif not allowed_pairs_final: 
              allowed_pairs_final = TradingConfig.model_fields['allowed_pairs'].default_factory()

        values['trading'] = {
            "mode": values.get('TRADING_MODE', trading_yaml_data.get('mode', TradingConfig.model_fields['mode'].default)),
            "base_currency": values.get('TRADING_BASE_CURRENCY', trading_yaml_data.get('base_currency', TradingConfig.model_fields['base_currency'].default)),
            "allowed_pairs": allowed_pairs_final
        }

        # --- Monitoring Config ---
        monitoring_yaml_data = values.pop('monitoring', {})
        if not isinstance(monitoring_yaml_data, dict): monitoring_yaml_data = {}
        
        webhook_url = values.get('ALERT_WEBHOOK_URL', monitoring_yaml_data.get('alert_webhook_url'))
        if webhook_url == PLACEHOLDER_WEBHOOK_URL: 
            webhook_url = None

        values['monitoring'] = {
            "prometheus_enabled": monitoring_yaml_data.get('prometheus_enabled', MonitoringConfig.model_fields['prometheus_enabled'].default),
            "prometheus_port": monitoring_yaml_data.get('prometheus_port', MonitoringConfig.model_fields['prometheus_port'].default),
            "alert_webhook_url": webhook_url, 
            "log_level_console": monitoring_yaml_data.get('log_level_console', MonitoringConfig.model_fields['log_level_console'].default),
            "log_level_file": monitoring_yaml_data.get('log_level_file', MonitoringConfig.model_fields['log_level_file'].default),
            "log_to_file_enabled": monitoring_yaml_data.get('log_to_file_enabled', MonitoringConfig.model_fields['log_to_file_enabled'].default),
            "log_rotation": monitoring_yaml_data.get('log_rotation', MonitoringConfig.model_fields['log_rotation'].default),
            "log_retention": monitoring_yaml_data.get('log_retention', MonitoringConfig.model_fields['log_retention'].default),
            "log_compression": monitoring_yaml_data.get('log_compression', MonitoringConfig.model_fields['log_compression'].default),
        }

        # --- System Config (NEW) ---
        system_yaml_data = values.pop('system', {}) # Pop 'system' section from YAML if present
        if not isinstance(system_yaml_data, dict): system_yaml_data = {}
        values['system'] = {
            "thread_pool_workers": values.get( # Check env var first (e.g., SYSTEM_THREAD_POOL_WORKERS)
                'SYSTEM_THREAD_POOL_WORKERS', 
                system_yaml_data.get(
                    'thread_pool_workers', 
                    SystemConfig.model_fields['thread_pool_workers'].default
                )
            ),
            # Add other system fields here if they can be set by env vars
        }
        return values

    @model_validator(mode='after')
    @classmethod
    def perform_cross_model_validation(cls, model) -> 'Settings':
        binance_cfg = model.binance
        if not binance_cfg or not isinstance(binance_cfg, BinanceConfigModel):
            raise MissingConfigurationError(message="Binance configuration is required.", item="binance")
        if not binance_cfg.api_key or not binance_cfg.api_secret:
              raise MissingConfigurationError(message="Binance api_key and api_secret are required.", item="binance.api_key/api_secret")

        db_config = model.database
        if not db_config or not isinstance(db_config, DatabaseConfigModel):
              raise MissingConfigurationError(message="Database configuration is required.", item="database")
        if not db_config.url and db_config.db_type != "sqlite" and not (db_config.sqlite_path and db_config.sqlite_path == ":memory:"):
            raise MissingConfigurationError(message=f"{db_config.db_type} URL is missing and could not be constructed.", item="database.url")

        trading_config = model.trading
        if not trading_config or not isinstance(trading_config, TradingConfig):
            raise MissingConfigurationError(message="Trading configuration section is missing or invalid.", item="trading")
        if not trading_config.allowed_pairs:
            raise InvalidConfigurationValueError(message="Trading configuration must include at least one pair in 'allowed_pairs'.", parameter="trading.allowed_pairs")

        data_config = model.data
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
        
        system_config = model.system
        if not system_config or not isinstance(system_config, SystemConfig):
            raise MissingConfigurationError(message="System configuration section is missing or invalid.", item="system")

        return model

def load_raw_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {} 
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        if raw_config is None:
              return {} 
        if not isinstance(raw_config, dict):
            raise InvalidConfigurationValueError(
                message=f"Configuration file {config_path} content is not a valid YAML dictionary.",
                parameter="config_file_format"
            )
        return raw_config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}", original_exception=e)
    except InvalidConfigurationValueError: 
        raise
    except Exception as e:
        raise ConfigurationError(f"Could not load configuration from {config_path}: {e}", original_exception=e)

def load_settings() -> Settings:
    try:
        yaml_data = load_raw_config_from_yaml(CONFIG_FILE_PATH)
        settings_instance = Settings(**yaml_data)
        return settings_instance
    except ValidationError as e:
        error_details = []
        for error in e.errors(include_url=False, include_input=False):
            loc_str = " -> ".join(map(str, error['loc']))
            msg = error['msg']
            error_details.append(f"   - Location: '{loc_str}', Message: '{msg}'")
        error_messages_str = "\n".join(error_details)
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages_str}", original_exception=e)
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"An unexpected error occurred while loading settings: {type(e).__name__} - {e}", original_exception=e)

# Singleton pattern for settings
_settings_instance = None

def get_settings() -> Settings:
    """
    Get the settings instance (singleton pattern).
    If the settings haven't been loaded yet, load them.
    
    Returns:
        Settings: The application settings
    """
    global _settings_instance
    if _settings_instance is None:
        try:
            _settings_instance = load_settings()
            from src.core.logging_config import setup_logging
            setup_logging(_settings_instance)
            logger.info(f"Configuration loaded successfully. Environment: {_settings_instance.app.environment}, App Version: {_settings_instance.app.version}")
            if _settings_instance.monitoring and _settings_instance.monitoring.log_level_console == "DEBUG":
                logger.debug(f"Full settings loaded (sensitive fields may be masked):\n{_settings_instance.model_dump_json(indent=2)}")
        except ConfigurationError as e:
            print(f"CRITICAL CONFIGURATION ERROR during settings load: {e}")
            raise
        except Exception as e:
            print(f"CRITICAL UNHANDLED ERROR during settings load or logging setup: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise
    return _settings_instance

# For backward compatibility during transition
# This will be deprecated soon - import get_settings() instead
settings = get_settings()
