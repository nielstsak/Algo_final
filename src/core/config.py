import os
import yaml
import json
from pathlib import Path
from typing import List, Literal, Optional, Any, Dict, Union
from functools import lru_cache

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    HttpUrl,
    ValidationError,
    PostgresDsn,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Importation de loguru pour le logging
from loguru import logger

from src.core.exceptions import (
    ConfigurationError,
    InvalidConfigurationValueError,
    MissingConfigurationError,
)
# Note: L'import de 'normalize_pair_symbol' n'est pas utilisé, mais est conservé
# au cas où il serait nécessaire pour des évolutions futures.
from src.utils.exchange_utils import normalize_pair_list, normalize_pair_symbol

# --- Constantes de Configuration ---
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
    api_key: Optional[SecretStr] = None
    api_secret: Optional[SecretStr] = None
    testnet: bool = False
    api_key_2: Optional[SecretStr] = None
    api_secret_2: Optional[SecretStr] = None

class DatabaseConfigModel(BaseModel):
    """Database connection and pooling settings."""
    db_type: Literal["postgresql", "sqlite"] = "postgresql"
    url: Optional[Union[PostgresDsn, str]] = None

    pg_user: Optional[str] = "algobot_user"
    pg_password: Optional[SecretStr] = "your_secure_password"
    pg_host: Optional[str] = "localhost"
    pg_port: Optional[Union[int, str]] = 5432
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

class TradingConfig(BaseModel):
    mode: Literal['cross_margin'] = "cross_margin"
    base_currency: Literal['USDC', 'USDT', 'BUSD', 'BTC', 'ETH'] = "USDC"
    allowed_pairs: List[str] = Field(default_factory=list)

    @field_validator('allowed_pairs', mode='before')
    @classmethod
    def normalize_and_validate_allowed_pairs(cls, v: Any) -> List[str]:
        if isinstance(v, str):
             v = [p.strip() for p in v.split(',') if p.strip()]
        if not isinstance(v, list):
            raise InvalidConfigurationValueError(
                message="allowed_pairs must be a list.",
                parameter="allowed_pairs", value=v
            )
        return normalize_pair_list(v)

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

    @field_validator('alert_webhook_url', mode='before')
    @classmethod
    def handle_placeholder_webhook_url(cls, v: Any) -> Optional[Any]:
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

class SystemConfig(BaseModel):
    thread_pool_workers: int = Field(default=4, gt=0, description="Number of workers for thread pools.")

class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    binance: BinanceConfigModel
    database: DatabaseConfigModel
    redis: RedisConfigModel = Field(default_factory=RedisConfigModel)
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_prefix='',
        extra='ignore',
        populate_by_name=True,
    )

    @model_validator(mode='before')
    @classmethod
    def _populate_nested_models(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Manually construct nested models from a flat dict (from YAML) and env vars."""
        
        def _get_val(env_key, yaml_key, yaml_data, default=None):
            return values.get(env_key, yaml_data.get(yaml_key, default))

        model_fields = cls.model_fields

        for field_name, field_info in model_fields.items():
            if hasattr(field_info.annotation, '__mro__') and BaseModel in field_info.annotation.__mro__:
                nested_model_cls = field_info.annotation
                yaml_data = values.get(field_name, {})
                if not isinstance(yaml_data, dict):
                    yaml_data = {}

                nested_values = {}
                for sub_field_name in nested_model_cls.model_fields:
                    env_var_name = f"{field_name.upper()}_{sub_field_name.upper()}"
                    nested_values[sub_field_name] = _get_val(
                        env_var_name,
                        sub_field_name,
                        yaml_data,
                        nested_model_cls.model_fields[sub_field_name].default
                    )
                values[field_name] = nested_values

        return values


def load_raw_config_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    # MODIFIÉ : Assurer que config_path est un objet Path
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
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
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Could not load configuration from {config_path}: {e}") from e

# --- Alias pour la rétrocompatibilité ---
load_config = load_raw_config_from_yaml
load_strategies_config = load_raw_config_from_yaml


def load_settings() -> Settings:
    """Loads settings from YAML and environment variables, then validates."""
    try:
        yaml_data = load_raw_config_from_yaml(CONFIG_FILE_PATH)
        settings_instance = Settings(**yaml_data)
        return settings_instance
    except ValidationError as e:
        error_details = [f"  - Location: {' -> '.join(map(str, error['loc']))}, Message: '{error['msg']}'" for error in e.errors(include_url=False, include_input=False)]
        error_messages_str = "\n".join(error_details)
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages_str}") from e
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"An unexpected error occurred while loading settings: {type(e).__name__} - {e}") from e

# Singleton pattern for settings
_settings_instance = None

def get_settings() -> Settings:
    """
    Returns the settings instance (singleton pattern).
    If the settings haven't been loaded yet, it loads them.
    This also triggers the logging setup on first load.
    """
    global _settings_instance
    if _settings_instance is None:
        try:
            _settings_instance = load_settings()
            from src.core.logging_config import setup_logging
            setup_logging(_settings_instance)
            logger.info(f"Configuration loaded. Environment: {_settings_instance.app.environment}, App Version: {_settings_instance.app.version}")
        except ConfigurationError as e:
            # Logging might not be set up, use print for critical errors
            print(f"CRITICAL CONFIGURATION ERROR: {e}")
            raise
        except Exception as e:
            print(f"CRITICAL UNHANDLED ERROR during settings load: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise
    return _settings_instance

# For backward compatibility during transition
settings = get_settings()
