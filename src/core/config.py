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

from src.core.exceptions import (
    ConfigurationError,
    InvalidConfigurationValueError,
    MissingConfigurationError
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

# --- Modèles Pydantic pour la configuration (deviennent des BaseModel simples) ---

class AppConfig(BaseModel):
    name: str = "Algo Trading Bot"
    version: str = "1.0.0"
    environment: Literal['development', 'production', 'test'] = "development"

class BinanceConfigModel(BaseModel): 
    api_key: SecretStr
    api_secret: SecretStr
    testnet: bool = False
    api_key_2: Optional[SecretStr] = None
    api_secret_2: Optional[SecretStr] = None

class DatabaseConfigModel(BaseModel): 
    db_type: Literal["postgresql", "sqlite"] = "postgresql"
    url: Optional[Union[PostgresDsn, str]] = None 

    pg_user: Optional[str] = None
    pg_password: Optional[SecretStr] = None
    pg_host: Optional[str] = "localhost"
    pg_port: Optional[Union[int, str]] = 5432
    pg_db: Optional[str] = "algobot_db"
    sqlite_path: Optional[str] = None

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout_seconds: int = 30
    pool_recycle_seconds: int = 3600
    pool_pre_ping_enabled: bool = True
    connect_max_retries: int = 5
    connect_retry_min_wait_seconds: int = 1
    connect_retry_max_wait_seconds: int = 10

    @root_validator(pre=True)
    def assemble_db_url_if_not_provided(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("url"):
            return values 

        db_type = values.get("db_type", "postgresql")

        if db_type == "postgresql":
            pg_user = values.get("pg_user")
            pg_password_secret = values.get("pg_password")
            pg_password = pg_password_secret.get_secret_value() if isinstance(pg_password_secret, SecretStr) else pg_password_secret
            pg_host = values.get("pg_host", "localhost")
            pg_port_val = values.get("pg_port", 5432)
            pg_port = str(pg_port_val) if pg_port_val is not None else "5432"
            pg_db = values.get("pg_db", "algobot_db")

            if all(val is not None for val in [pg_user, pg_password, pg_host, pg_db]):
                values["url"] = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        
        elif db_type == "sqlite":
            sqlite_path = values.get("sqlite_path")
            if sqlite_path:
                if sqlite_path == ":memory:":
                    values["url"] = "sqlite:///:memory:"
                else:
                    path_obj = Path(sqlite_path)
                    if not path_obj.is_absolute():
                        path_obj = PROJECT_ROOT / path_obj
                    values["url"] = f"sqlite:///{path_obj.resolve()}"
        return values

    @validator('url', always=True)
    def check_db_url_final_state(cls, v: Optional[Union[PostgresDsn, str]], values: Dict[str, Any]) -> Optional[Union[PostgresDsn, str]]:
        db_type = values.get('db_type')
        # Si l'URL est None mais qu'elle est requise (PostgreSQL ou SQLite avec chemin), lever une erreur.
        if not v and (db_type == 'postgresql' or (db_type == 'sqlite' and values.get('sqlite_path'))):
             raise ValueError(f"{db_type.capitalize()} database URL is required but could not be constructed and was not provided.")

        if v: 
            if db_type == 'postgresql':
                # La validation PostgresDsn est faite par Pydantic si le type est Union[PostgresDsn, str]
                # On vérifie juste le préfixe si c'est une chaîne.
                if isinstance(v, str) and not v.startswith("postgresql+psycopg2://"):
                    raise ValueError("Invalid PostgreSQL URL format. Expected 'postgresql+psycopg2://...'")
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
    cache_ttl: int = 3600
    parquet_export_enabled: bool = False
    parquet_export_path: Optional[Path] = None
    
    @validator('storage_path', pre=True, always=True)
    def ensure_storage_path_is_absolute(cls, v: Union[str, Path]) -> Path:
        if isinstance(v, str): v = Path(v)
        if not v.is_absolute(): return (PROJECT_ROOT / v).resolve()
        return v.resolve()

    @validator('parquet_export_path', pre=True, always=True)
    def check_export_path_if_enabled(cls, v: Optional[Union[str, Path]], values: Dict[str, Any]) -> Optional[Path]:
        if values.get('parquet_export_enabled') and v is None:
            raise InvalidConfigurationValueError("parquet_export_path must be set if parquet_export_enabled is True")
        if v:
            if isinstance(v, str): v = Path(v)
            if not v.is_absolute(): return (PROJECT_ROOT / v).resolve()
            return v.resolve() if v else None
        return None

class TradingConfig(BaseModel): 
    mode: Literal['cross_margin'] = "cross_margin"
    base_currency: Literal['USDC', 'USDT', 'BUSD', 'BTC', 'ETH'] = "USDC"
    allowed_pairs: List[str]

    @validator('allowed_pairs', each_item=True)
    def check_pair_format(cls, v: str) -> str:
        if not v.isupper() or not v.isalnum() or len(v) < 6:
            raise InvalidConfigurationValueError(f"Trading pair '{v}' format is invalid. Expected format like 'BTCUSDC'.")
        return v

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


class Settings(BaseSettings):
    # Variables d'environnement directes que pydantic-settings va charger.
    # Ces noms DOIVENT correspondre à vos variables dans .env (ou utiliser des alias ici).
    BINANCE_API_KEY: Optional[SecretStr] = None # Rendu optionnel ici, la validation se fera dans BinanceConfigModel
    BINANCE_API_SECRET: Optional[SecretStr] = None
    BINANCE_TESTNET: Optional[bool] = False # Optionnel avec défaut
    BINANCE_API_KEY_2: Optional[SecretStr] = None
    BINANCE_API_SECRET_2: Optional[SecretStr] = None

    DB_TYPE: Optional[Literal["postgresql", "sqlite"]] = "postgresql"
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[SecretStr] = None
    POSTGRES_HOST: Optional[str] = "localhost"
    POSTGRES_PORT: Optional[Union[int, str]] = 5432
    POSTGRES_DB: Optional[str] = "algobot_db"
    SQLITE_PATH: Optional[str] = None
    DATABASE_URL: Optional[Union[PostgresDsn, str]] = None 

    DB_POOL_SIZE: Optional[int] = 5
    DB_MAX_OVERFLOW: Optional[int] = 10
    DB_POOL_TIMEOUT_SECONDS: Optional[int] = 30
    DB_POOL_RECYCLE_SECONDS: Optional[int] = 3600
    DB_POOL_PRE_PING_ENABLED: Optional[bool] = True
    DB_CONNECT_MAX_RETRIES: Optional[int] = 5
    DB_CONNECT_RETRY_MIN_WAIT_SECONDS: Optional[int] = 1
    DB_CONNECT_RETRY_MAX_WAIT_SECONDS: Optional[int] = 10

    REDIS_HOST: Optional[str] = "localhost"
    REDIS_PORT: Optional[int] = 6379
    REDIS_PASSWORD: Optional[SecretStr] = None
    
    ALERT_WEBHOOK_URL: Optional[HttpUrl] = None # Directement HttpUrl, Pydantic essaiera de parser


    # Sections qui seront principalement peuplées depuis le fichier YAML.
    # Leurs valeurs par défaut sont utilisées si la section/clé est absente du YAML.
    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig # Doit être dans YAML
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Modèles imbriqués qui seront construits par le root_validator.
    # Ils sont déclarés ici pour que Pydantic connaisse leur type.
    binance: BinanceConfigModel
    database: DatabaseConfigModel
    redis: RedisConfigModel 

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_prefix='', # Les noms de champs ci-dessus sont les noms exacts des variables d'env
        extra='ignore',
        populate_by_name=True # Utile si on utilisait des alias sur les champs directs de Settings
    )

    @root_validator(pre=True) # S'exécute AVANT la validation des champs individuels de Settings
    def prepare_nested_model_input_data(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les dictionnaires d'entrée pour les modèles imbriqués (binance, database, redis)
        en utilisant les variables d'environnement (déjà chargées dans `values` par pydantic-settings
        pour les champs directs de Settings) et les sections correspondantes du fichier YAML (également dans `values`).
        """
        # Pour Binance:
        # `values` contient déjà BINANCE_API_KEY, etc., depuis .env
        # et potentiellement une section `binance` depuis YAML.
        binance_yaml_data = values.pop('binance', {}) # Récupère et retire la section YAML 'binance' de `values`
        if not isinstance(binance_yaml_data, dict): binance_yaml_data = {}
        
        values['binance_input_data'] = { # Créer une nouvelle clé pour les données d'entrée de BinanceConfigModel
            "api_key": values.get('BINANCE_API_KEY', binance_yaml_data.get('api_key')),
            "api_secret": values.get('BINANCE_API_SECRET', binance_yaml_data.get('api_secret')),
            "testnet": values.get('BINANCE_TESTNET', binance_yaml_data.get('testnet', False)),
            "api_key_2": values.get('BINANCE_API_KEY_2', binance_yaml_data.get('api_key_2')),
            "api_secret_2": values.get('BINANCE_API_SECRET_2', binance_yaml_data.get('api_secret_2')),
        }

        # Pour Database:
        database_yaml_data = values.pop('database', {})
        if not isinstance(database_yaml_data, dict): database_yaml_data = {}
        values['database_input_data'] = {
            "db_type": values.get('DB_TYPE', database_yaml_data.get('db_type', "postgresql")),
            "pg_user": values.get('POSTGRES_USER', database_yaml_data.get('pg_user')),
            "pg_password": values.get('POSTGRES_PASSWORD', database_yaml_data.get('pg_password')),
            "pg_host": values.get('POSTGRES_HOST', database_yaml_data.get('pg_host', "localhost")),
            "pg_port": values.get('POSTGRES_PORT', database_yaml_data.get('pg_port', 5432)),
            "pg_db": values.get('POSTGRES_DB', database_yaml_data.get('pg_db', "algobot_db")),
            "sqlite_path": values.get('SQLITE_PATH', database_yaml_data.get('sqlite_path')),
            "url": values.get('DATABASE_URL', database_yaml_data.get('url')),
            "pool_size": values.get('DB_POOL_SIZE', database_yaml_data.get('pool_size', 5)),
            "max_overflow": values.get('DB_MAX_OVERFLOW', database_yaml_data.get('max_overflow', 10)),
            # ... inclure tous les champs de DatabaseConfigModel ...
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
        values['redis_input_data'] = {
            "host": values.get('REDIS_HOST', redis_yaml_data.get('host', "localhost")),
            "port": values.get('REDIS_PORT', redis_yaml_data.get('port', 6379)),
            "password": values.get('REDIS_PASSWORD', redis_yaml_data.get('password')),
        }

        # Pour Monitoring (ALERT_WEBHOOK_URL est déjà un champ direct de Settings)
        monitoring_yaml_data = values.get('monitoring', {}) # Ne pas pop, car 'monitoring' est un champ de Settings
        if isinstance(monitoring_yaml_data, dict) and values.get('ALERT_WEBHOOK_URL'):
            monitoring_yaml_data['alert_webhook_url'] = values.get('ALERT_WEBHOOK_URL')
        values['monitoring'] = monitoring_yaml_data # Mettre à jour la section monitoring dans values

        return values

    @root_validator(pre=False, skip_on_failure=True) # S'exécute APRES la validation des champs individuels
    def construct_final_nested_models(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Utiliser les dictionnaires préparés pour instancier les modèles imbriqués
        if 'binance_input_data' in values:
            values['binance'] = BinanceConfigModel(**values.pop('binance_input_data'))
        else: # Devrait être impossible si BINANCE_API_KEY est requis par Settings
            raise MissingConfigurationError("Binance input data not prepared.")

        if 'database_input_data' in values:
            values['database'] = DatabaseConfigModel(**values.pop('database_input_data'))
        else:
            raise MissingConfigurationError("Database input data not prepared.")

        if 'redis_input_data' in values:
            values['redis'] = RedisConfigModel(**values.pop('redis_input_data'))
        else: # Si redis est optionnel dans YAML et pas d'env vars, on pourrait créer un RedisConfigModel par défaut
            values['redis'] = RedisConfigModel() # Ou le rendre optionnel sur Settings

        # 'monitoring' est déjà un MonitoringConfig grâce à Field(default_factory=MonitoringConfig)
        # et mis à jour par prepare_nested_model_input_data.
        # On pourrait revalider ici si nécessaire.
        # values['monitoring'] = MonitoringConfig(**values.get('monitoring', {}))

        # Valider la cohérence finale
        data_config: Optional[DataConfig] = values.get('data')
        db_config: Optional[DatabaseConfigModel] = values.get('database')

        if data_config and db_config and data_config.storage_type == 'postgres':
            if db_config.db_type != 'postgresql' or not db_config.url:
                raise InvalidConfigurationValueError(
                    "If data.storage_type is 'postgres', database.db_type must be 'postgresql' "
                    "and a valid PostgreSQL URL must be configured or constructible."
                )
            if isinstance(db_config.url, str) and not db_config.url.startswith("postgresql"):
                 raise InvalidConfigurationValueError("URL for postgres storage_type must be a PostgreSQL DSN.")
        
        if not values.get('trading'): 
            raise MissingConfigurationError("Trading configuration section is missing in YAML.")
        return values


def _substitute_env_vars_in_yaml_data(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _substitute_env_vars_in_yaml_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars_in_yaml_data(item) for item in data]
    elif isinstance(data, str):
        import re
        data = re.sub(r'\$\{(\w+)\}', lambda m: os.getenv(m.group(1), f'${{{m.group(1)}}}'), data)
        return data
    return data

def load_raw_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {} 
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        if raw_config is None: 
             return {} 
        if not isinstance(raw_config, dict):
            raise InvalidConfigurationValueError(f"Configuration file {config_path} is not a valid YAML dictionary.")
        return raw_config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Could not load configuration from {config_path}: {e}")

def load_settings() -> Settings:
    try:
        yaml_data = load_raw_config_from_yaml(CONFIG_FILE_PATH)
        substituted_yaml_data = _substitute_env_vars_in_yaml_data(yaml_data)
        
        # Pydantic-Settings chargera d'abord les variables d'env directes dans Settings,
        # puis fusionnera avec substituted_yaml_data.
        # Ensuite, les root_validators s'exécuteront.
        settings_instance = Settings(**substituted_yaml_data)
        return settings_instance

    except ValidationError as e:
        error_details = []
        for error in e.errors():
            loc_str = " -> ".join(map(str, error['loc']))
            msg = error['msg']
            inp = error.get('input', 'N/A')
            error_details.append(f"  - Location: '{loc_str}', Message: '{msg}', Input: {inp}")
        error_messages = "\n".join(error_details)
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages}")
    except ConfigurationError: 
        raise
    except Exception as e: 
        raise ConfigurationError(f"An unexpected error occurred while loading settings: {type(e).__name__} - {e}")

try:
    settings: Settings = load_settings()
except ConfigurationError as e:
    print(f"CRITICAL CONFIGURATION ERROR: {e}")
    raise 
