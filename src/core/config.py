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
        if values.get("url"): # Si une URL complète est déjà fournie, on l'utilise
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
                        path_obj = PROJECT_ROOT / path_obj # Assurer que le chemin est absolu
                    values["url"] = f"sqlite:///{path_obj.resolve()}"
        return values

    @validator('url', always=True) # S'exécute après `assemble_db_url_if_not_provided`
    def check_db_url_final_state(cls, v: Optional[Union[PostgresDsn, str]], values: Dict[str, Any]) -> Optional[Union[PostgresDsn, str]]:
        db_type = values.get('db_type')
        # Une URL est requise si db_type est postgresql, ou si c'est sqlite avec un chemin de fichier (pas :memory:)
        is_persistent_sqlite = db_type == 'sqlite' and values.get('sqlite_path') and values.get('sqlite_path') != ":memory:"
        if not v and (db_type == 'postgresql' or is_persistent_sqlite):
             raise ValueError(f"{db_type.capitalize()} database URL is required but could not be constructed and was not provided.")

        if v: # Si une URL a été fournie ou construite
            if db_type == 'postgresql':
                # La validation PostgresDsn est gérée par Pydantic. Si c'est une chaîne, on vérifie le préfixe.
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
    cache_ttl: int = 3600 # en secondes
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
            raise InvalidConfigurationValueError(
                message="parquet_export_path must be set if parquet_export_enabled is True",
                parameter="parquet_export_path" # Ajout du paramètre pour plus de clarté
            )
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
            raise InvalidConfigurationValueError(
                message=f"Trading pair '{v}' format is invalid. Expected format like 'BTCUSDC'.",
                parameter="allowed_pairs" # Ajout du paramètre
            )
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
    log_rotation: str = "1 day" # ex: "100 MB", "1 week", "00:00"
    log_retention: str = "7 days" # ex: "1 month"
    log_compression: Optional[Literal["gz", "bz2", "zip", "xz", "lzma", "tar", "tar.gz", "tar.bz2", "tar.xz"]] = "zip"

    @validator('alert_webhook_url', pre=True)
    def handle_placeholder_webhook_url(cls, v: Any) -> Optional[Any]:
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

class Settings(BaseSettings):
    # Variables d'environnement directes pour pydantic-settings
    BINANCE_API_KEY: Optional[SecretStr] = None
    BINANCE_API_SECRET: Optional[SecretStr] = None
    BINANCE_TESTNET: Optional[bool] = False
    BINANCE_API_KEY_2: Optional[SecretStr] = None
    BINANCE_API_SECRET_2: Optional[SecretStr] = None

    DB_TYPE: Optional[Literal["postgresql", "sqlite"]] = None # Laissé optionnel, DatabaseConfigModel aura une valeur par défaut
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[SecretStr] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[Union[int, str]] = None
    POSTGRES_DB: Optional[str] = None
    SQLITE_PATH: Optional[str] = None
    DATABASE_URL: Optional[Union[PostgresDsn, str]] = None # URL complète peut être fournie

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
    
    ALERT_WEBHOOK_URL: Optional[HttpUrl] = None

    # Sections lues depuis YAML, avec des valeurs par défaut via Field(default_factory=...)
    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    # 'trading' est requis par le YAML, donc pas de default_factory. Sa présence sera validée.
    trading: Optional[TradingConfig] = None # Rendu Optional ici, validé dans perform_cross_model_validation
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Champs pour les modèles imbriqués. Pydantic essaiera de les construire
    # à partir des dictionnaires préparés par prepare_nested_model_inputs.
    binance: Optional[BinanceConfigModel] = None
    database: Optional[DatabaseConfigModel] = None
    redis: Optional[RedisConfigModel] = None

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_prefix='', # Pas de préfixe pour les variables d'env
        extra='ignore', # Ignorer les variables d'env supplémentaires
        populate_by_name=True # Pour les alias (non utilisé ici mais bonne pratique)
    )

    @validator('ALERT_WEBHOOK_URL', pre=True)
    def handle_placeholder_top_level_webhook_url(cls, v: Any) -> Optional[Any]:
        if isinstance(v, str) and v == PLACEHOLDER_WEBHOOK_URL:
            return None
        return v

    @root_validator(pre=True) # S'exécute AVANT la validation des champs individuels de Settings
    def prepare_nested_model_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les dictionnaires d'entrée pour les champs de type modèle (binance, database, redis)
        en fusionnant les données du YAML et les variables d'environnement directes (déjà dans `values`).
        Les clés résultantes ('binance', 'database', 'redis') dans `values` seront des dictionnaires
        que Pydantic utilisera ensuite pour instancier BinanceConfigModel, DatabaseConfigModel, etc.
        """
        # Pour Binance:
        binance_yaml_data = values.pop('binance', {}) # Récupère la section YAML 'binance' et la retire de `values` pour éviter conflit de type
        if not isinstance(binance_yaml_data, dict): binance_yaml_data = {}
        
        values['binance'] = { # Prépare le dict pour le champ Settings.binance
            "api_key": values.get('BINANCE_API_KEY', binance_yaml_data.get('api_key')),
            "api_secret": values.get('BINANCE_API_SECRET', binance_yaml_data.get('api_secret')),
            "testnet": values.get('BINANCE_TESTNET', binance_yaml_data.get('testnet', False)),
            "api_key_2": values.get('BINANCE_API_KEY_2', binance_yaml_data.get('api_key_2')),
            "api_secret_2": values.get('BINANCE_API_SECRET_2', binance_yaml_data.get('api_secret_2')),
        }

        # Pour Database:
        database_yaml_data = values.pop('database', {})
        if not isinstance(database_yaml_data, dict): database_yaml_data = {}
        values['database'] = { # Prépare le dict pour le champ Settings.database
            "db_type": values.get('DB_TYPE', database_yaml_data.get('db_type', "postgresql")),
            "pg_user": values.get('POSTGRES_USER', database_yaml_data.get('pg_user')),
            "pg_password": values.get('POSTGRES_PASSWORD', database_yaml_data.get('pg_password')),
            "pg_host": values.get('POSTGRES_HOST', database_yaml_data.get('pg_host', "localhost")),
            "pg_port": values.get('POSTGRES_PORT', database_yaml_data.get('pg_port', 5432)),
            "pg_db": values.get('POSTGRES_DB', database_yaml_data.get('pg_db', "algobot_db")),
            "sqlite_path": values.get('SQLITE_PATH', database_yaml_data.get('sqlite_path')),
            "url": values.get('DATABASE_URL', database_yaml_data.get('url')), # Important: .env DATABASE_URL a priorité
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
        values['redis'] = { # Prépare le dict pour le champ Settings.redis
            "host": values.get('REDIS_HOST', redis_yaml_data.get('host', "localhost")),
            "port": values.get('REDIS_PORT', redis_yaml_data.get('port', 6379)),
            "password": values.get('REDIS_PASSWORD', redis_yaml_data.get('password')),
        }

        # Pour Monitoring (ALERT_WEBHOOK_URL est un champ direct de Settings)
        # Le champ 'monitoring' de Settings sera créé par Pydantic en utilisant MonitoringConfig.
        # Nous devons nous assurer que la valeur pour 'alert_webhook_url' dans le dict 'monitoring'
        # est correctement définie en prenant en compte le ALERT_WEBHOOK_URL de haut niveau.
        monitoring_data_from_yaml = values.get('monitoring', {}) # Ne pas popper ici, 'monitoring' est un champ de Settings
        if not isinstance(monitoring_data_from_yaml, dict): monitoring_data_from_yaml = {}
        
        # Priorité: .env (via Settings.ALERT_WEBHOOK_URL) > YAML (monitoring.alert_webhook_url)
        top_level_webhook_url_value = values.get('ALERT_WEBHOOK_URL') # Déjà traité par son propre validateur (placeholder -> None)
        
        if top_level_webhook_url_value is not None:
             monitoring_data_from_yaml['alert_webhook_url'] = top_level_webhook_url
        # Si top_level_webhook_url_value est None, la valeur de YAML (si elle existe et n'est pas placeholder) sera utilisée.
        # Le validateur de MonitoringConfig gérera le placeholder s'il vient du YAML.
        
        values['monitoring'] = monitoring_data_from_yaml # Assure que 'monitoring' est un dict pour Pydantic

        # 'trading' est lu depuis YAML (pydantic-settings le place dans `values` s'il est dans yaml_data)
        # Sa validation en TradingConfig se fera automatiquement par Pydantic.
        # Idem pour app, data, risk.

        return values

    @root_validator(pre=False, skip_on_failure=True) # S'exécute APRES la validation des champs individuels
    def perform_cross_model_validation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # À ce stade, values['binance'], values['database'], values['redis'], values['trading']
        # devraient être des instances de leurs modèles respectifs (ou None si Optional et non fourni).
        
        binance_cfg = values.get('binance')
        if not isinstance(binance_cfg, BinanceConfigModel):
            # Cela signifie que la création de BinanceConfigModel a échoué ou que les données étaient manquantes.
            # Si api_key/secret sont requis dans BinanceConfigModel, Pydantic aurait déjà dû lever une erreur.
            # On peut ajouter une vérification ici si Binance est absolument requis.
            raise MissingConfigurationError(message="Binance configuration (API keys) is required and could not be loaded.", item="binance")

        db_config = values.get('database')
        if not isinstance(db_config, DatabaseConfigModel):
             raise MissingConfigurationError(message="Database configuration is required and could not be loaded.", item="database")
        if not db_config.url and (db_config.db_type == 'postgresql' or (db_config.db_type == 'sqlite' and db_config.sqlite_path != ":memory:")):
             raise InvalidConfigurationValueError(message="Database URL is still missing after attempting to construct it.", parameter="database.url")


        # Valider la présence de la section 'trading'
        trading_config = values.get('trading')
        if not isinstance(trading_config, TradingConfig):
            raise MissingConfigurationError(message="Trading configuration section ('trading:') is missing or invalid in YAML config file.", item="trading")

        # Valider la cohérence entre data.storage_type et la configuration de la base de données
        data_config: Optional[DataConfig] = values.get('data')
        # db_config est déjà récupéré

        if data_config and db_config and data_config.storage_type == 'postgres':
            if db_config.db_type != 'postgresql': # L'URL est déjà validée pour être une DSN postgresql par DatabaseConfigModel si elle existe
                raise InvalidConfigurationValueError(
                    message="If data.storage_type is 'postgres', database.db_type must be 'postgresql'.",
                    parameter="data.storage_type/database.db_type"
                )
            if not db_config.url or (isinstance(db_config.url, str) and not db_config.url.startswith("postgresql")):
                 raise InvalidConfigurationValueError(
                    message="A valid PostgreSQL URL is required when data.storage_type is 'postgres'.",
                    parameter="database.url"
                )
        return values


def _substitute_env_vars_in_yaml_data(data: Any, env_vars: Dict[str, str]) -> Any:
    # Cette fonction n'est plus activement utilisée si pydantic-settings gère bien tout,
    # mais conservée au cas où. La logique de fusion est maintenant dans le root_validator.
    if isinstance(data, dict):
        return {k: _substitute_env_vars_in_yaml_data(v, env_vars) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars_in_yaml_data(item, env_vars) for item in data]
    elif isinstance(data, str):
        import re
        # Ne pas substituer si la var d'env n'existe pas, laisser Pydantic gérer la validation du champ.
        return re.sub(r'\$\{(\w+)\}', lambda m: env_vars.get(m.group(1), m.group(0)), data)
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
            raise InvalidConfigurationValueError(
                message=f"Configuration file {config_path} content is not a valid YAML dictionary.",
                parameter="config_file_format"
            )
        return raw_config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}", original_exception=e)
    except Exception as e: # Autres erreurs de lecture de fichier
        raise ConfigurationError(f"Could not load configuration from {config_path}: {e}", original_exception=e)

def load_settings() -> Settings:
    try:
        yaml_data = load_raw_config_from_yaml(CONFIG_FILE_PATH)
        
        # Pydantic-Settings s'occupe de charger .env et de fusionner.
        # Les validateurs s'exécuteront sur les données combinées.
        settings_instance = Settings(**yaml_data)
        return settings_instance

    except ValidationError as e: # Erreur de validation Pydantic (y compris celles de nos validateurs)
        error_details = []
        # Utiliser include_input=False pour ne pas afficher les valeurs sensibles dans les logs
        for error in e.errors(include_url=False, include_input=False): 
            loc_str = " -> ".join(map(str, error['loc']))
            msg = error['msg']
            # inp_val = error.get('input') # Éviter d'afficher l'input pour les secrets
            # Pour le débogage local, on peut réactiver l'affichage de l'input
            inp_val_debug = error.get('input')
            if isinstance(inp_val_debug, dict) and len(inp_val_debug) > 5:
                inp_repr_debug = f"{{...dict with {len(inp_val_debug)} keys...}}"
            elif isinstance(inp_val_debug, SecretStr):
                inp_repr_debug = "[SECRET]"
            else:
                inp_repr_debug = repr(inp_val_debug)
            
            # error_details.append(f"  - Location: '{loc_str}', Message: '{msg}'") # Version sans input
            error_details.append(f"  - Location: '{loc_str}', Message: '{msg}', Input: {inp_repr_debug}")


        error_messages = "\n".join(error_details)
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages}", original_exception=e)
    except ConfigurationError: 
        raise 
    except Exception as e: 
        raise ConfigurationError(f"An unexpected error occurred while loading settings: {type(e).__name__} - {e}", original_exception=e)

try:
    settings: Settings = load_settings()
except ConfigurationError as e:
    print(f"CRITICAL CONFIGURATION ERROR: {e}") 
    raise