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
    ValidationError
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Détermine la racine du projet en remontant depuis l'emplacement de ce fichier
# src/core/config.py -> src/core -> src -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

# Charger les variables d'environnement du fichier .env s'il existe
# load_dotenv(find_dotenv(usecwd=True)) a été mentionné, mais pydantic-settings le gère.
# Cependant, pour la substitution manuelle dans YAML, chargeons-le explicitement.
if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)
else:
    # Fallback pour les environnements où .env peut ne pas être à la racine attendue
    # ou pour charger les variables d'environnement système si .env est manquant.
    load_dotenv()


# --- Exceptions Personnalisées (sera défini dans exceptions.py, pour l'instant placeholder) ---
class ConfigurationError(Exception):
    """Erreur de base pour les problèmes de configuration."""
    pass

class InvalidConfigurationValueError(ConfigurationError):
    """Une valeur spécifique dans la configuration est invalide."""
    pass

class MissingConfigurationError(ConfigurationError):
    """Un fichier ou une section de configuration est manquant."""
    pass


# --- Modèles Pydantic pour la configuration ---

class AppConfig(BaseModel):
    name: str = "Algo Trading Bot"
    version: str = "1.0.0"
    environment: Literal['development', 'production', 'test'] = "development"

class BinanceConfig(BaseModel):
    api_key: SecretStr
    api_secret: SecretStr
    testnet: bool = False
    # Pour la deuxième paire de clés API, si nécessaire
    api_key_2: Optional[SecretStr] = None
    api_secret_2: Optional[SecretStr] = None

class DatabaseConfig(BaseModel):
    url: Optional[str] = None # Exemple: "postgresql+psycopg2://user:pass@host:port/db"

    @validator('url')
    def check_db_url_format(cls, v):
        if v and not (v.startswith("postgresql+psycopg2://") or v.startswith("sqlite:///")):
            raise ValueError("Database URL format is invalid. Expected 'postgresql+psycopg2://...' or 'sqlite:///...'")
        return v

class RedisConfig(BaseModel):
    host: Optional[str] = "localhost"
    port: Optional[int] = 6379
    password: Optional[SecretStr] = None # Ajouté pour correspondre au .env.example

class DataConfig(BaseModel):
    storage_type: Literal['parquet', 'postgres'] = "parquet"
    storage_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "historical")
    parquet_partition_cols: List[str] = ['pair', 'year', 'month']
    cache_enabled: bool = True
    cache_ttl: int = 3600  # en secondes

    parquet_export_enabled: bool = False # Pour l'exemple de validateur
    parquet_export_path: Optional[Path] = None

    @validator('storage_path')
    def ensure_storage_path_is_absolute(cls, v: Path) -> Path:
        if not v.is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return v.resolve()

    @validator('parquet_export_path', always=True)
    def check_export_path_if_enabled(cls, v: Optional[Path], values: Dict[str, Any]) -> Optional[Path]:
        if values.get('parquet_export_enabled') and v is None:
            raise InvalidConfigurationValueError(
                "parquet_export_path must be set if parquet_export_enabled is True"
            )
        if v and not v.is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return v.resolve() if v else None


class TradingConfig(BaseModel):
    mode: Literal['cross_margin'] = "cross_margin"
    base_currency: Literal['USDC', 'USDT', 'BUSD', 'BTC', 'ETH'] = "USDC" # Étendu pour plus de flexibilité
    allowed_pairs: List[str]

    @validator('allowed_pairs', each_item=True)
    def check_pair_format(cls, v: str) -> str:
        if not v.isupper() or not v.isalnum() or len(v) < 6: # Simple validation
            raise InvalidConfigurationValueError(
                f"Trading pair '{v}' format is invalid. Expected format like 'BTCUSDC'."
            )
        return v

class RiskConfig(BaseModel):
    max_position_pct: float = Field(0.1, gt=0, le=1, description="Maximum percentage of capital per position")
    max_drawdown_pct: float = Field(0.15, gt=0, le=1, description="Maximum overall drawdown percentage for the account")
    daily_loss_limit_pct: float = Field(0.05, gt=0, le=1, description="Maximum daily loss percentage")

class MonitoringConfig(BaseModel):
    prometheus_enabled: bool = False
    prometheus_port: int = Field(9090, gt=1023, lt=65536)
    alert_webhook_url: Optional[HttpUrl] = None

    # Champs pour la configuration du logging (INF-007)
    log_level_console: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "INFO"
    log_level_file: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = "DEBUG"
    log_to_file_enabled: bool = True
    # log_file_path_template: str = "logs/app_{time:YYYY-MM-DD}.log" # Géré par Loguru directement
    log_rotation: str = "1 day" # ex: "500 MB", "1 week", "00:00"
    log_retention: str = "7 days" # ex: "1 month"
    log_compression: Optional[Literal["gz", "bz2", "zip", "xz", "lzma", "tar", "tar.gz", "tar.bz2", "tar.xz"]] = "zip"


# Modèle principal des paramètres, utilisant pydantic-settings pour charger depuis .env
class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    binance: BinanceConfig
    database: Optional[DatabaseConfig] = Field(default_factory=DatabaseConfig)
    redis: Optional[RedisConfig] = Field(default_factory=RedisConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None, # Chemin vers le fichier .env
        env_nested_delimiter='__',  # Pour les variables d'environnement imbriquées, ex: BINANCE__API_KEY
        env_prefix='', # Pas de préfixe global pour les variables d'environnement
        extra='ignore' # Ignorer les variables d'environnement supplémentaires
    )

    @root_validator(pre=False, skip_on_failure=True) # post validation
    def check_storage_config_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        data_config: Optional[DataConfig] = values.get('data')
        db_config: Optional[DatabaseConfig] = values.get('database')

        if data_config and data_config.storage_type == 'postgres':
            if not db_config or not db_config.url:
                raise InvalidConfigurationValueError(
                    "DatabaseConfig with a valid URL must be provided if data.storage_type is 'postgres'."
                )
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def ensure_binance_keys_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        binance_config: Optional[BinanceConfig] = values.get('binance')
        if not binance_config:
             # This case should ideally be caught by Pydantic if BinanceConfig is not Optional
            raise MissingConfigurationError("Binance configuration section is missing.")
        if not binance_config.api_key or not binance_config.api_secret:
            # pydantic-settings would raise an error if these are not found in env or .env
            # This validator is more of a double-check or for specific logic if they were optional.
            # Since they are mandatory in BinanceConfig, Pydantic handles this.
            pass # Pydantic's BaseSettings handles mandatory fields from env.
        return values


# --- Logique de substitution des variables d'environnement dans YAML ---
def _substitute_env_vars_in_yaml_data(data: Any) -> Any:
    """
    Parcourt récursivement une structure de données (dict/list/value) chargée depuis YAML
    et substitue les placeholders comme ${VAR_NAME} ou $VAR_NAME par la variable d'environnement correspondante.
    """
    if isinstance(data, dict):
        return {k: _substitute_env_vars_in_yaml_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars_in_yaml_data(item) for item in data]
    elif isinstance(data, str):
        # Gère ${VAR_NAME} et $VAR_NAME
        # Pour éviter les erreurs avec des patterns comme '$$.$$', on cible des variables plus spécifiques
        import re
        # Cherche ${...}
        data = re.sub(r'\$\{(\w+)\}', lambda m: os.getenv(m.group(1), m.group(0)), data)
        # Cherche $VARNAME (plus simple, mais peut avoir des faux positifs)
        # Pour être plus précis, on peut chercher $ suivi par des caractères de variable valides.
        # data = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', lambda m: os.getenv(m.group(1), m.group(0)), data)
        return data
    return data

# --- Fonction de chargement de la configuration ---
def load_raw_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Charge la configuration brute depuis le fichier YAML.
    """
    if not config_path.exists():
        raise MissingConfigurationError(f"Configuration file not found at {config_path}")
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        if not isinstance(raw_config, dict):
            raise InvalidConfigurationValueError(f"Configuration file {config_path} is not a valid YAML dictionary.")
        return raw_config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Could not load configuration from {config_path}: {e}")


def load_settings() -> Settings:
    """
    Charge la configuration depuis le fichier YAML, substitue les variables d'environnement
    dans les valeurs YAML, puis utilise Pydantic Settings pour fusionner avec les variables
    d'environnement (chargées depuis .env et l'environnement système).
    """
    try:
        # 1. Charger la configuration brute depuis YAML
        yaml_config_data = load_raw_config_from_yaml(CONFIG_FILE_PATH)

        # 2. Substituer les variables d'environnement dans les données YAML
        #    Ceci est pour les placeholders comme ${VAR} DANS le fichier YAML.
        #    Les variables d'environnement directes (ex: BINANCE__API_KEY) sont gérées par Pydantic-Settings.
        substituted_yaml_config = _substitute_env_vars_in_yaml_data(yaml_config_data)

        # 3. Créer l'instance Settings.
        #    Pydantic-Settings va:
        #    a. Charger les valeurs depuis le fichier .env (si défini dans model_config et existant).
        #    b. Charger les variables d'environnement système (ex: BINANCE__API_KEY).
        #    c. Utiliser les valeurs de `substituted_yaml_config` comme valeurs initiales.
        #       Les variables d'environnement (de .env ou système) surchargeront les valeurs de YAML
        #       si les noms de champs correspondent (ou via des alias/préfixes).
        #       Pour que YAML soit la base et que les env vars surchargent,
        #       on passe les données YAML lors de l'initialisation.
        settings_instance = Settings(**substituted_yaml_config)
        return settings_instance

    except ValidationError as e:
        error_messages = "\n".join([f"  - {err['loc']}: {err['msg']} (input_value: {err.get('input')})" for err in e.errors()])
        raise ConfigurationError(f"Configuration validation failed:\n{error_messages}")
    except ConfigurationError: # Re-raise nos propres erreurs de configuration
        raise
    except Exception as e: # Attraper d'autres erreurs potentielles
        raise ConfigurationError(f"An unexpected error occurred while loading settings: {e}")


# Instance globale des paramètres
# Cette instance sera importée par les autres modules.
try:
    settings: Settings = load_settings()
except ConfigurationError as e:
    # Gérer l'erreur de configuration au démarrage, par exemple, logger et quitter.
    # Pour l'instant, on la propage pour que le problème soit visible.
    print(f"CRITICAL CONFIGURATION ERROR: {e}") # Utiliser un logger ici une fois configuré
    # sys.exit(1) # Décommenter pour quitter l'application en cas d'erreur de config
    # Pour permettre l'import même en cas d'erreur (pour tests ou inspection), on peut assigner None
    # settings = None # Ou une config par défaut minimale si cela a du sens
    raise # Propage l'erreur pour arrêter l'application si la config est critique


# --- Exemples d'utilisation (pour référence) ---
if __name__ == "__main__":
    # Ceci ne s'exécutera que si le fichier est lancé directement.
    # Dans l'application, on importera `settings` directement.
    try:
        # Recharger pour l'exemple, normalement `settings` est déjà chargé ci-dessus.
        current_settings = load_settings()
        print("Configuration loaded successfully!")
        print(f"App Name: {current_settings.app.name}")
        print(f"App Environment: {current_settings.app.environment}")
        
        # Accès aux secrets (nécessite .get_secret_value())
        print(f"Binance API Key (type): {type(current_settings.binance.api_key)}")
        # print(f"Binance API Key (value): {current_settings.binance.api_key.get_secret_value()}") # Décommenter avec de vraies clés

        if current_settings.database and current_settings.database.url:
            print(f"Database URL: {current_settings.database.url}")
        else:
            print("Database URL: Not configured")

        if current_settings.redis and current_settings.redis.host:
             print(f"Redis Host: {current_settings.redis.host}:{current_settings.redis.port}")
        else:
            print("Redis: Not configured")

        print(f"Data Storage Type: {current_settings.data.storage_type}")
        print(f"Data Storage Path: {current_settings.data.storage_path}")
        print(f"Allowed Trading Pairs: {current_settings.trading.allowed_pairs}")
        print(f"Max Position Pct: {current_settings.risk.max_position_pct}")

        print(f"Log Level Console: {current_settings.monitoring.log_level_console}")
        print(f"Log to File Enabled: {current_settings.monitoring.log_to_file_enabled}")

    except ConfigurationError as e:
        print(f"Error loading configuration for example: {e}")
    except Exception as e:
        print(f"An unexpected error in example: {e}")

