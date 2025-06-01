import pytest
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import SecretStr, HttpUrl, PostgresDsn, ValidationError

# Supposons que PROJECT_ROOT est correctement défini dans src.core.config
# Pour les tests, nous pourrions avoir besoin de le redéfinir ou de s'assurer qu'il est accessible
# Si src.core.config est importé, son PROJECT_ROOT sera utilisé.

# Pour éviter les problèmes d'import circulaire ou de module non trouvé lors des tests,
# il est parfois nécessaire d'ajuster sys.path si les tests sont exécutés d'une manière
# qui ne reconnaît pas 'src' comme un package.
# Cependant, avec une structure de projet standard et pytest, cela devrait fonctionner.

from src.core.config import (
    Settings,
    AppConfig,
    BinanceConfigModel,
    DatabaseConfigModel,
    DataConfig,
    TradingConfig,
    RiskConfig,
    MonitoringConfig,
    RedisConfigModel,
    load_settings, # Fonction principale à tester
    load_raw_config_from_yaml,
    ConfigurationError,
    InvalidConfigurationValueError,
    MissingConfigurationError,
    PLACEHOLDER_WEBHOOK_URL,
    PROJECT_ROOT, # Importer pour l'utiliser dans les tests si besoin
    CONFIG_FILE_PATH # Importer pour le patcher
)
from src.core.exceptions import ConfigurationError as SrcConfigurationError


# --- Fixtures Pytest ---

@pytest.fixture
def minimal_valid_yaml_data() -> Dict[str, Any]:
    """Fournit des données YAML minimales mais valides."""
    return {
        "app": {"environment": "test"},
        "binance": {}, # Les clés API viendront de l'env
        "database": {"db_type": "sqlite", "sqlite_path": ":memory:"},
        "data": {"storage_path": "data_test"},
        "trading": {"allowed_pairs": ["BTCUSDC", "ETHUSDC"], "base_currency": "USDC"},
        # risk et monitoring utiliseront les valeurs par défaut de Pydantic
    }

@pytest.fixture
def full_valid_yaml_data(minimal_valid_yaml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fournit des données YAML plus complètes et valides."""
    data = minimal_valid_yaml_data.copy()
    data["app"] = {
        "name": "Test Algo Bot",
        "version": "0.1.0",
        "environment": "test",
    }
    data["binance"].update({"testnet": True})
    data["database"] = {
        "db_type": "postgresql",
        # pg_user, pg_password, etc. seront pris de l'env ou DATABASE_URL
        "pg_host": "test_db_host",
        "pg_port": 5433,
        "pg_db": "test_algobot_db",
        "pool_size": 3,
    }
    data["data"].update({
        "storage_type": "parquet",
        "storage_path": "test_data_output/crypto_klines",
        "parquet_partition_cols": ["pair", "year"],
        "cache_enabled": False,
        "cache_ttl": 1800,
    })
    data["trading"].update({
        "base_currency": "USDT",
        "allowed_pairs": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    })
    data["risk"] = {
        "max_position_pct": 0.05,
        "max_drawdown_pct": 0.25,
        "daily_loss_limit_pct": 0.08,
    }
    data["monitoring"] = {
        "prometheus_enabled": True,
        "prometheus_port": 9099,
        "alert_webhook_url": "http://test.webhook.com/alert",
        "log_level_console": "DEBUG",
        "log_level_file": "INFO",
        "log_to_file_enabled": True,
        "log_rotation": "5 MB",
        "log_retention": "3 days",
        "log_compression": "gz",
    }
    data["redis"] = {"host": "test_redis_host", "port": 6380}
    return data


@pytest.fixture
def create_temp_config_file(tmp_path: Path):
    """
    Fixture pour créer un fichier config.yaml temporaire.
    Retourne une fonction qui prend un dict de données et écrit le fichier.
    """
    def _writer(data: Dict[str, Any], filename: str = "config.yaml") -> Path:
        config_file = tmp_path / filename
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        return config_file
    return _writer

# --- Tests ---

def test_load_settings_minimal_valid_yaml(
    create_temp_config_file, monkeypatch, minimal_valid_yaml_data
):
    """Teste le chargement avec un YAML minimal valide et des secrets d'environnement."""
    temp_config_path = create_temp_config_file(minimal_valid_yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Mocker les variables d'environnement requises pour Binance
    monkeypatch.setenv("BINANCE_API_KEY", "test_key_env")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret_env")

    settings = load_settings()

    assert settings.app.environment == "test"
    assert settings.binance.api_key.get_secret_value() == "test_key_env"
    assert settings.binance.api_secret.get_secret_value() == "test_secret_env"
    assert settings.binance.testnet is False # Valeur par défaut du modèle BinanceConfigModel
    assert settings.database.db_type == "sqlite"
    assert str(settings.database.url) == "sqlite:///:memory:" # Vérifier la construction de l'URL
    assert settings.data.storage_path == PROJECT_ROOT / "data_test" # Vérifier la résolution du chemin
    assert settings.trading.allowed_pairs == ["BTCUSDC", "ETHUSDC"]
    assert settings.trading.base_currency == "USDC"
    # Risk et Monitoring devraient avoir leurs valeurs par défaut du modèle Pydantic
    assert settings.risk.max_position_pct == 0.1 # Default from RiskConfig model
    assert settings.monitoring.log_level_console == "INFO" # Default from MonitoringConfig model
    assert settings.redis.host == "localhost" # Default from RedisConfigModel

def test_load_settings_full_valid_yaml_and_env_override(
    create_temp_config_file, monkeypatch, full_valid_yaml_data
):
    """Teste le chargement avec un YAML complet et des surcharges par variables d'environnement."""
    temp_config_path = create_temp_config_file(full_valid_yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Mocker les variables d'environnement qui surchargeront le YAML ou fourniront des secrets
    monkeypatch.setenv("BINANCE_API_KEY", "env_api_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "env_api_secret")
    monkeypatch.setenv("BINANCE_TESTNET", "true") # Surcharge full_valid_yaml_data["binance"]["testnet"]
    monkeypatch.setenv("POSTGRES_USER", "env_pg_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "env_pg_password")
    monkeypatch.setenv("REDIS_HOST", "env_redis_host") # Surcharge full_valid_yaml_data["redis"]["host"]
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://env.webhook.com/alert") # Surcharge le YAML

    settings = load_settings()

    # Vérifications
    assert settings.app.name == "Test Algo Bot"
    assert settings.binance.api_key.get_secret_value() == "env_api_key"
    assert settings.binance.api_secret.get_secret_value() == "env_api_secret"
    assert settings.binance.testnet is True # Doit être surchargé par l'env

    assert settings.database.db_type == "postgresql"
    assert settings.database.pg_user == "env_pg_user" # Surchargé/fourni par l'env
    assert settings.database.pg_password.get_secret_value() == "env_pg_password"
    assert settings.database.pg_host == "test_db_host" # Du YAML
    assert str(settings.database.url) == "postgresql+psycopg2://env_pg_user:env_pg_password@test_db_host:5433/test_algobot_db"

    assert settings.data.storage_type == "parquet"
    assert settings.data.storage_path == PROJECT_ROOT / "test_data_output" / "crypto_klines"
    assert settings.data.cache_enabled is False

    assert settings.trading.base_currency == "USDT"

    assert settings.risk.max_position_pct == 0.05

    assert settings.monitoring.prometheus_enabled is True
    assert str(settings.monitoring.alert_webhook_url) == "http://env.webhook.com/alert" # Surchargé par l'env
    assert settings.monitoring.log_level_console == "DEBUG"

    assert settings.redis.host == "env_redis_host" # Surchargé par l'env
    assert settings.redis.port == 6380 # Du YAML

def test_default_values_applied(create_temp_config_file, monkeypatch):
    """Teste que les valeurs par défaut sont appliquées si les sections/clés manquent."""
    minimal_data = {
        "trading": {"allowed_pairs": ["SOLUSDC"], "base_currency": "USDC"}
        # binance, database, etc., utiliseront les variables d'env ou les défauts de Pydantic
    }
    temp_config_path = create_temp_config_file(minimal_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    monkeypatch.setenv("BINANCE_API_KEY", "default_test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "default_test_secret")
    # Pour la DB, on laisse Pydantic essayer de construire avec ses défauts et ce qui est dans l'env.
    # Si on veut tester le défaut SQLite :memory:, il faut s'assurer que pg_* ne sont pas dans l'env.
    # Ou définir db_type: sqlite, sqlite_path: :memory: dans minimal_data.
    # Ici, nous allons compter sur le fait que `DatabaseConfigModel` a des valeurs par défaut.
    # Si DATABASE_URL est dans l'env, il sera utilisé.
    # Pour ce test, on s'assure qu'aucun paramètre de DB n'est dans l'env pour tester la construction avec défauts.
    db_env_vars = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_DB", "DATABASE_URL", "DB_TYPE", "SQLITE_PATH"]
    for var in db_env_vars:
        monkeypatch.delenv(var, raising=False)


    settings = load_settings()

    assert settings.app.name == "Algo Trading Bot" # Défaut de AppConfig
    assert settings.binance.testnet is False # Défaut de BinanceConfigModel
    assert settings.database.db_type == "postgresql" # Défaut de DatabaseConfigModel
    # Sans pg_user etc. dans l'env, l'URL ne sera pas construite et check_db_url_final_state lèvera une erreur
    # Sauf si on change le db_type par défaut ou qu'on fournit une URL.
    # Pour que ce test passe, nous devons fournir une configuration de base de données minimale
    # qui n'échoue pas à la validation de l'URL. Modifions minimal_data.
    minimal_data_with_db = {
        "trading": {"allowed_pairs": ["SOLUSDC"], "base_currency": "USDC"},
        "database": {"db_type": "sqlite", "sqlite_path": ":memory:"}
    }
    temp_config_path_2 = create_temp_config_file(minimal_data_with_db, "config2.yaml")
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path_2)
    
    settings_with_db = load_settings()
    assert str(settings_with_db.database.url) == "sqlite:///:memory:"
    assert settings_with_db.data.cache_ttl == 3600 # Défaut de DataConfig
    assert settings_with_db.risk.max_drawdown_pct == 0.15 # Défaut de RiskConfig
    assert settings_with_db.monitoring.log_compression == "zip" # Défaut de MonitoringConfig

def test_missing_required_trading_section(create_temp_config_file, monkeypatch):
    """Teste qu'une ConfigurationError est levée si la section 'trading' est manquante."""
    invalid_data = {
        "app": {"environment": "test"},
        # Section "trading" manquante
    }
    temp_config_path = create_temp_config_file(invalid_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    with pytest.raises(SrcConfigurationError) as exc_info:
        load_settings()
    assert "Trading configuration section ('trading:') is missing" in str(exc_info.value)

def test_invalid_data_type_in_yaml(create_temp_config_file, monkeypatch):
    """Teste une ConfigurationError pour un type de donnée incorrect."""
    invalid_data = {
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"},
        "risk": {"max_position_pct": "not-a-float"} # Incorrect type
    }
    temp_config_path = create_temp_config_file(invalid_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    with pytest.raises(SrcConfigurationError) as exc_info: # Encapsule ValidationError
        load_settings()
    # Pydantic v2 error message structure
    assert "Input should be a valid number" in str(exc_info.value).lower()
    assert "risk -> max_position_pct" in str(exc_info.value)


def test_database_url_construction_postgres(create_temp_config_file, monkeypatch):
    """Teste la construction de l'URL PostgreSQL."""
    yaml_data = {
        "database": {
            "db_type": "postgresql",
            "pg_host": "myhost",
            "pg_port": "5432", # Test avec port en string
            "pg_db": "mydb",
        },
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")
    monkeypatch.setenv("POSTGRES_USER", "myuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "mypass")

    settings = load_settings()
    assert isinstance(settings.database.url, PostgresDsn)
    assert str(settings.database.url) == "postgresql+psycopg2://myuser:mypass@myhost:5432/mydb"

def test_database_url_provided_directly_takes_precedence(create_temp_config_file, monkeypatch):
    """Teste que DATABASE_URL de l'env surcharge les paramètres pg_*."""
    yaml_data = {
        "database": {
            "db_type": "postgresql",
            "pg_host": "yaml_host", # Ne devrait pas être utilisé
        },
         "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    direct_url = "postgresql+psycopg2://direct_user:direct_pass@direct_host:1234/direct_db"
    
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")
    monkeypatch.setenv("DATABASE_URL", direct_url)
    # On s'assure qu'aucun autre PG_ ne peut interférer.
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)


    settings = load_settings()
    assert str(settings.database.url) == direct_url

def test_sqlite_path_resolution(create_temp_config_file, monkeypatch):
    """Teste la résolution du chemin SQLite."""
    relative_path = "test_db/my_local.db"
    yaml_data = {
        "database": {"db_type": "sqlite", "sqlite_path": relative_path},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings = load_settings()
    expected_abs_path = (PROJECT_ROOT / relative_path).resolve()
    assert str(settings.database.url) == f"sqlite:///{expected_abs_path}"

def test_storage_path_resolution(create_temp_config_file, monkeypatch):
    """Teste la résolution du storage_path."""
    relative_path = "my_test_data_dir"
    yaml_data = {
        "data": {"storage_path": relative_path},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")
    
    settings = load_settings()
    expected_abs_path = (PROJECT_ROOT / relative_path).resolve()
    assert settings.data.storage_path == expected_abs_path

def test_missing_parquet_export_path_if_enabled(create_temp_config_file, monkeypatch):
    """Teste l'erreur si parquet_export_path est manquant quand activé."""
    yaml_data = {
        "data": {"parquet_export_enabled": True, "parquet_export_path": None},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    with pytest.raises(SrcConfigurationError) as exc_info:
        load_settings()
    assert "parquet_export_path must be set if parquet_export_enabled is True" in str(exc_info.value)

def test_webhook_url_placeholder_to_none(create_temp_config_file, monkeypatch):
    """Teste que le placeholder pour webhook URL est converti en None."""
    yaml_data = {
        "monitoring": {"alert_webhook_url": PLACEHOLDER_WEBHOOK_URL},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings = load_settings()
    assert settings.monitoring.alert_webhook_url is None

    # Test avec la variable d'environnement de haut niveau
    monkeypatch.setenv("ALERT_WEBHOOK_URL", PLACEHOLDER_WEBHOOK_URL)
    settings_env = load_settings() # recharge avec la nouvelle variable d'env
    assert settings_env.monitoring.alert_webhook_url is None


def test_valid_http_webhook_url(create_temp_config_file, monkeypatch):
    """Teste une URL de webhook valide."""
    valid_url = "https://hooks.example.com/services/T000/B000/XXXXXXXX"
    yaml_data = {
        "monitoring": {"alert_webhook_url": valid_url},
         "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings = load_settings()
    assert isinstance(settings.monitoring.alert_webhook_url, HttpUrl)
    assert str(settings.monitoring.alert_webhook_url) == valid_url + "/" # Pydantic HttpUrl ajoute un /

def test_invalid_webhook_url(create_temp_config_file, monkeypatch):
    """Teste une URL de webhook invalide."""
    invalid_url = "not_a_valid_url"
    yaml_data = {
        "monitoring": {"alert_webhook_url": invalid_url},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    with pytest.raises(SrcConfigurationError) as exc_info:
        load_settings()
    assert "Input should be a valid URL" in str(exc_info.value)
    assert "monitoring -> alert_webhook_url" in str(exc_info.value)

def test_load_raw_config_from_yaml_file_not_found(monkeypatch):
    """Teste le comportement si config.yaml est introuvable."""
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", Path("non_existent_config.yaml"))
    # Pydantic lèvera une erreur si les champs obligatoires ne sont pas dans l'env
    # Ici, load_raw_config_from_yaml retournera juste {}
    raw_data = load_raw_config_from_yaml(Path("non_existent_config.yaml"))
    assert raw_data == {}

def test_load_raw_config_from_yaml_invalid_yaml(create_temp_config_file):
    """Teste le comportement si config.yaml a un YAML invalide."""
    invalid_yaml_content = "app: name: Test\n  bad_indent: Problem"
    temp_file_path = create_temp_config_file({}) # Crée un tmp_path
    invalid_file = temp_file_path.parent / "invalid.yaml"
    with open(invalid_file, "w") as f:
        f.write(invalid_yaml_content)

    with pytest.raises(ConfigurationError) as exc_info:
        load_raw_config_from_yaml(invalid_file)
    assert "Error parsing YAML configuration file" in str(exc_info.value)

def test_load_raw_config_from_yaml_not_a_dict(create_temp_config_file):
    """Teste si le contenu du YAML n'est pas un dictionnaire."""
    not_a_dict_content = "- item1\n- item2" # Une liste, pas un dict
    temp_file_path = create_temp_config_file({}) 
    invalid_file = temp_file_path.parent / "not_a_dict.yaml"
    with open(invalid_file, "w") as f:
        f.write(not_a_dict_content)
    
    with pytest.raises(InvalidConfigurationValueError) as exc_info:
        load_raw_config_from_yaml(invalid_file)
    assert "content is not a valid YAML dictionary" in str(exc_info.value)


def test_cross_validation_db_storage_mismatch(create_temp_config_file, monkeypatch):
    """Teste l'erreur si data.storage_type est postgres mais database.db_type ne l'est pas."""
    yaml_data = {
        "data": {"storage_type": "postgres"},
        "database": {"db_type": "sqlite", "sqlite_path": ":memory:"}, # Incohérent
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    with pytest.raises(SrcConfigurationError) as exc_info: # Encapsule InvalidConfigurationValueError
        load_settings()
    assert "If data.storage_type is 'postgres', database.db_type must be 'postgresql'" in str(exc_info.value)

def test_prepare_nested_model_inputs_priority(create_temp_config_file, monkeypatch):
    """
    Teste la priorité des variables d'environnement sur le YAML dans prepare_nested_model_inputs.
    """
    yaml_data = {
        "binance": {"api_key": "yaml_key", "api_secret": "yaml_secret", "testnet": False},
        "database": {"db_type": "postgresql", "pg_host": "yaml_host"},
        "redis": {"host": "yaml_redis_host"},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Variables d'environnement qui devraient surcharger le YAML
    monkeypatch.setenv("BINANCE_API_KEY", "env_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "env_secret")
    # Note: BINANCE_TESTNET n'est pas dans prepare_nested_model_inputs, il est géré par pydantic-settings directement.
    monkeypatch.setenv("POSTGRES_HOST", "env_host") # Pour database
    monkeypatch.setenv("REDIS_HOST", "env_redis_host_override") # Pour redis

    # Variables nécessaires pour la construction de l'URL de la DB
    monkeypatch.setenv("POSTGRES_USER", "user_from_env")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass_from_env")
    monkeypatch.setenv("POSTGRES_DB", "db_from_env")


    settings = load_settings()

    assert settings.binance.api_key.get_secret_value() == "env_key"
    assert settings.binance.api_secret.get_secret_value() == "env_secret"
    assert settings.database.pg_host == "env_host" # Surchargé
    assert settings.redis.host == "env_redis_host_override" # Surchargé