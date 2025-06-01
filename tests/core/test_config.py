# tests/strategies/test_config.py
"""
Tests pour le module de configuration src.core.config.
Ces tests vérifient le chargement, la validation, les valeurs par défaut,
la surcharge par variables d'environnement et la gestion des erreurs.
"""
import os
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, Generator

from pydantic import ValidationError, HttpUrl, PostgresDsn, SecretStr
from pydantic_core import PydanticUndefined

from src.core.config import (
    load_settings,
    Settings,
    AppConfig,
    BinanceConfigModel,
    DatabaseConfigModel,
    DataConfig,
    TradingConfig,
    RiskConfig,
    MonitoringConfig,
    RedisConfigModel,
    CONFIG_FILE_PATH,
    BASE_DIR,
    load_raw_config_from_yaml,
)
from src.core.exceptions import (
    ConfigurationError as SrcConfigurationError, # Renommé pour éviter conflit avec pytest.ConfigurationError
    InvalidConfigurationValueError,
)

# Utiliser un alias pour les exceptions Pydantic pour plus de clarté
PydanticValidationError = ValidationError


# Fixture pour créer un fichier de configuration temporaire
@pytest.fixture
def create_temp_config_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Crée un fichier YAML de configuration temporaire et retourne son chemin."""
    files_created = []

    def _writer(data: Dict[str, Any], filename: str = "config.yaml") -> Path:
        temp_file = tmp_path / filename
        with open(temp_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        files_created.append(temp_file)
        return temp_file

    yield _writer

    # Nettoyage : supprimer les fichiers créés après le test si nécessaire
    # for f_path in files_created:
    #     if f_path.exists():
    #         f_path.unlink()


# Données YAML valides pour les tests
@pytest.fixture
def minimal_valid_yaml_data() -> Dict[str, Any]:
    """Données YAML minimales valides."""
    return {
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"},
        # Binance keys seront mockées par env vars
    }


@pytest.fixture
def full_valid_yaml_data() -> Dict[str, Any]:
    """Données YAML complètes et valides."""
    return {
        "app": {
            "name": "Test Algo Bot",
            "version": "0.1.0",
            "environment": "test_yaml_env", # Sera surchargé par env var
        },
        "binance": {
            "api_key": "yaml_api_key", # Sera surchargé par env var
            "api_secret": "yaml_api_secret", # Sera surchargé par env var
            "testnet": False, # Sera surchargé par env var
        },
        "database": {
            "db_type": "postgresql",
            "pg_host": "test_db_host_yaml", # Valeur YAML
            "pg_port": 5433, # Valeur YAML
            "pg_user": "yaml_pg_user", # Sera surchargé par env var
            "pg_password": "yaml_pg_password", # Sera surchargé par env var
            "pg_db": "test_algobot_db_yaml", # Valeur YAML
        },
        "data": {
            "storage_type": "parquet",
            "storage_path": "data/test_klines_yaml",
            "cache_enabled": False, # Sera surchargé par env var
        },
        "trading": {
            "mode": "spot", # Sera surchargé par env var
            "base_currency": "USDT", # Sera surchargé par env var
            "allowed_pairs": ["ETHUSDT_yaml", "ADAUSDT_yaml"],
        },
        "risk": {"max_position_pct": 0.03}, # Valeur YAML
        "monitoring": {
            "prometheus_enabled": False, # Valeur YAML
            "alert_webhook_url": "http://yaml.webhook.com/alert", # Sera surchargé
            "log_level_console": "DEBUG", # Sera surchargé
        },
        "redis": {"host": "yaml_redis_host", "port": 6380}, # Sera surchargé
    }


def test_load_settings_minimal_valid_yaml(
    create_temp_config_file, monkeypatch, minimal_valid_yaml_data
):
    """Teste le chargement avec un YAML minimal valide et des secrets via env."""
    temp_config_path = create_temp_config_file(minimal_valid_yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Mocker les variables d'environnement minimales requises
    monkeypatch.setenv("BINANCE_API_KEY", "env_api_key_minimal")
    monkeypatch.setenv("BINANCE_API_SECRET", "env_api_secret_minimal")
    # S'assurer que les variables DB et Redis ne sont pas là pour prendre les défauts
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)


    settings = load_settings()

    assert settings.app.name == "Algo Trading Bot" # Valeur par défaut
    assert settings.binance.api_key.get_secret_value() == "env_api_key_minimal"
    assert settings.trading.base_currency == "USDC" # Du YAML minimal
    assert settings.database.db_type == "sqlite" # Valeur par défaut si rien n'est spécifié
    assert settings.redis is None # Valeur par défaut si rien n'est spécifié


def test_load_settings_full_valid_yaml_and_env_override(
    create_temp_config_file, monkeypatch, full_valid_yaml_data
):
    """Teste le chargement avec un YAML complet et des surcharges par variables d'environnement."""
    temp_config_path = create_temp_config_file(full_valid_yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Mocker les variables d'environnement qui surchargeront le YAML ou fourniront des secrets
    monkeypatch.setenv("APP_ENV", "env_override_app_env") # Surcharge YAML
    monkeypatch.setenv("BINANCE_API_KEY", "env_api_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "env_api_secret")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("POSTGRES_USER", "env_pg_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "env_pg_password")
    # Pour ce test, nous ne surchargeons PAS POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB par l'env
    # pour vérifier que les valeurs YAML sont prises pour ceux-là.
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)

    monkeypatch.setenv("DATA_CACHE_ENABLED", "true") # Surcharge YAML
    monkeypatch.setenv("TRADING_MODE", "cross_margin") # Surcharge YAML
    monkeypatch.setenv("TRADING_BASE_CURRENCY", "BTC") # Surcharge YAML
    monkeypatch.setenv("RISK_MAX_POSITION_PCT", "0.08") # Surcharge YAML

    monkeypatch.setenv("REDIS_HOST", "env_redis_host")
    monkeypatch.setenv("REDIS_PORT", "6399") # Surcharge YAML
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://env.webhook.com/alert")
    monkeypatch.setenv("LOG_LEVEL_CONSOLE", "WARNING") # Surcharge YAML


    settings = load_settings()

    # Vérifications des surcharges et des valeurs YAML
    assert settings.app.environment == "env_override_app_env" # Surchargé par env
    assert settings.app.name == "Test Algo Bot" # Du YAML

    assert settings.binance.api_key.get_secret_value() == "env_api_key"
    assert settings.binance.api_secret.get_secret_value() == "env_api_secret"
    assert settings.binance.testnet is True # Surchargé par env

    assert settings.database.db_type == "postgresql" # Du YAML
    assert settings.database.pg_user == "env_pg_user" # Surchargé par env
    assert settings.database.pg_password.get_secret_value() == "env_pg_password" # Surchargé par env
    assert settings.database.pg_host == "test_db_host_yaml" # Doit venir du YAML car non surchargé par env
    assert settings.database.pg_port == 5433 # Du YAML (converti en int)
    assert settings.database.pg_db == "test_algobot_db_yaml" # Du YAML

    assert settings.data.cache_enabled is True # Surchargé par env
    assert settings.data.storage_path.name == "test_klines_yaml" # Du YAML

    assert settings.trading.mode == "cross_margin" # Surchargé par env
    assert settings.trading.base_currency == "BTC" # Surchargé par env
    assert settings.trading.allowed_pairs == ["ETHUSDT_yaml", "ADAUSDT_yaml"] # Du YAML

    assert settings.risk.max_position_pct == 0.08 # Surchargé par env

    assert settings.monitoring.alert_webhook_url == HttpUrl("http://env.webhook.com/alert") # Surchargé
    assert settings.monitoring.log_level_console == "WARNING" # Surchargé

    assert settings.redis is not None
    assert settings.redis.host == "env_redis_host" # Surchargé
    assert settings.redis.port == 6399 # Surchargé


def test_default_values_applied(create_temp_config_file, monkeypatch):
    """Teste que les valeurs par défaut sont appliquées si non spécifiées."""
    # YAML minimal ne spécifiant pas grand chose
    minimal_data = {
        "trading": {"allowed_pairs": ["SOLUSDC"], "base_currency": "SOL"}
    }
    temp_config_path = create_temp_config_file(minimal_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Mocker les secrets requis
    monkeypatch.setenv("BINANCE_API_KEY", "default_test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "default_test_secret")
    # S'assurer qu'aucune autre variable d'environnement ne surcharge les défauts
    for var in [
        "APP_ENV", "BINANCE_TESTNET", "DB_TYPE", "POSTGRES_USER", "REDIS_HOST",
        "DATA_STORAGE_TYPE", "RISK_MAX_POSITION_PCT", "LOG_LEVEL_CONSOLE"
    ]:
        monkeypatch.delenv(var, raising=False)


    settings = load_settings()

    # Vérifier les valeurs par défaut des modèles Pydantic
    assert settings.app.name == "Algo Trading Bot"
    assert settings.app.version == "1.0.0" # Valeur par défaut de AppConfig
    assert settings.app.environment == "development" # Valeur par défaut

    assert settings.binance.testnet is False # Valeur par défaut

    assert settings.database.db_type == "sqlite" # Valeur par défaut
    assert settings.database.sqlite_path.name == "algobot_main.db" # Valeur par défaut
    assert settings.database.pool_size == 5 # Valeur par défaut

    assert settings.data.storage_type == "parquet" # Valeur par défaut
    assert settings.data.storage_path.name == "historical" # Valeur par défaut
    assert settings.data.cache_enabled is False # Valeur par défaut
    assert settings.data.cache_ttl == 3600 # Valeur par défaut

    assert settings.risk.max_position_pct == 0.1 # Valeur par défaut de RiskConfig
    assert settings.risk.max_drawdown_pct == 0.15 # Valeur par défaut

    assert settings.monitoring.prometheus_enabled is False # Valeur par défaut
    assert settings.monitoring.log_level_console == "INFO" # Valeur par défaut

    assert settings.redis is None # Par défaut, si host n'est pas fourni


def test_missing_required_trading_section(create_temp_config_file, monkeypatch):
    """Teste une ConfigurationError si une section requise (trading) manque."""
    # YAML sans la section "trading"
    invalid_data = {"app": {"name": "Test Bot"}}
    temp_config_path = create_temp_config_file(invalid_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey") # Requis par BinanceConfigModel
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret") # Requis

    with pytest.raises(SrcConfigurationError) as exc_info:
        load_settings()
    # Pydantic v2 error message structure for missing field
    assert "Field required" in str(exc_info.value)
    assert "'trading'" in str(exc_info.value) # Vérifie que c'est bien la section trading


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
    # Le message exact peut varier légèrement, mais la substance doit être là.
    error_str_lower = str(exc_info.value).lower()
    assert "input should be a valid number" in error_str_lower
    assert "unable to parse string as a number" in error_str_lower
    assert "'risk -> max_position_pct'" in error_str_lower or "'risk.max_position_pct'" in error_str_lower


def test_database_url_construction_postgres(create_temp_config_file, monkeypatch):
    """Teste la construction de l'URL PostgreSQL."""
    yaml_data = {
        "database": {
            "db_type": "postgresql",
            # pg_host, pg_port, pg_db seront fournis par env pour ce test
        },
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)

    # Mocker toutes les variables d'environnement pour PostgreSQL
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")
    monkeypatch.setenv("POSTGRES_USER", "test_pg_user_env")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pg_pass_env")
    monkeypatch.setenv("POSTGRES_HOST", "test_pg_host_env")
    monkeypatch.setenv("POSTGRES_PORT", "5439") # Port différent pour tester
    monkeypatch.setenv("POSTGRES_DB", "test_pg_db_env")

    settings = load_settings()

    assert settings.database is not None
    assert settings.database.db_type == "postgresql"
    assert isinstance(settings.database.sqlalchemy_database_url, str) # Doit être une string

    # Vérifier que la string est un DSN Postgres valide
    # Pydantic le fait déjà lors de la validation du modèle si le champ est typé PostgresDsn
    # Ici, on vérifie la string construite.
    try:
        parsed_dsn = PostgresDsn(settings.database.sqlalchemy_database_url)
    except PydanticValidationError:
        pytest.fail(f"Generated DSN is not a valid PostgresDsn: {settings.database.sqlalchemy_database_url}")

    expected_url_parts = [
        "postgresql+psycopg2://",
        "test_pg_user_env",
        ":",
        "test_pg_pass_env",
        "@",
        "test_pg_host_env",
        ":5439",
        "/",
        "test_pg_db_env",
    ]
    for part in expected_url_parts:
        assert part in settings.database.sqlalchemy_database_url

    # Vérifier que les champs individuels sont aussi peuplés
    assert settings.database.pg_user == "test_pg_user_env"
    assert settings.database.pg_password.get_secret_value() == "test_pg_pass_env"
    assert settings.database.pg_host == "test_pg_host_env"
    assert settings.database.pg_port == 5439 # Doit être un int
    assert settings.database.pg_db == "test_pg_db_env"


def test_database_url_provided_directly_takes_precedence(
    create_temp_config_file, monkeypatch
):
    """Teste que DATABASE_URL fourni directement a la priorité."""
    direct_db_url = "postgresql+psycopg2://direct_user:direct_pass@direct_host:1234/direct_db"
    yaml_data = {
        "database": { # Ces valeurs devraient être ignorées
            "db_type": "sqlite",
            "sqlite_path": "some_other.db",
            "pg_host": "ignored_host",
        },
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")
    monkeypatch.setenv("DATABASE_URL", direct_db_url) # Fournir l'URL directement

    settings = load_settings()

    assert settings.database is not None
    assert settings.database.sqlalchemy_database_url == direct_db_url
    # Les autres champs de DatabaseConfig (comme db_type, pg_host) pourraient ne pas
    # être peuplés si DATABASE_URL est fourni, selon la logique de Pydantic.
    # Le plus important est que sqlalchemy_database_url soit correct.
    # Si db_type est nécessaire même avec DATABASE_URL, il faut le valider.
    # D'après config.py, db_type est inféré de DATABASE_URL.
    assert settings.database.db_type == "postgresql"


def test_sqlite_path_resolution(create_temp_config_file, monkeypatch):
    """Teste la résolution du chemin SQLite."""
    relative_path = "data/test_sqlite.db"
    yaml_data = {
        "database": {"db_type": "sqlite", "sqlite_path": relative_path},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings = load_settings()
    assert settings.database is not None
    expected_path = BASE_DIR / relative_path
    assert settings.database.sqlite_path == expected_path
    assert settings.database.sqlalchemy_database_url == f"sqlite:///{expected_path.as_posix()}"


def test_storage_path_resolution(create_temp_config_file, monkeypatch):
    """Teste la résolution du chemin de stockage des données."""
    relative_path = "data/my_klines"
    yaml_data = {
        "data": {"storage_path": relative_path},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings = load_settings()
    assert settings.data is not None
    expected_path = BASE_DIR / relative_path
    assert settings.data.storage_path == expected_path


def test_missing_parquet_export_path_if_enabled(
    create_temp_config_file, monkeypatch
):
    """
    Teste une ConfigurationError si parquet_export_enabled est true
    mais parquet_export_path n'est pas fourni.
    """
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
    assert "parquet_export_path must be set if parquet_export_enabled is true" in str(exc_info.value).lower()


def test_webhook_url_placeholder_to_none(create_temp_config_file, monkeypatch):
    """Teste que le placeholder pour webhook URL est converti en None."""
    yaml_data = {
        "monitoring": {"alert_webhook_url": "YOUR_WEBHOOK_URL_HERE"}, # Placeholder
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path = create_temp_config_file(yaml_data)
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings = load_settings()
    assert settings.monitoring is not None
    assert settings.monitoring.alert_webhook_url is None


def test_valid_http_webhook_url(create_temp_config_file, monkeypatch):
    """Teste une URL de webhook valide."""
    valid_url_no_slash = "https://hooks.example.com/services/T000/B000/XXXXXXXX"
    valid_url_with_slash = "http://another.hook.org/path/"

    # Test 1: URL sans slash final
    yaml_data_1 = {
        "monitoring": {"alert_webhook_url": valid_url_no_slash},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path_1 = create_temp_config_file(yaml_data_1, "config1.yaml")
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path_1)
    monkeypatch.setenv("BINANCE_API_KEY", "somekey")
    monkeypatch.setenv("BINANCE_API_SECRET", "somesecret")

    settings1 = load_settings()
    assert settings1.monitoring is not None
    assert isinstance(settings1.monitoring.alert_webhook_url, HttpUrl)
    # Pydantic HttpUrl ne garantit pas l'ajout d'un slash final.
    # Sa représentation str() est généralement l'URL telle que validée.
    assert str(settings1.monitoring.alert_webhook_url) == valid_url_no_slash

    # Test 2: URL avec slash final
    yaml_data_2 = {
        "monitoring": {"alert_webhook_url": valid_url_with_slash},
        "trading": {"allowed_pairs": ["BTCUSDC"], "base_currency": "USDC"}
    }
    temp_config_path_2 = create_temp_config_file(yaml_data_2, "config2.yaml")
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_path_2)
    # Les env vars BINANCE sont déjà settées

    settings2 = load_settings()
    assert settings2.monitoring is not None
    assert isinstance(settings2.monitoring.alert_webhook_url, HttpUrl)
    assert str(settings2.monitoring.alert_webhook_url) == valid_url_with_slash


def test_invalid_webhook_url(create_temp_config_file, monkeypatch):
    """Teste une URL de webhook invalide."""
    invalid_url = "not_a_url"
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
    assert "invalid or missing URL scheme" in str(exc_info.value).lower()


def test_load_raw_config_from_yaml_file_not_found(tmp_path, monkeypatch):
    """Teste le chargement si le fichier YAML n'existe pas (doit retourner un dict vide)."""
    non_existent_path = tmp_path / "does_not_exist.yaml"
    # Pas besoin de mocker CONFIG_FILE_PATH ici, on appelle load_raw_config_from_yaml directement
    raw_config = load_raw_config_from_yaml(non_existent_path)
    assert raw_config == {}


def test_load_raw_config_from_yaml_invalid_yaml(create_temp_config_file):
    """Teste une erreur de parsing YAML."""
    invalid_yaml_content = "app: name: Test\n  bad_indent: here" # YAML malformé
    temp_file_path = create_temp_config_file({}) # Crée un fichier temporaire vide pour obtenir le chemin
    with open(temp_file_path, "w") as f:
        f.write(invalid_yaml_content)

    with pytest.raises(SrcConfigurationError) as exc_info:
        load_raw_config_from_yaml(temp_file_path)
    assert "error parsing yaml configuration file" in str(exc_info.value).lower()


def test_load_raw_config_from_yaml_not_a_dict(create_temp_config_file):
    """Teste si le contenu du YAML n'est pas un dictionnaire."""
    not_a_dict_content = "- item1\n- item2" # Une liste, pas un dict
    # Crée un fichier temporaire vide pour obtenir le chemin du répertoire parent
    base_temp_file_path = create_temp_config_file({}, "base.yaml")
    invalid_file = base_temp_file_path.parent / "not_a_dict.yaml"
    with open(invalid_file, "w", encoding="utf-8") as f:
        f.write(not_a_dict_content)

    with pytest.raises(InvalidConfigurationValueError) as exc_info: # S'attendre à l'exception spécifique
        load_raw_config_from_yaml(invalid_file)
    assert "content is not a valid yaml dictionary" in str(exc_info.value).lower()
    assert str(invalid_file) in str(exc_info.value) # Vérifier que le chemin est dans le message


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

    with pytest.raises(SrcConfigurationError) as exc_info: # Le validateur de Settings lève SrcConfigurationError
        load_settings()

    # Le message exact peut être imbriqué si InvalidConfigurationValueError est causée par une ValidationError
    # Vérifions la présence des mots clés importants.
    error_message = str(exc_info.value).lower()
    assert "if data.storage_type is 'postgres'" in error_message
    assert "then database.db_type must be 'postgresql'" in error_message


def test_prepare_nested_model_inputs_priority(monkeypatch):
    """
    Teste la priorité des sources pour les modèles imbriqués:
    1. Variables d'environnement spécifiques au champ (ex: BINANCE_TESTNET pour binance.testnet)
    2. Dictionnaire YAML pour le modèle imbriqué (ex: yaml_data['binance'])
    3. Variables d'environnement pour le modèle imbriqué entier (ex: BINANCE='{"testnet": true}')
    4. Valeurs par défaut du modèle Pydantic.
    """
    # Simuler un CONFIG_FILE_PATH qui n'existe pas pour que load_raw_config_from_yaml retourne {}
    # Cela force à se baser sur les variables d'environnement et les valeurs par défaut.
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", Path("non_existent_config.yaml"))

    # Cas 1: Env var spécifique au champ a la priorité
    monkeypatch.setenv("BINANCE_API_KEY", "key_env_direct")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret_env_direct")
    monkeypatch.setenv("BINANCE_TESTNET", "true") # Env var spécifique au champ
    monkeypatch.setenv("BINANCE", '{"testnet": false, "api_key": "key_env_model"}') # Env var pour le modèle entier

    settings = load_settings()
    assert settings.binance.testnet is True # Priorité à BINANCE_TESTNET
    assert settings.binance.api_key.get_secret_value() == "key_env_direct" # Priorité à BINANCE_API_KEY

    # Nettoyer les variables d'environnement
    monkeypatch.delenv("BINANCE_TESTNET")
    monkeypatch.delenv("BINANCE") # api_key et secret restent

    # Cas 2: Env var pour le modèle entier a la priorité sur les défauts
    # (BINANCE_TESTNET est maintenant non défini, donc on s'attend à ce que BINANCE='...' soit utilisé)
    monkeypatch.setenv("BINANCE", '{"testnet": false, "api_key": "key_env_model_2"}')
    # BINANCE_API_KEY est toujours "key_env_direct" et a priorité sur la clé dans BINANCE='...'
    # BINANCE_API_SECRET est toujours "secret_env_direct"

    settings_case2 = load_settings()
    assert settings_case2.binance.testnet is False # Vient de BINANCE='...'
    assert settings_case2.binance.api_key.get_secret_value() == "key_env_direct" # Toujours prioritaire
    assert settings_case2.binance.api_secret.get_secret_value() == "secret_env_direct"

    # Nettoyer
    monkeypatch.delenv("BINANCE_API_KEY")
    monkeypatch.delenv("BINANCE_API_SECRET")
    monkeypatch.delenv("BINANCE")

    # Cas 3: Valeurs par défaut si rien n'est fourni
    # Requis: BINANCE_API_KEY et BINANCE_API_SECRET doivent être fournis pour que BinanceConfigModel valide
    monkeypatch.setenv("BINANCE_API_KEY", "default_case_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "default_case_secret")
    # S'assurer que les autres env vars pertinentes sont absentes
    monkeypatch.delenv("TRADING_MODE", raising=False)


    settings_case3 = load_settings()
    assert settings_case3.binance.testnet is False # Valeur par défaut de BinanceConfigModel
    assert settings_case3.trading.mode == "spot" # Valeur par défaut de TradingConfig

