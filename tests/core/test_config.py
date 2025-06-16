# tests/core/test_config.py
import pytest
import os
from pydantic import ValidationError

from src.core.config import TradingConfig, Settings, load_raw_config_from_yaml, CONFIG_FILE_PATH
from src.core.exceptions import InvalidConfigurationValueError

@pytest.fixture
def sample_trading_config_data():
    return {
        "mode": "cross_margin",
        "base_currency": "USDC",
    }

def test_trading_config_normalize_allowed_pairs(sample_trading_config_data):
    """Teste la normalisation et validation de allowed_pairs dans TradingConfig."""
    config_data = sample_trading_config_data.copy()
    config_data["allowed_pairs"] = ["BTC/USDT", "eth/usdc", "ADAUSDT  "]
    
    trading_config = TradingConfig(**config_data)
    assert trading_config.allowed_pairs == ["BTCUSDT", "ETHUSDC", "ADAUSDT"]

def test_trading_config_allowed_pairs_empty(sample_trading_config_data):
    config_data = sample_trading_config_data.copy()
    config_data["allowed_pairs"] = []
    trading_config = TradingConfig(**config_data)
    assert trading_config.allowed_pairs == []

def test_trading_config_allowed_pairs_invalid_item_type(sample_trading_config_data):
    config_data = sample_trading_config_data.copy()
    config_data["allowed_pairs"] = ["BTC/USDT", 123]
    with pytest.raises(InvalidConfigurationValueError) as excinfo:
        TradingConfig(**config_data)
    assert "Each item in allowed_pairs must be a string" in str(excinfo.value)

def test_trading_config_allowed_pairs_not_a_list(sample_trading_config_data):
    config_data = sample_trading_config_data.copy()
    config_data["allowed_pairs"] = "BTC/USDT" # Devrait être une liste
    with pytest.raises(InvalidConfigurationValueError) as excinfo:
        TradingConfig(**config_data)
    assert "allowed_pairs must be a list" in str(excinfo.value)

def test_trading_config_allowed_pairs_too_short(sample_trading_config_data):
    config_data = sample_trading_config_data.copy()
    config_data["allowed_pairs"] = ["BTC"]
    with pytest.raises(InvalidConfigurationValueError) as excinfo:
        TradingConfig(**config_data)
    assert "format is invalid or too short" in str(excinfo.value)


@pytest.fixture
def minimal_yaml_data_for_settings():
    # Fournit une structure YAML minimale pour que Settings puisse s'initialiser
    # sans erreurs sur les champs manquants non liés à 'trading.allowed_pairs'.
    return {
        "app": {"name": "TestApp", "version": "0.1", "environment": "test"},
        "binance": {"api_key": "test_key", "api_secret": "test_secret", "testnet": True},
        "database": {"db_type": "sqlite", "sqlite_path": ":memory:"},
        # data, risk, monitoring, system utiliseront leurs valeurs par défaut ou default_factory
    }

def test_settings_allowed_pairs_from_yaml(minimal_yaml_data_for_settings, monkeypatch):
    """Teste la normalisation de TRADING_ALLOWED_PAIRS venant du YAML."""
    yaml_data = minimal_yaml_data_for_settings.copy()
    yaml_data["trading"] = {"allowed_pairs": ["ETH/USDC", "solana/USDC"]}

    # Simuler l'absence de variables d'environnement pour ces champs
    monkeypatch.delenv("TRADING_ALLOWED_PAIRS", raising=False)

    settings = Settings(**yaml_data)
    assert settings.trading.allowed_pairs == ["ETHUSDC", "SOLANAUSDC"]

def test_settings_allowed_pairs_from_env_comma_separated(minimal_yaml_data_for_settings, monkeypatch):
    """Teste la normalisation de TRADING_ALLOWED_PAIRS venant d'une variable d'env (string)."""
    monkeypatch.setenv("TRADING_ALLOWED_PAIRS", "BTC/USDT,ETH/usdc")
    
    # Le YAML peut avoir ou non une section trading, l'env devrait la surcharger pour allowed_pairs
    yaml_data = minimal_yaml_data_for_settings.copy()
    yaml_data.pop("trading", None) # S'assurer qu'il n'y a pas de conflit YAML

    settings = Settings(**yaml_data)
    assert settings.trading.allowed_pairs == ["BTCUSDT", "ETHUSDC"]

def test_settings_allowed_pairs_from_env_json_list(minimal_yaml_data_for_settings, monkeypatch):
    """Teste la normalisation de TRADING_ALLOWED_PAIRS venant d'une variable d'env (JSON list string)."""
    monkeypatch.setenv("TRADING_ALLOWED_PAIRS", '["XRP/USDT", "ada/usdt"]')
    
    yaml_data = minimal_yaml_data_for_settings.copy()
    yaml_data.pop("trading", None)

    settings = Settings(**yaml_data)
    assert settings.trading.allowed_pairs == ["XRPUSDT", "ADAUSDT"]

def test_settings_allowed_pairs_env_takes_precedence(minimal_yaml_data_for_settings, monkeypatch):
    """Vérifie que la variable d'environnement pour allowed_pairs a la priorité sur le YAML."""
    monkeypatch.setenv("TRADING_ALLOWED_PAIRS", "DOT/USDC")
    
    yaml_data = minimal_yaml_data_for_settings.copy()
    yaml_data["trading"] = {"allowed_pairs": ["ETH/USDC", "SOL/USDC"]} # YAML différent

    settings = Settings(**yaml_data)
    assert settings.trading.allowed_pairs == ["DOTUSDC"]