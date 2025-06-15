import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

@pytest.fixture
def sample_ohlcv_data():
    """Données OHLCV de test."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    return pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

@pytest.fixture
def test_config():
    """Configuration de test."""
    return {
        'initial_capital': 10000,
        'fee_config': {'method': 'PERCENTAGE', 'value': 0.001},
        'slippage_config': {'method': 'PERCENTAGE', 'value': 0.0001}
    }

@pytest.fixture
def mock_settings(monkeypatch):
    """Mock des paramètres globaux pour isoler les tests."""
    from src.core.config import Settings
    from pydantic import SecretStr
    
    test_settings = Settings(
        APP_NAME="Test Trading Bot",
        APP_ENV="test",
        BINANCE_API_KEY=SecretStr("test_key"),
        BINANCE_API_SECRET=SecretStr("test_secret"),
        BINANCE_TESTNET=True,
        DB_TYPE="sqlite",
        SQLITE_PATH=":memory:",
        TRADING_MODE="cross_margin",
        TRADING_BASE_CURRENCY="USDC",
        TRADING_ALLOWED_PAIRS="BTC/USDC,ETH/USDC"
    )
    
    # Patch get_settings pour retourner notre instance de test
    from src.core.config import get_settings
    monkeypatch.setattr("src.core.config.get_settings", lambda: test_settings)
    
    return test_settings