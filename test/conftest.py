"""
Configuration file for pytest.
Defines fixtures and other test configurations.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from core.config_manager import ConfigManager


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "metatrader5": {
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            "server": "Demo",
            "login": 12345678,
            "password": "password123"
        },
        "global_settings": {
            "max_open_positions": 5,
            "default_lot_size": 0.1,
            "max_daily_drawdown_pct": 5.0
        },
        "logging": {
            "level": "INFO",
            "file": "trading_bot.log",
            "max_size_mb": 10,
            "backup_count": 5
        },
        "defaults": {
            "indicators": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2.0
            },
            "sr": {
                "pivot_lookback": 5,
                "fractal_lookback": 2,
                "merge_threshold_pct": 0.5
            },
            "risk": {
                "risk_per_trade_pct": 1.0,
                "max_risk_pct": 5.0,
                "min_risk_reward_ratio": 1.5
            }
        },
        "symbols": {
            "EURUSD": {
                "enabled": True,
                "risk": {
                    "risk_per_trade_pct": 0.5,
                    "max_risk_pct": 3.0
                }
            },
            "GBPUSD": {
                "enabled": False
            }
        }
    }


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.3)
    low = close - np.abs(np.random.randn(100) * 0.3)
    volume = np.random.randint(100, 1000, size=100)
    
    return pd.DataFrame({
        'open': close - 0.1,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def mock_config():
    """Create a mock ConfigManager for testing."""
    config = MagicMock(spec=ConfigManager)
    config.get_indicator_params.return_value = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2.0
    }
    return config
