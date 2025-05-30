"""
Tests for the ConfigManager class.
"""
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from core.config_manager import ConfigManager, ConfigError

# Sample configuration for testing
SAMPLE_CONFIG = {
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
            "macd_signal": 9
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


def test_config_manager_initialization():
    """Test ConfigManager initialization with valid config."""
    with patch('builtins.open', mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        config = ConfigManager("dummy_path.json")
        assert config is not None


def test_get_mt5_config():
    """Test getting MT5 configuration."""
    with patch('builtins.open', mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        config = ConfigManager("dummy_path.json")
        mt5_config = config.get_mt5_config()
        assert mt5_config["server"] == "Demo"
        assert mt5_config["login"] == 12345678


def test_get_global_settings():
    """Test getting global settings."""
    with patch('builtins.open', mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        config = ConfigManager("dummy_path.json")
        settings = config.get_global_settings()
        assert settings["max_open_positions"] == 5
        assert settings["default_lot_size"] == 0.1


def test_get_active_symbols():
    """Test getting active symbols."""
    with patch('builtins.open', mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        config = ConfigManager("dummy_path.json")
        active_symbols = config.get_active_symbols()
        assert "EURUSD" in active_symbols
        assert "GBPUSD" not in active_symbols


def test_get_symbol_config():
    """Test getting symbol configuration with defaults."""
    with patch('builtins.open', mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        config = ConfigManager("dummy_path.json")
        symbol_config = config.get_symbol_config("EURUSD")
        
        # Check that defaults are merged
        assert symbol_config["indicators"]["rsi_period"] == 14
        
        # Check that symbol-specific overrides work
        assert symbol_config["risk"]["risk_per_trade_pct"] == 0.5
        assert symbol_config["risk"]["max_risk_pct"] == 3.0
        assert symbol_config["risk"]["min_risk_reward_ratio"] == 1.5  # From defaults


def test_invalid_config_missing_section():
    """Test initialization with invalid config (missing required section)."""
    invalid_config = SAMPLE_CONFIG.copy()
    del invalid_config["metatrader5"]
    
    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_config))):
        with pytest.raises(ConfigError, match="Missing required section in config"):
            ConfigManager("dummy_path.json")


def test_nonexistent_config_file():
    """Test initialization with non-existent config file."""
    with pytest.raises(ConfigError, match="Configuration file not found"):
        ConfigManager("nonexistent_config.json")


def test_invalid_json_config():
    """Test initialization with invalid JSON config."""
    with patch('builtins.open', mock_open(read_data='{invalid json}')), \
         pytest.raises(ConfigError, match="Invalid JSON in config file"):
        ConfigManager("invalid.json")
