"""
Tests for the ConfigManager class.
"""
import os
import json
import pytest
from pathlib import Path
import time
from unittest.mock import patch, mock_open, MagicMock, call
import jsonschema # For asserting validation errors specifically

from core.config_manager import ConfigManager, ConfigError
from core import constants as C

# Valid, comprehensive configuration data fixture
@pytest.fixture
def valid_config_data():
    return {
        C.CONFIG_METATRADER5: {
            C.CONFIG_MT5_PATH: "C:/Program Files/MetaTrader 5/terminal64.exe",
            C.CONFIG_MT5_SERVER: "TestServer",
            C.CONFIG_MT5_LOGIN: 12345678,
            C.CONFIG_MT5_PASSWORD: "password123",
            C.CONFIG_MT5_TIMEOUT: 60000,
            C.CONFIG_MT5_PORTABLE: False
        },
        C.CONFIG_GLOBAL_SETTINGS: {
            C.CONFIG_MAX_TOTAL_TRADES: 10,
            C.CONFIG_MAX_SLIPPAGE_PIPS: 2.0,
            C.CONFIG_MAGIC_NUMBER: 12345,
            C.CONFIG_DEVIATION: 20,
            C.CONFIG_PAPER_TRADING: True,
            C.CONFIG_MAX_SLIPPAGE_POINTS: 20,
            C.CONFIG_KILL_SWITCH_FILE_PATH: "KILL_SWITCH.txt",
            C.CONFIG_KILL_SWITCH_CLOSE_POSITIONS: False,
            C.CONFIG_LOOP_INTERVAL: 2
        },
        C.CONFIG_LOGGING: {
            C.CONFIG_LOGGING_LEVEL: "DEBUG",
            C.CONFIG_LOGGING_FILE: "logs/bot_test.log",
            C.CONFIG_LOGGING_MAX_BYTES: 5242880,
            C.CONFIG_LOGGING_BACKUP_COUNT: 3,
            C.CONFIG_LOGGING_FORMAT: "%(asctime)s - %(levelname)s - %(message)s"
        },
        C.CONFIG_DEFAULTS: {
            C.CONFIG_INDICATORS: {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "atr_period": 10, "atr_multiplier": 3.0},
            C.CONFIG_SR: {"method": "fractals", "fractal_window": 5},
            C.CONFIG_RISK: {"risk_per_trade": 0.015, "max_risk_per_trade": 0.025, "max_daily_drawdown": 0.06}
        },
        C.CONFIG_SYMBOLS: {
            "EURUSD": {C.CONFIG_ENABLED: True, C.CONFIG_LOT_SIZE: 0.01, C.CONFIG_SPREAD_LIMIT_PIPS: 2.5,
                       C.CONFIG_RISK: {C.CONFIG_RISK_PER_TRADE: 0.01}},
            "GBPUSD": {C.CONFIG_ENABLED: True, C.CONFIG_LOT_SIZE: 0.02}
        },
        "timeframes": {"M5": "M5", "H1": "H1"} # Example
    }

# Simplified mock schema for testing purposes
MOCK_SCHEMA_CONTENT = {
    "type": "object",
    "required": [C.CONFIG_METATRADER5, C.CONFIG_GLOBAL_SETTINGS, C.CONFIG_LOGGING, C.CONFIG_DEFAULTS, C.CONFIG_SYMBOLS],
    "properties": {
        C.CONFIG_METATRADER5: {"type": "object", "required":["path", "server", "login", "password", "timeout", "portable"]},
        C.CONFIG_GLOBAL_SETTINGS: {"type": "object", "required": [
            C.CONFIG_MAX_TOTAL_TRADES, C.CONFIG_MAX_SLIPPAGE_PIPS, C.CONFIG_MAGIC_NUMBER, C.CONFIG_DEVIATION,
            C.CONFIG_PAPER_TRADING, C.CONFIG_MAX_SLIPPAGE_POINTS",
            C.CONFIG_KILL_SWITCH_FILE_PATH, C.CONFIG_KILL_SWITCH_CLOSE_POSITIONS, C.CONFIG_LOOP_INTERVAL
        ]},
        C.CONFIG_LOGGING: {"type": "object"},
        C.CONFIG_DEFAULTS: {"type": "object"},
        C.CONFIG_SYMBOLS: {"type": "object"},
        "timeframes": {"type": "object"}
        # Add more properties if needed for specific schema validation tests
    }
}


@pytest.fixture
def mock_config_manager_dependencies(valid_config_data):
    """Mocks dependencies for ConfigManager initialization and operation."""
    # Reset class schema to ensure it's reloaded with mock
    ConfigManager._schema = None

    # Mock for schema file
    mock_schema_file = mock_open(read_data=json.dumps(MOCK_SCHEMA_CONTENT))
    # Mock for main config file
    mock_main_config_file = mock_open(read_data=json.dumps(valid_config_data))

    # Path.exists will be True for schema, then for main config
    # Path.stat().st_mtime will return a fixed time
    with patch('core.config_manager.Path.exists', MagicMock(return_value=True)) as mock_exists, \
         patch('builtins.open', mock_main_config_file) as mock_open_main, \
         patch('core.config_manager.Path.stat') as mock_stat:

        # This is tricky: open is called first for schema, then for main config.
        # We need side_effect for `open` to handle this.
        # The first call to open (for schema) should use mock_schema_file.
        # The second call (for main config) should use mock_main_config_file.
        
        # Temporarily load schema directly using a separate mock for its open call
        with patch('builtins.open', mock_schema_file):
             ConfigManager._load_schema()
        
        # Now builtins.open is restored to mock_main_config_file for ConfigManager constructor

        mock_stat.return_value.st_mtime = time.time()

        cm = ConfigManager("dummy_config.json")
        # Attach mocks to the instance for tests to use/assert
        cm.mock_open_main = mock_open_main
        cm.mock_stat = mock_stat
        cm.mock_path_exists = mock_exists # General Path.exists, might need more specific for schema vs config
        yield cm # Yield the instance

    # Cleanup: Reset schema after tests using this fixture are done
    ConfigManager._schema = None


def test_config_manager_initialization_valid(mock_config_manager_dependencies):
    cm = mock_config_manager_dependencies
    assert cm is not None
    assert cm.get_global_settings()[C.CONFIG_PAPER_TRADING] is True

def test_schema_validation_success(mock_config_manager_dependencies, valid_config_data):
    # Initialization of the fixture itself tests successful validation
    assert mock_config_manager_dependencies._config == valid_config_data

def test_schema_validation_missing_required_global_setting(valid_config_data):
    invalid_data = valid_config_data.copy()
    # Ensure the key exists in global_settings before deleting for robust test
    if C.CONFIG_GLOBAL_SETTINGS not in invalid_data: invalid_data[C.CONFIG_GLOBAL_SETTINGS] = {}
    if C.CONFIG_PAPER_TRADING in invalid_data[C.CONFIG_GLOBAL_SETTINGS]:
        del invalid_data[C.CONFIG_GLOBAL_SETTINGS][C.CONFIG_PAPER_TRADING]

    ConfigManager._schema = None # Force schema reload with mock
    with patch('core.config_manager.Path.exists', MagicMock(return_value=True)), \
         patch('builtins.open', mock_open(read_data=json.dumps(MOCK_SCHEMA_CONTENT))), \
         patch('core.config_manager.Path.stat'): # Mock stat too
        ConfigManager._load_schema() # Load our MOCK_SCHEMA_CONTENT

    with patch('builtins.open', mock_open(read_data=json.dumps(invalid_data))), \
         patch('core.config_manager.Path.exists', MagicMock(return_value=True)), \
         patch('core.config_manager.Path.stat'), \
         pytest.raises(ConfigError) as excinfo:
        ConfigManager("dummy_invalid.json")
    assert "is a required property" in str(excinfo.value) or C.CONFIG_PAPER_TRADING in str(excinfo.value)
    ConfigManager._schema = None


def test_hot_reload_no_change(mock_config_manager_dependencies):
    cm = mock_config_manager_dependencies
    cm.clear_cache = MagicMock() # Mock clear_cache for assertion
    # Ensure st_mtime returns the same time as loaded
    cm.mock_stat.return_value.st_mtime = cm._last_modified_time
    reloaded, changes = cm.check_and_reload_config()
    assert not reloaded
    assert not changes
    cm.clear_cache.assert_not_called()

def test_hot_reload_file_modified_valid_changes(mock_config_manager_dependencies, valid_config_data):
    cm = mock_config_manager_dependencies
    cm.clear_cache = MagicMock()

    new_loop_interval = 5
    modified_data = json.loads(json.dumps(valid_config_data)) # Deep copy
    modified_data[C.CONFIG_GLOBAL_SETTINGS][C.CONFIG_LOOP_INTERVAL] = new_loop_interval
    
    cm.mock_stat.return_value.st_mtime = cm._last_modified_time + 10
    # Configure the mock for the main config file to return new data
    cm.mock_open_main.return_value.read.return_value = json.dumps(modified_data)

    reloaded, changes = cm.check_and_reload_config()

    assert reloaded
    assert changes[C.RELOAD_CHANGES_GLOBAL_SETTINGS][C.CONFIG_LOOP_INTERVAL] == new_loop_interval
    assert cm._config[C.CONFIG_GLOBAL_SETTINGS][C.CONFIG_LOOP_INTERVAL] == new_loop_interval
    cm.clear_cache.assert_called_once()

def test_hot_reload_file_modified_invalid_schema(mock_config_manager_dependencies, valid_config_data):
    cm = mock_config_manager_dependencies
    cm.clear_cache = MagicMock()
    original_config_copy = json.loads(json.dumps(cm._config)) # Deep copy of current config

    invalid_modified_data = json.loads(json.dumps(valid_config_data))
    # Make it invalid by removing a required field from global_settings for schema check
    del invalid_modified_data[C.CONFIG_GLOBAL_SETTINGS][C.CONFIG_PAPER_TRADING]

    cm.mock_stat.return_value.st_mtime = cm._last_modified_time + 10
    cm.mock_open_main.return_value.read.return_value = json.dumps(invalid_modified_data)

    reloaded, changes = cm.check_and_reload_config()

    assert not reloaded
    assert not changes
    # Check that the config was reverted to the original one
    assert cm._config[C.CONFIG_GLOBAL_SETTINGS][C.CONFIG_PAPER_TRADING] == original_config_copy[C.CONFIG_GLOBAL_SETTINGS][C.CONFIG_PAPER_TRADING]
    cm.clear_cache.assert_not_called()


def test_nonexistent_config_file(): # No fixture needed, tests constructor directly
    ConfigManager._schema = None # Ensure schema isn't pre-loaded from other tests
    with patch('core.config_manager.Path.exists', MagicMock(return_value=True)) as mock_schema_exists, \
         patch('builtins.open', mock_open(read_data=json.dumps(MOCK_SCHEMA_CONTENT))) as mock_schema_open:
        ConfigManager._load_schema()

    # Now, mock Path.exists to be False for the main config file
    with patch('core.config_manager.Path.exists', MagicMock(return_value=False)), \
         pytest.raises(ConfigError, match="Configuration file not found"):
        ConfigManager("nonexistent_config.json")
    ConfigManager._schema = None # Clean up

def test_invalid_json_format_config_file():
    ConfigManager._schema = None
    with patch('core.config_manager.Path.exists', MagicMock(return_value=True)), \
         patch('builtins.open', mock_open(read_data=json.dumps(MOCK_SCHEMA_CONTENT))):
        ConfigManager._load_schema()

    with patch('builtins.open', mock_open(read_data="{invalid_json_structure,")), \
         patch('core.config_manager.Path.exists', MagicMock(return_value=True)), \
         patch('core.config_manager.Path.stat'), \
         pytest.raises(ConfigError, match="Invalid JSON in config file"):
        ConfigManager("invalid_format.json")
    ConfigManager._schema = None
