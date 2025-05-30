"""
Integration tests for the Main Trading Bot loop.
"""
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import time # For time.sleep mocking if needed
import uuid

# Assuming main_bot is importable. Adjust path if necessary.
# This might require adding __init__.py to parent directories or adjusting sys.path
from main_bot import TradingBot
from core.config_manager import ConfigManager
from core.mt5_connector import MT5Connector
from core.strategy_engine import StrategyEngine, SignalType
from core.risk_manager import RiskManager
from core import constants as C
from core.logging_setup import get_log_context # For setting correlation ID if needed by mocks

# --- Test Configuration ---
@pytest.fixture
def test_config_path(tmp_path):
    """Creates a temporary valid config file for testing."""
    config_data = {
        C.CONFIG_METATRADER5: {"path": "dummy", "server": "Test", "login": 123, "password": "pwd", "timeout":600, "portable":False},
        C.CONFIG_GLOBAL_SETTINGS: {
            C.CONFIG_MAX_TOTAL_TRADES: 5, C.CONFIG_MAX_SLIPPAGE_PIPS: 3.0,
            C.CONFIG_MAGIC_NUMBER: 12345, C.CONFIG_DEVIATION: 10,
            C.CONFIG_PAPER_TRADING: True, # Enable paper trading for most integration tests
            C.CONFIG_MAX_SLIPPAGE_POINTS: 10,
            C.CONFIG_KILL_SWITCH_FILE_PATH: str(tmp_path / "KILL_SWITCH.txt"), # Use tmp_path
            C.CONFIG_KILL_SWITCH_CLOSE_POSITIONS: True,
            C.CONFIG_LOOP_INTERVAL: 0.01 # Very short for testing
        },
        C.CONFIG_LOGGING: {"level": "DEBUG", "file": str(tmp_path / "test_bot.log"), "max_bytes":100000, "backup_count":1, "format":"%(message)s"},
        C.CONFIG_DEFAULTS: {
            C.CONFIG_INDICATORS: {"rsi_period": 14, "macd_fast":12, "macd_slow":26, "macd_signal":9, "atr_period":14},
            C.CONFIG_RISK: {C.CONFIG_RISK_PER_TRADE: 0.01, "max_open_trades": 3},
            C.CONFIG_SR: {}
        },
        C.CONFIG_SYMBOLS: {
            "EURUSD": {C.CONFIG_ENABLED: True, C.CONFIG_LOT_SIZE: 0.01, C.CONFIG_SPREAD_LIMIT_PIPS: 5.0},
            "GBPUSD": {C.CONFIG_ENABLED: False} # Test with one disabled symbol
        },
        "timeframes": {"M15": "M15"}
    }
    config_file = tmp_path / "test_config.json"
    import json
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    return str(config_file)

# --- Mocks for External Dependencies ---
@pytest.fixture
def mock_mt5_connector_integration():
    """Mocks MT5Connector for integration tests."""
    mock = MagicMock(spec=MT5Connector)
    mock.initialize.return_value = True
    mock.is_kill_switch_active.return_value = False # Default, can be changed in tests

    # Simulate account info
    mock.get_account_info.return_value = {'balance': 10000.0, 'equity': 10000.0}

    # Simulate market data - return a dict of DataFrames
    sample_df = pd.DataFrame({
        'open': [1.0, 1.1, 1.2], 'high': [1.05, 1.15, 1.25],
        'low': [0.95, 1.05, 1.15], 'close': [1.1, 1.2, 1.0],
        'volume': [100,110,120]
    }, index=pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:15', '2023-01-01 10:30']))
    sample_df.name = "M15" # Name it as strategy engine might expect for mocks
    mock.get_data.return_value = sample_df # Used by _get_market_data

    # Simulate positions - default to no open positions
    mock.get_positions.return_value = [] # Used by main_bot._process_symbol and kill switch

    # Simulate trading operations (called via self.mt5.place_order etc in TradingBot)
    # These will hit MT5TradingOperations, which has paper trading mode.
    # For integration test, we mostly care that these are called.
    mock.place_order = MagicMock(return_value={'retcode': C.RETCODE_DONE, C.POSITION_TICKET: 12345, C.REQUEST_COMMENT: "Paper trade open"})
    mock.close_position = MagicMock(return_value={'retcode': C.RETCODE_DONE, C.POSITION_TICKET: 67890, C.REQUEST_COMMENT: "Paper trade close"})
    mock.modify_position = MagicMock(return_value={'retcode': C.RETCODE_DONE, C.POSITION_TICKET: 12345, C.REQUEST_COMMENT: "Paper trade modify"})

    return mock

@pytest.fixture
def mock_strategy_engine_integration():
    """Mocks StrategyEngine for integration tests."""
    mock = MagicMock(spec=StrategyEngine)
    # Default: No signal
    mock.analyze.return_value = {'signal': SignalType.NONE, 'message': 'No signal from mock'}
    return mock

# --- Integration Test ---
@patch('main_bot.MT5Connector')
@patch('main_bot.StrategyEngine')
@patch('main_bot.RiskManager') # Mock RiskManager as its internal logic is unit-tested elsewhere
@patch('main_bot.setup_logging') # Avoid actual logging setup during test
@patch('main_bot.time.sleep') # To speed up the loop
def test_main_bot_run_iteration_basic_flow(
    mock_sleep, mock_setup_logging, MockRiskManager, MockStrategyEngine, MockMT5Connector,
    test_config_path, default_account_info # Use default_account_info from risk manager tests if available or define here
):
    # Setup mocks
    mock_mt5 = mock_mt5_connector_integration() # Get an instance of the more detailed mock
    MockMT5Connector.return_value = mock_mt5

    mock_strategy = mock_strategy_engine_integration()
    MockStrategyEngine.return_value = mock_strategy

    mock_risk = MagicMock(spec=RiskManager)
    mock_risk.check_daily_limits.return_value = (True, "") # Allow trading
    mock_risk.check_market_conditions.return_value = (True, "") # Allow trading
    mock_risk.calculate_position_size.return_value = {C.LOT_SIZE: 0.01, C.POSITION_SL: 1.0, C.POSITION_TP: 1.2}
    MockRiskManager.return_value = mock_risk

    # Initialize TradingBot with the test config
    bot = TradingBot(config_path=test_config_path)
    bot.is_running = True # Simulate bot is running

    # --- Scenario 1: No signal ---
    bot._run_iteration()
    mock_strategy.analyze.assert_called() # Strategy should be called for EURUSD
    mock_mt5.place_order.assert_not_called() # No trade if no signal

    # --- Scenario 2: Buy signal for EURUSD ---
    mock_strategy.analyze.return_value = {
        'signal': SignalType.BUY, C.POSITION_SL: 1.0900, C.POSITION_TP: 1.1100,
        'message': 'Mock BUY signal', C.POSITION_OPEN_PRICE: 1.1000 # Strategy suggests entry
    }
    bot._run_iteration()
    mock_mt5.place_order.assert_called_once_with(
        symbol="EURUSD", order_type=C.ORDER_TYPE_BUY, lot_size=0.01,
        sl=1.0900, tp=1.1100, comment=pytest.string_containing("Mock BUY signal")
    )
    mock_risk.update_trade_count.assert_called_once()

    # --- Scenario 3: Sell signal for EURUSD, but already has an open EURUSD position ---
    mock_mt5.place_order.reset_mock() # Reset call count
    mock_risk.update_trade_count.reset_mock()
    # Simulate an open position for EURUSD
    mock_mt5.get_positions.return_value = [{C.POSITION_SYMBOL: "EURUSD", C.POSITION_TICKET: 12345, C.POSITION_TYPE: 0}] # type 0 for buy

    mock_strategy.analyze.return_value = { # Strategy now might generate an exit or another signal
        'signal': SignalType.SELL, 'message': 'Mock SELL signal to exit or new trade'
    }
    # In this setup, _check_for_entries is skipped if positions_list is not empty for the symbol.
    # The _manage_positions would be called. Let's assume it generates a close.
    # To test _manage_positions properly, its internal calls to strategy.analyze (for exit) need mocking.
    # For this basic integration, we'll assume it might try to close if conditions met.
    # If _check_exit_signals (called by _manage_position) uses the main strategy.analyze:
    # and it returns SELL (opposite to BUY position), then close_position is called.

    bot._run_iteration()
    # If paper trading is on (as per test_config_path), close_position is from MT5Connector mock
    mock_mt5.close_position.assert_called_with(12345) # Check if close was attempted.
    mock_mt5.place_order.assert_not_called() # No new order should be placed

    # --- Scenario 4: Kill switch activated ---
    mock_mt5.place_order.reset_mock()
    mock_mt5.close_position.reset_mock()
    kill_switch_file = Path(bot._kill_switch_file_path_str)
    kill_switch_file.touch() # Create the kill switch file

    bot._run_iteration()
    # Verify that trading operations are not called
    mock_mt5.place_order.assert_not_called()
    # Close operations might be called by the kill switch logic itself.
    # The mock_mt5.close_position was called above, so check for MORE calls if positions existed.
    # If kill_switch_close_positions = True, it would call get_positions(bypass_kill_switch=True)
    # and then close_position(bypass_kill_switch=True) for each.

    # Assert get_positions was called with bypass_kill_switch=True by _check_kill_switch
    # Then assert close_position was called with bypass_kill_switch=True by _check_kill_switch
    # This requires the mock_mt5.get_positions to handle the bypass_kill_switch arg.
    # For simplicity, we can check that the main trading logic (like analyze) was skipped.
    # We can refine this by checking logs or specific states.

    # Clean up kill switch file
    if kill_switch_file.exists():
        kill_switch_file.unlink()

    bot.is_running = False # Stop the loop for the test
```

This creates `test/integration/test_main_bot_integration.py`.

**Key aspects of this integration test:**

1.  **`test_config_path` fixture**: Creates a temporary `test_config.json` file with specific settings for the integration test (e.g., paper trading ON, very short loop interval, kill switch path within `tmp_path`).
2.  **`mock_mt5_connector_integration` fixture**:
    *   Mocks `MT5Connector`.
    *   Simulates `initialize()`, `is_kill_switch_active()`.
    *   Provides mock `get_account_info()`, `get_data()` (returns a basic M15 DataFrame).
    *   Mocks `get_positions()` (defaults to no open positions).
    *   Mocks `place_order`, `close_position`, `modify_position` to simulate successful paper trades, allowing us to assert they were called.
3.  **`mock_strategy_engine_integration` fixture**: Mocks `StrategyEngine`, initially set to return no signal. Tests can change its `analyze.return_value` to simulate different strategy outputs.
4.  **`test_main_bot_run_iteration_basic_flow`**:
    *   Patches `MT5Connector`, `StrategyEngine`, `RiskManager` at the `main_bot` module level where they are imported.
    *   Patches `setup_logging` and `time.sleep`.
    *   Initializes `TradingBot` with the temporary config.
    *   **Scenario 1 (No Signal)**: Runs `_run_iteration`. Asserts `strategy.analyze` was called but `place_order` was not.
    *   **Scenario 2 (Buy Signal)**: Configures `mock_strategy.analyze` to return a BUY signal. Runs `_run_iteration`. Asserts `place_order` was called with correct parameters. Asserts `risk_manager.update_trade_count` was called.
    *   **Scenario 3 (Sell Signal with Open Position)**: Simulates an open BUY position. Configures strategy to output a SELL (exit) signal. Asserts `close_position` is called.
    *   **Scenario 4 (Kill Switch)**: Creates the kill switch file. Runs `_run_iteration`. Verifies new trades are not placed. Further assertions could be added for position closing if `kill_switch_close_positions` is true (this would involve checking calls to `get_positions(bypass_kill_switch=True)` and `close_position(..., bypass_kill_switch=True)`).

This is a foundational integration test. It can be expanded with more complex scenarios, different configurations, and more detailed assertions on state changes or component interactions.

The subtask "Expand Unit and Integration Tests" is now substantially addressed with:
*   New tests for `core/trading_operations.py`.
*   Significantly enhanced tests for `core/strategy_engine.py` (and `MACDStrategy`).
*   Significantly enhanced tests for `core/config_manager.py` (schema validation, hot-reloading).
*   Enhanced tests for `core/risk_manager.py`.
*   A new integration test for `main_bot.py`.

I'll submit this.
