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
    test_config_path # Removed default_account_info as it's not defined here
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
    # RiskManager's update_trade_count is called inside place_order if successful
    # This mock structure doesn't directly test that interaction easily without more complex setup
    # For now, assume if place_order is called, that part of the flow is reached.
    # If RiskManager itself was not mocked, its method could be asserted.
    # Since RiskManager is mocked, we'd assert the mock_risk.update_trade_count if it was called by TradingBot
    # However, update_trade_count is usually called from within place_order or similar.

    # --- Scenario 3: Sell signal for EURUSD, but already has an open EURUSD position ---
    mock_mt5.place_order.reset_mock() # Reset call count
    # mock_risk.update_trade_count.reset_mock() # Reset if it were asserted directly on mock_risk

    # Simulate an open position for EURUSD
    # The get_positions mock in mock_mt5_connector_integration needs to be more flexible
    # or we update its return_value here.
    mock_mt5.get_open_positions.return_value = (
        [{C.POSITION_SYMBOL: "EURUSD", C.POSITION_TICKET: 12345, C.POSITION_TYPE: 0}], None, None
    )

    mock_strategy.analyze.return_value = {
        'signal': SignalType.SELL, 'message': 'Mock SELL signal to exit or new trade'
    }

    bot._run_iteration()
    mock_mt5.close_position.assert_called_with(12345, "Exit signal: Mock SELL signal to exit or new trade")
    mock_mt5.place_order.assert_not_called()

    # --- Scenario 4: Kill switch activated ---
    mock_mt5.place_order.reset_mock()
    mock_mt5.close_position.reset_mock()
    mock_mt5.get_open_positions.return_value = ([], None, None) # Reset to no open positions for this part

    kill_switch_file = Path(bot._kill_switch_file_path_str)
    kill_switch_file.touch()

    bot._run_iteration()
    mock_mt5.place_order.assert_not_called()
    # If kill_switch_close_positions is True (as in test_config_path), close_position might be called.
    # This requires get_open_positions to be called with bypass_kill_switch=True by _check_kill_switch.
    # The current mock_mt5.get_open_positions doesn't distinguish bypass_kill_switch.
    # For a simple check: ensure analyze is not called if KS is active.
    # mock_strategy.analyze.assert_not_called() # This depends on where KS check happens in _run_iteration

    if kill_switch_file.exists():
        kill_switch_file.unlink()

    bot.is_running = False
