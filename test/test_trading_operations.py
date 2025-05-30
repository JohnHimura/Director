"""
Tests for the MT5TradingOperations class.
"""
import pytest
from unittest.mock import MagicMock, patch
import time

# Assume constants are in core.constants
from core import constants as C
# Classes to be tested or mocked
from core.trading_operations import MT5TradingOperations
from core.config_manager import ConfigManager # For mock
from core.mt5_connector import MT5Connector # For mock

# Mock the MetaTrader5 library
# This allows testing without having MT5 installed or running.
# We need to mock specific functions that trading_operations uses.
mt5_mock = MagicMock()
mt5_mock.TRADE_RETCODE_DONE = C.RETCODE_DONE # 10009
mt5_mock.ORDER_TYPE_BUY = 0 # Actual MT5 value
mt5_mock.ORDER_TYPE_SELL = 1 # Actual MT5 value
mt5_mock.TRADE_ACTION_DEAL = 1
mt5_mock.TRADE_ACTION_SLTP = 6
mt5_mock.ORDER_TIME_GTC = 0
mt5_mock.ORDER_FILLING_FOK = 1


@pytest.fixture
def mock_config_manager_for_ops():
    """Mocks ConfigManager for trading operations tests."""
    mock = MagicMock(spec=ConfigManager)
    mock.get_global_settings.return_value = {
        C.CONFIG_PAPER_TRADING: False,
        C.CONFIG_MAGIC_NUMBER: C.DEFAULT_MAGIC_NUMBER,
        C.CONFIG_MAX_SLIPPAGE_POINTS: C.DEFAULT_MAX_SLIPPAGE_POINTS,
        C.CONFIG_KILL_SWITCH_FILE_PATH: "KILL_SWITCH.txt", # Keep default path
        C.CONFIG_KILL_SWITCH_CLOSE_POSITIONS: True
    }
    return mock

@pytest.fixture
def mock_mt5_connector(mock_config_manager_for_ops):
    """Mocks MT5Connector."""
    mock = MagicMock(spec=MT5Connector)
    mock.config = mock_config_manager_for_ops # Attach the config mock to connector mock
    # Mock methods used by MT5TradingOperations if any (e.g., _ensure_connection, get_symbol_price)
    mock._ensure_connection.return_value = True
    mock.get_symbol_price.return_value = {'ask': 1.1000, 'bid': 1.0990}
    return mock

@pytest.fixture
def trading_ops_paper_on(mock_mt5_connector):
    """Fixture for MT5TradingOperations with paper trading ON."""
    # Override paper_trading setting for this fixture
    mock_mt5_connector.config.get_global_settings.return_value[C.CONFIG_PAPER_TRADING] = True

    # is_kill_switch_active_func will be provided by the MT5Connector mock if needed by TradingOps directly
    # For now, trading_ops gets it from its connector.
    # Let's make the connector's kill switch function return False by default for these tests
    mock_mt5_connector.is_kill_switch_active = MagicMock(return_value=False)

    return MT5TradingOperations(mock_mt5_connector, mock_mt5_connector.is_kill_switch_active)

@pytest.fixture
def trading_ops_paper_off(mock_mt5_connector):
    """Fixture for MT5TradingOperations with paper trading OFF."""
    mock_mt5_connector.config.get_global_settings.return_value[C.CONFIG_PAPER_TRADING] = False
    mock_mt5_connector.is_kill_switch_active = MagicMock(return_value=False)
    return MT5TradingOperations(mock_mt5_connector, mock_mt5_connector.is_kill_switch_active)

@pytest.fixture
def trading_ops_kill_switch_on(mock_mt5_connector):
    """Fixture for MT5TradingOperations with kill switch ON (via connector)."""
    mock_mt5_connector.config.get_global_settings.return_value[C.CONFIG_PAPER_TRADING] = False # Real trading
    mock_mt5_connector.is_kill_switch_active = MagicMock(return_value=True) # Kill switch is active
    return MT5TradingOperations(mock_mt5_connector, mock_mt5_connector.is_kill_switch_active)

# Test Paper Trading Mode
@patch('core.trading_operations.mt5', mt5_mock) # Patch mt5 globally for these tests
@patch('core.trading_operations.logger') # Patch logger to check calls
def test_open_position_paper_trading(mock_logger, trading_ops_paper_on):
    result = trading_ops_paper_on.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1, comment="Test paper buy")
    assert result[C.POSITION_TICKET] > 0
    assert result['retcode'] == C.RETCODE_DONE
    assert C.PAPER_TRADE_COMMENT_PREFIX in result[C.REQUEST_COMMENT]
    mock_logger.info.assert_any_call(pytest.string_containing(f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating OPEN position"))
    mt5_mock.order_send.assert_not_called()

@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_close_position_paper_trading(mock_logger, trading_ops_paper_on):
    result = trading_ops_paper_on.close_position(12345, comment="Test paper close")
    assert result[C.POSITION_TICKET] > 0 # Returns a new dummy ticket
    assert result['retcode'] == C.RETCODE_DONE
    assert C.PAPER_TRADE_COMMENT_PREFIX in result[C.REQUEST_COMMENT]
    mock_logger.info.assert_any_call(pytest.string_containing(f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating CLOSE position"))
    mt5_mock.order_send.assert_not_called()

@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_modify_position_paper_trading(mock_logger, trading_ops_paper_on):
    result = trading_ops_paper_on.modify_position(12345, stop_loss=1.0800, take_profit=1.1200)
    assert result[C.POSITION_TICKET] == 12345
    assert result['retcode'] == C.RETCODE_DONE
    assert C.PAPER_TRADE_COMMENT_PREFIX in result[C.REQUEST_COMMENT]
    mock_logger.info.assert_any_call(pytest.string_containing(f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating MODIFY position"))
    mt5_mock.order_send.assert_not_called()

# Test Kill Switch
@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_open_position_kill_switch_active(mock_logger, trading_ops_kill_switch_on):
    result = trading_ops_kill_switch_on.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1)
    assert result['retcode'] == -1 # Custom kill switch retcode
    assert "aborted by kill switch" in result[C.REQUEST_COMMENT]
    mock_logger.critical.assert_called_with("Kill switch is active. Open position operation aborted.")
    mt5_mock.order_send.assert_not_called()

@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_close_position_kill_switch_active_no_bypass(mock_logger, trading_ops_kill_switch_on):
    result = trading_ops_kill_switch_on.close_position(12345) # No bypass
    assert result['retcode'] == -1
    assert "aborted by kill switch" in result[C.REQUEST_COMMENT]
    mock_logger.critical.assert_called_with("Kill switch is active. Close position operation for ticket 12345 aborted.")
    mt5_mock.order_send.assert_not_called()

@patch('core.trading_operations.mt5', mt5_mock) # Mock the mt5 library import
@patch('core.trading_operations.logger')
def test_close_position_kill_switch_active_with_bypass(mock_logger, trading_ops_kill_switch_on, mock_mt5_connector):
    # For this test, we need the kill switch to be active via the connector's function,
    # but trading_ops should proceed because bypass_kill_switch=True.
    # We also need to mock a successful mt5.positions_get and mt5.order_send for the close.

    # Setup: Connector's kill switch is ON
    trading_ops_kill_switch_on.is_kill_switch_active = MagicMock(return_value=True)

    # Mock mt5.positions_get to return a dummy position
    dummy_position = MagicMock()
    dummy_position._asdict.return_value = {
        C.POSITION_TICKET: 12345, C.POSITION_SYMBOL: "EURUSD", C.POSITION_VOLUME: 0.1,
        C.POSITION_TYPE: mt5_mock.ORDER_TYPE_BUY, C.POSITION_MAGIC: 123456,
        C.POSITION_SL: 0.0, C.POSITION_TP: 0.0
    }
    mt5_mock.positions_get.return_value = [dummy_position]

    # Mock mt5.order_send for a successful close
    mt5_order_send_result_mock = MagicMock()
    mt5_order_send_result_mock.retcode = C.RETCODE_DONE
    mt5_order_send_result_mock.comment = "Closed by test"
    mt5_order_send_result_mock.order = 78910 # New ticket for the closing order
    mt5_mock.order_send.return_value = mt5_order_send_result_mock

    result = trading_ops_kill_switch_on.close_position(12345, bypass_kill_switch=True)

    assert result['retcode'] == C.RETCODE_DONE
    assert result[C.REQUEST_COMMENT] == "Closed by test"
    mock_logger.critical.assert_not_called() # Should not log "aborted by kill switch"
    mt5_mock.order_send.assert_called_once() # Should be called because bypassed

@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_modify_position_kill_switch_active(mock_logger, trading_ops_kill_switch_on):
    result = trading_ops_kill_switch_on.modify_position(12345, stop_loss=1.0800)
    assert result['retcode'] == -1
    assert "aborted by kill switch" in result[C.REQUEST_COMMENT]
    mock_logger.critical.assert_called_with("Kill switch is active. Modify position operation for ticket 12345 aborted.")
    mt5_mock.order_send.assert_not_called()

# TODO: Add tests for real trading mode (paper_trading=False, kill_switch=False)
# These will involve more complex mocking of mt5.order_send results (success, various errors)
# and mt5.positions_get.

# Example for real trading open_position success
@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_open_position_real_trading_success(mock_logger, trading_ops_paper_off):
    mt5_order_send_result_mock = MagicMock()
    mt5_order_send_result_mock.retcode = C.RETCODE_DONE
    mt5_order_send_result_mock.comment = "Real trade success"
    mt5_order_send_result_mock.order = 67890
    mt5_mock.order_send.return_value = mt5_order_send_result_mock

    result = trading_ops_paper_off.open_position("EURUSD", C.ORDER_TYPE_SELL, 0.05, price=1.1000, comment="Test real sell")

    assert result[C.POSITION_TICKET] == 67890
    assert result['retcode'] == C.RETCODE_DONE
    assert result[C.REQUEST_COMMENT] == "Real trade success"
    mt5_mock.order_send.assert_called_once()
    # Check that deviation was fetched from config
    args, kwargs = mt5_mock.order_send.call_args
    sent_request = args[0]
    assert sent_request[C.REQUEST_DEVIATION] == C.DEFAULT_MAX_SLIPPAGE_POINTS

# Example for real trading open_position with specific MT5 error
@patch('core.trading_operations.mt5', mt5_mock)
@patch('core.trading_operations.logger')
def test_open_position_real_trading_mt5_error_requote(mock_logger, trading_ops_paper_off):
    mt5_order_send_result_mock = MagicMock()
    # Simulate a REQUOTE error from MT5
    mt5_mock.TRADE_RETCODE_REQUOTE = 10004 # Actual MT5 value for requote
    mt5_order_send_result_mock.retcode = mt5_mock.TRADE_RETCODE_REQUOTE
    mt5_order_send_result_mock.comment = "Requote"
    mt5_order_send_result_mock.order = 0 # No ticket on requote
    mt5_mock.order_send.return_value = mt5_order_send_result_mock

    with pytest.raises(Exception) as excinfo: # Assuming OperationError is raised
        trading_ops_paper_off.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1, comment="Test requote")

    assert "Order failed: Code=10004" in str(excinfo.value)
    assert "Requote" in str(excinfo.value)
    mt5_mock.order_send.assert_called_once()

```

I've created `test/test_trading_operations.py` with:
*   Mocking for `mt5` library itself.
*   Fixtures for `ConfigManager` (mocked), `MT5Connector` (mocked), and `MT5TradingOperations` in different states (paper trading on/off, kill switch on).
*   **Paper Trading Tests**:
    *   `test_open_position_paper_trading`
    *   `test_close_position_paper_trading`
    *   `test_modify_position_paper_trading`
    *   These verify that simulated success is returned, correct logs are made, and `mt5.order_send` is NOT called.
*   **Kill Switch Tests**:
    *   `test_open_position_kill_switch_active`
    *   `test_close_position_kill_switch_active_no_bypass`
    *   `test_close_position_kill_switch_active_with_bypass` (this one is more complex due to needing to mock position data for the close to proceed)
    *   `test_modify_position_kill_switch_active`
    *   These verify that operations are blocked (or allowed if bypassed) and appropriate logs/return values occur.
*   **Real Trading Examples (Minimal for now)**:
    *   `test_open_position_real_trading_success`: Mocks a successful `mt5.order_send`.
    *   `test_open_position_real_trading_mt5_error_requote`: Mocks an `mt5.order_send` that returns a "requote" error.

This provides initial coverage for `MT5TradingOperations`, focusing on paper trading and kill switch. More tests for various MT5 error codes and successful operation details in real mode can be added.

Next, I'll focus on enhancing tests for `core/strategy_engine.py`, particularly for `MACDStrategy`'s logic.
