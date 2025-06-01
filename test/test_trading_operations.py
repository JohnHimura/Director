import pytest
from unittest.mock import MagicMock, call # call is not used yet but might be

from core.trading_operations import MT5TradingOperations
from core.mt5_connector import MT5Connector
from core.config_manager import ConfigManager
from core import constants as C
# Import MockMT5 for type hinting and direct configuration in tests
from test.mocks.mock_mt5 import MockMT5, TRADE_RETCODE_DONE, ORDER_TYPE_BUY, ORDER_TYPE_SELL, TRADE_ACTION_DEAL, TRADE_ACTION_SLTP

# Constants for MT5 that might be used if not importing from MockMT5 directly in every test
# However, MockMT5 provides these as class attributes which is convenient.

@pytest.fixture
def mock_config_for_ops():
    """Provides a mock ConfigManager, focused on settings relevant to TradingOperations."""
    mock = MagicMock(spec=ConfigManager)
    # Default to real trading; tests can override this by modifying the return_value
    mock.get_global_settings.return_value = {
        C.CONFIG_PAPER_TRADING: False,
        C.CONFIG_MAGIC_NUMBER: C.DEFAULT_MAGIC_NUMBER,
        C.CONFIG_MAX_SLIPPAGE_POINTS: C.DEFAULT_MAX_SLIPPAGE_POINTS,
    }
    # MT5 config is needed by MT5Connector
    mock.get_mt5_config.return_value = {
        C.CONFIG_MT5_PATH: "dummy_path", C.CONFIG_MT5_SERVER: "dummy_server",
        C.CONFIG_MT5_LOGIN: 12345, C.CONFIG_MT5_PASSWORD: "pwd",
        C.CONFIG_MT5_TIMEOUT: 60000, C.CONFIG_MT5_PORTABLE: False,
        C.CONFIG_MT5_CONNECTION_MAX_RETRIES: 1,
        C.CONFIG_MT5_CONNECTION_RETRY_DELAY: 0.01 # Fast retries for tests
    }
    return mock

@pytest.fixture
def mt5_connector_with_mock_lib(mock_config_for_ops, mock_mt5_instance: MockMT5):
    """
    Provides a real MT5Connector instance that uses the MockMT5 library instance
    (due to patching by mock_mt5_instance fixture from conftest.py).
    """
    # mock_mt5_instance from conftest has already patched 'core.mt5_connector.mt5'
    # So, when MT5Connector() is created, it will use MockMT5.
    connector = MT5Connector(config=mock_config_for_ops, is_kill_switch_active_func=lambda: False)
    # Ensure the connector uses our specific mock_mt5_instance if the patching was general
    # This is usually handled by monkeypatch in conftest.py by patching the mt5 import
    # where MT5Connector looks for it.

    # Allow tests to access the underlying MockMT5 instance via the connector if needed,
    # though ideally, they configure mock_mt5_instance directly.
    connector.mt5_lib = mock_mt5_instance # For direct access if a test needs to verify calls on the lib used by connector
    return connector

@pytest.fixture
def trading_ops_paper_on(mt5_connector_with_mock_lib: MT5Connector):
    mt5_connector_with_mock_lib.config_manager.get_global_settings.return_value[C.CONFIG_PAPER_TRADING] = True
    return MT5TradingOperations(mt5_connector_with_mock_lib, mt5_connector_with_mock_lib.is_kill_switch_active)

@pytest.fixture
def trading_ops_paper_off(mt5_connector_with_mock_lib: MT5Connector):
    mt5_connector_with_mock_lib.config_manager.get_global_settings.return_value[C.CONFIG_PAPER_TRADING] = False
    return MT5TradingOperations(mt5_connector_with_mock_lib, mt5_connector_with_mock_lib.is_kill_switch_active)

@pytest.fixture
def trading_ops_kill_switch_on(mt5_connector_with_mock_lib: MT5Connector):
    # Modify the connector's kill switch function for this specific trading_ops instance
    mt5_connector_with_mock_lib.is_kill_switch_active = lambda: True
    ops = MT5TradingOperations(mt5_connector_with_mock_lib, mt5_connector_with_mock_lib.is_kill_switch_active)
    return ops


# Test Paper Trading Mode
def test_open_position_paper_trading(trading_ops_paper_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    result = trading_ops_paper_on.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1, comment="Test paper buy")
    assert result[C.POSITION_TICKET] > 0
    assert result['retcode'] == TRADE_RETCODE_DONE
    assert C.PAPER_TRADE_COMMENT_PREFIX in result[C.REQUEST_COMMENT]
    assert f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating OPEN position" in caplog.text
    mock_mt5_instance.order_send.assert_not_called()

def test_close_position_paper_trading(trading_ops_paper_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    result = trading_ops_paper_on.close_position(12345, comment="Test paper close")
    assert result[C.POSITION_TICKET] > 0
    assert result['retcode'] == TRADE_RETCODE_DONE
    assert C.PAPER_TRADE_COMMENT_PREFIX in result[C.REQUEST_COMMENT]
    assert f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating CLOSE position" in caplog.text
    mock_mt5_instance.order_send.assert_not_called()

def test_modify_position_paper_trading(trading_ops_paper_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    result = trading_ops_paper_on.modify_position(12345, stop_loss=1.0800, take_profit=1.1200)
    assert result[C.POSITION_TICKET] == 12345
    assert result['retcode'] == TRADE_RETCODE_DONE
    assert C.PAPER_TRADE_COMMENT_PREFIX in result[C.REQUEST_COMMENT]
    assert f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating MODIFY position" in caplog.text
    mock_mt5_instance.order_send.assert_not_called()

# Test Kill Switch
def test_open_position_kill_switch_active(trading_ops_kill_switch_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    result = trading_ops_kill_switch_on.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1)
    assert result['retcode'] == -1
    assert "aborted by kill switch" in result[C.REQUEST_COMMENT]
    assert "Kill switch is active. Open position operation aborted." in caplog.text
    mock_mt5_instance.order_send.assert_not_called()

def test_close_position_kill_switch_active_no_bypass(trading_ops_kill_switch_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    result = trading_ops_kill_switch_on.close_position(12345) # No bypass
    assert result['retcode'] == -1
    assert "aborted by kill switch" in result[C.REQUEST_COMMENT]
    assert "Kill switch is active. Close position operation for ticket 12345 aborted." in caplog.text
    mock_mt5_instance.order_send.assert_not_called()

def test_close_position_kill_switch_active_with_bypass(trading_ops_kill_switch_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    ticket_to_close = 12345
    symbol = "EURUSD"
    # Setup MockMT5 to return an open position
    mock_mt5_instance.add_open_position({
        'ticket': ticket_to_close, 'symbol': symbol, 'volume': 0.1,
        'type': ORDER_TYPE_BUY, 'price_open': 1.1000, 'magic': 12345
    })
    # Mock get_symbol_price used by TradingOps for closing price
    trading_ops_kill_switch_on.connector.get_symbol_price.return_value = {'bid': 1.1010, 'ask': 1.1012}
    # Configure a successful order_send response from MockMT5
    mock_mt5_instance.order_send_should_fail = False
    mock_mt5_instance.order_send.return_value = MagicMock(
        retcode=TRADE_RETCODE_DONE, comment="Closed by test (bypass)", order=78910, profit=10.0
    )

    result = trading_ops_kill_switch_on.close_position(ticket_to_close, bypass_kill_switch=True)

    assert result['retcode'] == TRADE_RETCODE_DONE
    assert result[C.REQUEST_COMMENT] == "Closed by test (bypass)"
    # Ensure critical log about KS abortion is NOT present for this call
    for record in caplog.records:
        if record.levelname == "CRITICAL" and "aborted by kill switch" in record.message:
            assert False, "Critical kill switch log found despite bypass"
    mock_mt5_instance.order_send.assert_called_once()

def test_modify_position_kill_switch_active(trading_ops_kill_switch_on: MT5TradingOperations, mock_mt5_instance: MockMT5, caplog):
    result = trading_ops_kill_switch_on.modify_position(12345, stop_loss=1.0800)
    assert result['retcode'] == -1
    assert "aborted by kill switch" in result[C.REQUEST_COMMENT]
    assert "Kill switch is active. Modify position operation for ticket 12345 aborted." in caplog.text
    mock_mt5_instance.order_send.assert_not_called()

# Test Real Trading Mode
def test_open_position_real_trading_success(trading_ops_paper_off: MT5TradingOperations, mock_mt5_instance: MockMT5):
    symbol = "EURUSD"
    order_type_enum = ORDER_TYPE_SELL
    volume = 0.05
    price = 1.1000
    expected_order_ticket = 67890

    mock_mt5_instance.order_send_should_fail = False
    # MockMT5's order_send will now use its internal logic to generate a result
    # We can pre-configure symbol state if needed for price fetching by order_send
    mock_mt5_instance.add_symbol_info(symbol, {'ask': 1.1002, 'bid': 1.1000, 'point': 0.00001, 'trade_contract_size': 100000})

    result = trading_ops_paper_off.open_position(symbol, C.ORDER_TYPE_SELL, volume, price=price, comment="Test real sell")

    # The ticket ID is now generated by MockMT5's order_send
    assert result[C.POSITION_TICKET] > 0 # Check if a ticket was assigned
    assert result['retcode'] == TRADE_RETCODE_DONE
    mock_mt5_instance.order_send.assert_called_once()
    request_arg = mock_mt5_instance.order_send.call_args[0][0]
    assert request_arg[C.REQUEST_DEVIATION] == C.DEFAULT_MAX_SLIPPAGE_POINTS # Check config usage

def test_open_position_real_trading_mt5_error_requote(trading_ops_paper_off: MT5TradingOperations, mock_mt5_instance: MockMT5):
    # Configure MockMT5 to simulate a requote error
    mock_mt5_instance.order_send_should_fail = True
    mock_mt5_instance.order_send_fail_retcode = MockMT5.TRADE_RETCODE_REQUOTE # Use constant from MockMT5
    mock_mt5_instance.order_send_fail_message = "Requote"

    with pytest.raises(Exception) as excinfo: # MT5TradingOperations raises OperationError
        trading_ops_paper_off.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1, comment="Test requote")

    assert "Order failed: Code=10004" in str(excinfo.value) # 10004 is requote
    assert "Requote" in str(excinfo.value)
    mock_mt5_instance.order_send.assert_called_once()

def test_open_position_real_trading_general_error(trading_ops_paper_off: MT5TradingOperations, mock_mt5_instance: MockMT5):
    # Configure MockMT5 to simulate a generic error
    mock_mt5_instance.order_send_should_fail = True
    mock_mt5_instance.order_send_fail_retcode = 10008 # TRADE_RETCODE_ERROR or some other generic error
    mock_mt5_instance.order_send_fail_message = "Generic MT5 Error"

    mock_mt5_instance.add_symbol_info("EURUSD", {'ask': 1.1002, 'bid': 1.1000, 'point': 0.00001, 'trade_contract_size': 100000})

    with pytest.raises(Exception) as excinfo: # MT5TradingOperations raises OperationError
        trading_ops_paper_off.open_position("EURUSD", C.ORDER_TYPE_BUY, 0.1, comment="Test generic error")

    assert f"Order failed: Code={mock_mt5_instance.order_send_fail_retcode}" in str(excinfo.value)
    assert mock_mt5_instance.order_send_fail_message in str(excinfo.value)
    mock_mt5_instance.order_send.assert_called_once()

def test_close_position_real_trading_profit_calculation(trading_ops_paper_off: MT5TradingOperations, mock_mt5_instance: MockMT5, mt5_connector_with_mock_lib: MT5Connector):
    ticket_to_close = 12345
    symbol = "EURUSD"
    open_price = 1.10000
    close_price = 1.10100 # 10 pips profit for a BUY
    volume = 0.1
    expected_profit = 10.0 # Example profit

    # Setup MockMT5 to return an open position
    mock_mt5_instance.add_open_position({
        'ticket': ticket_to_close, 'symbol': symbol, 'volume': volume,
        'type': ORDER_TYPE_BUY, 'price_open': open_price, 'magic': 12345
    })

    # Mock get_symbol_price used by TradingOps for determining closing price for request
    # The connector instance used by trading_ops_paper_off is mt5_connector_with_mock_lib
    mt5_connector_with_mock_lib.get_symbol_price.return_value = {'bid': close_price, 'ask': close_price + 0.00002} # Closing a BUY uses BID

    # Configure MockMT5 for successful closure order_send
    closing_order_ticket = 78910
    mock_mt5_instance.order_send_should_fail = False
    mock_mt5_instance.order_send.return_value = MagicMock(
        retcode=TRADE_RETCODE_DONE, comment="Position closed", order=closing_order_ticket
    )

    # Configure MockMT5 history_deals_get to return a deal with profit
    mock_deal = MagicMock()
    mock_deal.profit = expected_profit
    mock_deal.commission = -0.1 # example
    mock_deal.swap = 0.0
    mock_deal.order = closing_order_ticket # Link deal to the closing order
    mock_mt5_instance.history_deals_get.return_value = [mock_deal]

    # Mock time.sleep in trading_operations if it's called
    with patch('core.trading_operations.time.sleep'): # Patch time.sleep where it's used
        result = trading_ops_paper_off.close_position(ticket_to_close, comment="test close with profit")

    assert result['retcode'] == TRADE_RETCODE_DONE
    assert result['profit'] == expected_profit + mock_deal.commission + mock_deal.swap
    mock_mt5_instance.order_send.assert_called_once()
    mock_mt5_instance.history_deals_get.assert_called_once_with(order=closing_order_ticket)

```
