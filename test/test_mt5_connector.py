import pytest
from unittest.mock import MagicMock, call
import time # For checking sleep calls if necessary

from core.mt5_connector import MT5Connector, MT5ConnectionError
from core.config_manager import ConfigManager
from core import constants as C
from test.mocks.mock_mt5 import MockMT5 # Import the mock directly for type hinting if needed, though fixture provides it

# Constants for MT5 credentials and settings from environment/config
ENV_MT5_PATH = "/path/to/terminal64.exe"
ENV_MT5_SERVER = "TestServer"
ENV_MT5_LOGIN = 12345678
ENV_MT5_PASSWORD = "password"
CONFIG_TIMEOUT = 60000
CONFIG_MAX_RETRIES = 3
CONFIG_RETRY_DELAY = 0.1 # Use a small delay for tests

@pytest.fixture
def mock_config_for_connector(mocker):
    """Provides a mock ConfigManager for MT5Connector tests."""
    mock_cm = MagicMock(spec=ConfigManager)
    mt5_config_data = {
        C.CONFIG_MT5_PATH: ENV_MT5_PATH,
        C.CONFIG_MT5_SERVER: ENV_MT5_SERVER,
        C.CONFIG_MT5_LOGIN: ENV_MT5_LOGIN,
        C.CONFIG_MT5_PASSWORD: ENV_MT5_PASSWORD,
        C.CONFIG_MT5_TIMEOUT: CONFIG_TIMEOUT,
        C.CONFIG_MT5_PORTABLE: False,
        C.CONFIG_MT5_CONNECTION_MAX_RETRIES: CONFIG_MAX_RETRIES,
        C.CONFIG_MT5_CONNECTION_RETRY_DELAY: CONFIG_RETRY_DELAY # Using the constant for retry delay seconds
    }
    mock_cm.get_mt5_config.return_value = mt5_config_data
    return mock_cm

def test_mt5connector_init_successful_first_attempt(mock_config_for_connector, mock_mt5_instance: MockMT5):
    """Test successful initialization and connection on the first attempt."""
    mock_mt5_instance.should_initialize_fail = False
    mock_mt5_instance.should_login_fail = False

    connector = MT5Connector(config=mock_config_for_connector)

    assert connector.initialized
    assert connector.connected
    assert connector.trading is not None
    mock_mt5_instance.initialize.assert_called_once_with(path=ENV_MT5_PATH, timeout=CONFIG_TIMEOUT, portable=False)
    mock_mt5_instance.login.assert_called_once_with(login=ENV_MT5_LOGIN, password=ENV_MT5_PASSWORD, server=ENV_MT5_SERVER, timeout=CONFIG_TIMEOUT)

def test_mt5connector_init_fail_all_retries(mock_config_for_connector, mock_mt5_instance: MockMT5, mocker):
    """Test connection failure after exhausting all retries."""
    mock_mt5_instance.should_initialize_fail = True # Make initialize always fail
    mock_mt5_instance.set_last_error(1, "Mocked initialize failure")

    # Mock time.sleep to avoid actual delays during test
    mock_sleep = mocker.patch('time.sleep')

    connector = MT5Connector(config=mock_config_for_connector)

    assert not connector.initialized
    assert not connector.connected
    assert connector.trading is None
    # Total attempts = max_retries + 1
    assert mock_mt5_instance.initialize.call_count == CONFIG_MAX_RETRIES + 1
    mock_mt5_instance.login.assert_not_called() # Login should not be attempted if initialize fails

    # Check if sleep was called with exponential backoff (max_retries times)
    assert mock_sleep.call_count == CONFIG_MAX_RETRIES
    expected_delays = [CONFIG_RETRY_DELAY * (2 ** i) for i in range(CONFIG_MAX_RETRIES)]
    for i, expected_delay in enumerate(expected_delays):
        assert mock_sleep.call_args_list[i] == call(expected_delay)

def test_mt5connector_init_login_fails_then_succeeds(mock_config_for_connector, mock_mt5_instance: MockMT5, mocker):
    """Test connection succeeding after a few login failures."""
    # Simulate initialize succeeding, but login failing twice then succeeding
    mock_mt5_instance.should_initialize_fail = False
    mock_mt5_instance.login.side_effect = [
        False, # First login attempt fails
        False, # Second login attempt fails
        True   # Third login attempt succeeds
    ]
    # Configure last_error for login failures
    # This is tricky because last_error is global to the mock.
    # We assume it's set correctly by the mock's login method upon failure.
    # For more complex scenarios, mock_mt5_instance.last_error could be a MagicMock itself.

    mock_sleep = mocker.patch('time.sleep')

    connector = MT5Connector(config=mock_config_for_connector)

    assert connector.initialized
    assert connector.connected
    assert connector.trading is not None

    # Initialize should be called potentially multiple times if login fails and forces re-init attempt
    # Current _initialize logic: if login fails, it sets self.initialized = False and retries the whole sequence.
    assert mock_mt5_instance.initialize.call_count == 3 # init -> login fail -> init -> login fail -> init -> login success
    assert mock_mt5_instance.login.call_count == 3

    # Check sleep calls (it will sleep twice before the third successful attempt)
    assert mock_sleep.call_count == 2
    expected_delays = [CONFIG_RETRY_DELAY * (2**0), CONFIG_RETRY_DELAY * (2**1)]
    assert mock_sleep.call_args_list[0] == call(expected_delays[0])
    assert mock_sleep.call_args_list[1] == call(expected_delays[1])

def test_is_connected_live_check(mock_config_for_connector, mock_mt5_instance: MockMT5):
    """Test the is_connected method for live checks."""
    connector = MT5Connector(config=mock_config_for_connector) # Initial connection
    assert connector.is_connected() # Should be true after successful init

    # Simulate a scenario where terminal_info starts returning None (connection dropped)
    mock_mt5_instance.mock_terminal_info_obj.connected = False # Simulate terminal disconnect
    mock_mt5_instance.terminal_info.return_value = None # More direct way to signal terminal issue
    assert not connector.is_connected()
    assert not connector.connected # Flag should be updated by is_connected

    # Simulate terminal info OK, but account info fails
    mock_mt5_instance.terminal_info.return_value = mock_mt5_instance.mock_terminal_info_obj # Terminal responsive
    mock_mt5_instance.mock_terminal_info_obj.connected = True
    mock_mt5_instance.account_info.return_value = None # Login/account issue
    assert not connector.is_connected()
    assert not connector.connected

    # Simulate both OK again
    mock_mt5_instance.account_info.return_value = mock_mt5_instance.mock_account_info_obj
    assert connector.is_connected()
    assert connector.connected


def test_check_connection_and_reconnect_when_already_connected(mock_config_for_connector, mock_mt5_instance: MockMT5):
    """Test check_connection_and_reconnect when already connected."""
    connector = MT5Connector(config=mock_config_for_connector) # Connects on init

    # Ensure _initialize is not called again if already connected
    mock_mt5_instance.initialize.reset_mock()
    mock_mt5_instance.login.reset_mock()

    assert connector.check_connection_and_reconnect()
    mock_mt5_instance.initialize.assert_not_called()
    mock_mt5_instance.login.assert_not_called()

def test_check_connection_and_reconnect_when_disconnected_success(mock_config_for_connector, mock_mt5_instance: MockMT5, mocker):
    """Test check_connection_and_reconnect successfully reconnects."""
    # Initial connection
    connector = MT5Connector(config=mock_config_for_connector)
    assert connector.connected

    # Simulate disconnection
    mock_mt5_instance.mock_terminal_info_obj.connected = False
    mock_mt5_instance.terminal_info.return_value = None # is_connected will detect this

    # Configure _initialize to succeed on the next call (which check_connection_and_reconnect will trigger)
    # Reset side effects or failure flags on mock_mt5_instance if they were set by previous tests in other scopes
    mock_mt5_instance.should_initialize_fail = False
    mock_mt5_instance.should_login_fail = False
    mock_mt5_instance.initialize.reset_mock() # Reset call count from initial __init__
    mock_mt5_instance.login.reset_mock()

    mock_sleep = mocker.patch('time.sleep') # In case _initialize retries during reconnect

    assert connector.check_connection_and_reconnect() # This should trigger _initialize
    assert connector.connected
    assert connector.initialized
    mock_mt5_instance.initialize.assert_called_once() # Called once by check_connection_and_reconnect
    mock_mt5_instance.login.assert_called_once()


def test_ensure_connection_raises_error_if_reconnect_fails(mock_config_for_connector, mock_mt5_instance: MockMT5, mocker):
    """Test that _ensure_connection raises MT5ConnectionError if reconnection fails."""
    # Initial connection can succeed or fail, for this test, let's assume it failed initially
    # or a subsequent call finds it disconnected and reconnection also fails.
    mock_mt5_instance.should_initialize_fail = True # Make all _initialize attempts fail
    mock_mt5_instance.set_last_error(1, "Persistent init failure")
    mocker.patch('time.sleep') # Mock sleep to speed up retries

    # Initialize connector - this will try to connect and fail
    connector = MT5Connector(config=mock_config_for_connector)
    assert not connector.connected

    # Now, any method calling _ensure_connection should raise an error
    with pytest.raises(MT5ConnectionError, match="MT5 reconnection failed. Cannot proceed with operation."):
        connector._ensure_connection()

    # Verify _initialize was called multiple times (initial attempts + _ensure_connection attempt)
    # Initial call in __init__ will do MAX_RETRIES + 1 attempts.
    # _ensure_connection -> check_connection_and_reconnect -> _initialize will do another MAX_RETRIES + 1 attempts.
    assert mock_mt5_instance.initialize.call_count == (CONFIG_MAX_RETRIES + 1) * 2


def test_trading_methods_call_ensure_connection(mock_config_for_connector, mock_mt5_instance: MockMT5, mocker):
    """Test that trading methods call _ensure_connection."""
    connector = MT5Connector(config=mock_config_for_connector)
    assert connector.connected # Should be connected after init

    # Mock _ensure_connection to track calls
    mocker.patch.object(connector, '_ensure_connection', return_value=True) # Ensure it thinks it's connected

    # Mock the actual trading operations method on the trading object
    # Connector.trading is an instance of MT5TradingOperations
    mock_trading_ops_open = mocker.patch.object(connector.trading, 'open_position', return_value={'retcode': C.TRADE_RETCODE_DONE})
    mock_trading_ops_close = mocker.patch.object(connector.trading, 'close_position', return_value={'retcode': C.TRADE_RETCODE_DONE})
    mock_trading_ops_modify = mocker.patch.object(connector.trading, 'modify_position', return_value={'retcode': C.TRADE_RETCODE_DONE})

    connector.place_order(symbol="EURUSD", order_type="BUY", volume=0.1)
    connector._ensure_connection.assert_called_once()
    mock_trading_ops_open.assert_called_once()
    connector._ensure_connection.reset_mock() # Reset for next call

    connector.close_position(ticket=123)
    connector._ensure_connection.assert_called_once()
    mock_trading_ops_close.assert_called_once()
    connector._ensure_connection.reset_mock()

    connector.modify_position(ticket=123, sl=1.0)
    connector._ensure_connection.assert_called_once()
    mock_trading_ops_modify.assert_called_once()

def test_get_data_renamed_from_get_history_data(mock_config_for_connector, mock_mt5_instance: MockMT5):
    """ Test that get_data (renamed from get_history_data) works and calls _ensure_connection. """
    connector = MT5Connector(config=mock_config_for_connector)

    sample_data = pd.DataFrame({
        'time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:01:00']),
        'open': [1.0, 1.1], 'high': [1.2, 1.3], 'low': [0.9, 1.0], 'close': [1.1, 1.2],
        'tick_volume': [100, 120], 'spread': [2,2], 'real_volume': [0,0]
    })
    # Mock the direct mt5.copy_rates_from_pos call as get_data now uses it directly
    mock_mt5_instance.copy_rates_from_pos = MagicMock(return_value=sample_data.to_records(index=False))

    with patch.object(connector, '_ensure_connection', return_value=True) as mock_ensure_conn:
        df = connector.get_data("EURUSD", mock_mt5_instance.TIMEFRAME_M1, 100)
        mock_ensure_conn.assert_called_once()
        mock_mt5_instance.copy_rates_from_pos.assert_called_once_with("EURUSD", mock_mt5_instance.TIMEFRAME_M1, 0, 100)
        assert not df.empty
        assert C.DATETIME_COL in df.columns

```
