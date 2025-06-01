"""
Module for interacting with MetaTrader 5.
"""

import logging
import time # Import time for sleep
import pandas as pd
import numpy as np # Not strictly used here, but often with pandas
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pytz # For timezone handling in get_history_data
from datetime import datetime # For get_history_data

# Import from utility modules
from .utils.error_handler import ConnectionError as MT5ConnectionError # Specific import
from .utils.error_handler import OperationError, ValidationError # Keep other specific errors
# mt5_utils are mostly superseded by direct calls within this class now,
# but disconnect_mt5 might still be used from there.
from .utils.mt5_utils import disconnect_mt5

from .config_manager import ConfigManager
from .trading_operations import MT5TradingOperations
from . import constants as C

# Import MT5 with error handling
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # Define a dummy mt5 for type hinting and basic functionality if MT5 is not installed
    class DummyMT5:
        TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
        TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1, TIMEFRAME_MN1 = 101, 104, 201, 301, 401
        ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
        TRADE_ACTION_DEAL = 1
        ORDER_TIME_GTC = 0
        ORDER_FILLING_FOK = 0 # Or an appropriate default
        @staticmethod
        def initialize(*args, **kwargs): return False
        @staticmethod
        def login(*args, **kwargs): return False
        @staticmethod
        def terminal_info(*args, **kwargs): return None
        @staticmethod
        def account_info(*args, **kwargs): return None
        @staticmethod
        def symbol_info(*args, **kwargs): return None
        @staticmethod
        def symbol_info_tick(*args, **kwargs): return None
        @staticmethod
        def positions_get(*args, **kwargs): return None
        @staticmethod
        def copy_rates_range(*args, **kwargs): return None
        @staticmethod
        def copy_rates_from(*args, **kwargs): return None
        @staticmethod
        def last_error(*args, **kwargs): return "MT5 module not available"
        @staticmethod
        def shutdown(*args, **kwargs): pass

    mt5 = DummyMT5()
    logging.warning("MetaTrader5 module not found. MT5 functionality will be simulated or disabled.")

logger = logging.getLogger(__name__)

class MT5Connector:
    """Class for interacting with MetaTrader 5."""
    
    def __init__(self, config: ConfigManager, is_kill_switch_active_func: Optional[Callable[[], bool]] = None):
        self.config = config
        self.initialized = False
        self.connected = False
        self._is_kill_switch_active_func = is_kill_switch_active_func if is_kill_switch_active_func else lambda: False
        
        self.mt5_config = self.config.get_mt5_config()
        
        self.max_retries = self.mt5_config.get(
            C.CONFIG_MT5_CONNECTION_MAX_RETRIES,
            C.DEFAULT_MT5_CONNECTION_MAX_RETRIES
        )
        self.base_retry_delay = self.mt5_config.get(
            C.CONFIG_MT5_CONNECTION_RETRY_DELAY, # Corrected to match constant in previous step
            C.DEFAULT_MT5_CONNECTION_RETRY_DELAY_SECONDS
        )

        self._cache = {}
        
        if not self._initialize(): # This now has retry logic
             logger.error("MT5Connector failed to initialize and connect after all retries.")
        
        if self.initialized and self.connected:
            self.trading = MT5TradingOperations(self, self._is_kill_switch_active_func)
        else:
            self.trading = None
            logger.warning("Trading operations module not initialized due to MT5 connection failure.")

    def is_kill_switch_active(self) -> bool:
        return self._is_kill_switch_active_func()

    def is_connected(self) -> bool:
        if not MT5_AVAILABLE or not self.initialized:
            self.connected = False # Ensure flag is accurate
            return False
        try:
            term_info = mt5.terminal_info()
            if term_info is None or not term_info.connected:
                logger.debug("is_connected: terminal_info is None or not connected.")
                self.connected = False
            else:
                # If terminal is connected, check account status as well
                acc_info = mt5.account_info()
                if acc_info is None or acc_info.login == 0:
                    logger.debug("is_connected: terminal ok, but account_info is None or login is 0.")
                    self.connected = False
                else:
                    self.connected = True # Both terminal and account seem ok
        except Exception as e:
            logger.warning(f"is_connected: Exception during MT5 status check: {e}")
            self.connected = False
        return self.connected

    def check_connection_and_reconnect(self) -> bool:
        if self.is_connected():
            return True
        logger.warning("check_connection_and_reconnect: Connection lost or invalid. Attempting to re-initialize...")
        return self._initialize()

    def _initialize(self) -> bool:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 module not available.")
            return False

        terminal_path = self.mt5_config.get(C.CONFIG_MT5_PATH)
        server = self.mt5_config.get(C.CONFIG_MT5_SERVER)
        login_val = self.mt5_config.get(C.CONFIG_MT5_LOGIN)
        login = int(login_val) if login_val is not None else 0
        password = self.mt5_config.get(C.CONFIG_MT5_PASSWORD)
        timeout_val = self.mt5_config.get(C.CONFIG_MT5_TIMEOUT, 60000)

        current_retry = 0
        while current_retry <= self.max_retries:
            attempt_num = current_retry + 1
            logger.info(f"MT5 connection attempt {attempt_num} of {self.max_retries + 1}...")
            try:
                if self.initialized: # If previously initialized, shutdown before re-initializing
                    logger.debug("Shutting down existing MT5 instance before re-attempt.")
                    mt5.shutdown()
                    self.initialized = False
                    self.connected = False

                if not mt5.initialize(path=terminal_path, timeout=timeout_val, portable=self.mt5_config.get(C.CONFIG_MT5_PORTABLE, False)):
                    error = mt5.last_error()
                    logger.warning(f"MT5 initialize() failed on attempt {attempt_num}. Error: {error}")
                else:
                    self.initialized = True
                    logger.info("MT5 terminal initialized successfully.")
                    if not mt5.login(login=login, password=password, server=server, timeout=timeout_val):
                        error = mt5.last_error()
                        logger.warning(f"MT5 login() failed for account {login}@{server} on attempt {attempt_num}. Error: {error}")
                        self.initialized = False # Reset if login fails
                        # mt5.shutdown() # Ensure cleanup if login fails after successful init
                    else:
                        self.connected = True
                        logger.info(f"Successfully connected to MT5 account {login}@{server}.")
                        return True
            except Exception as e:
                logger.error(f"Unexpected error during MT5 connection attempt {attempt_num}: {e}")
                if self.initialized: mt5.shutdown() # Attempt to cleanup
                self.initialized = False
                self.connected = False

            current_retry += 1
            if current_retry <= self.max_retries:
                current_delay = self.base_retry_delay * (2 ** (current_retry - 1))
                logger.info(f"Retrying MT5 connection in {current_delay:.2f} seconds...")
                time.sleep(current_delay)
            else:
                logger.error(f"MT5 connection failed after {self.max_retries + 1} attempts.")

        self.initialized = False
        self.connected = False
        return False

    def _ensure_connection(self) -> bool:
        if not self.check_connection_and_reconnect():
            raise MT5ConnectionError("MT5 connection failed and could not be re-established.")
        return True
    
    @safe_operation("get_account_info")
    def get_account_info(self) -> Dict[str, Any]:
        self._ensure_connection()
        acc_info = mt5.account_info()
        if acc_info is None:
            raise OperationError(f"Failed to get account info: {mt5.last_error()}")
        return acc_info._asdict()
    
    @safe_operation("get_symbol_info")
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        self._ensure_connection()
        cache_key = f"symbol_info_{symbol}"
        if cache_key in self._cache:
            if self.is_connected(): # Re-validate cache if connection might have dropped
                 return self._cache[cache_key].copy()
            else: # Connection lost, cache might be stale
                 self._cache.pop(cache_key, None)


        s_info = mt5.symbol_info(symbol)
        if s_info is None:
            raise OperationError(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
        
        symbol_info_dict = s_info._asdict()
        self._cache[cache_key] = symbol_info_dict.copy()
        return symbol_info_dict

    @retry(max_retries=3, delay=1.0, exceptions=(MT5ConnectionError, TimeoutError, OperationError))
    @safe_operation("get_history_data")
    def get_data(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]: # Renamed for clarity from main_bot
        self._ensure_connection()
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No history data found for {symbol} / MT5 Timeframe {timeframe}")
            return pd.DataFrame() # Return empty DataFrame as per existing uses
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'time': C.DATETIME_COL, 'open': C.OPEN_COL, 'high': C.HIGH_COL,
                           'low': C.LOW_COL, 'close': C.CLOSE_COL, 'tick_volume': C.VOLUME_COL}, inplace=True)
        df.set_index(C.DATETIME_COL, inplace=True)
        return df

    @retry(max_retries=3, delay=1.0, exceptions=(MT5ConnectionError, TimeoutError, OperationError))
    @safe_operation("get_symbol_price")
    def get_symbol_price(self, symbol: str) -> Dict[str, float]:
        self._ensure_connection()
        ticker = mt5.symbol_info_tick(symbol)
        if ticker is None:
            raise OperationError(f"Failed to get ticker for {symbol}: {mt5.last_error()}")
        return {'bid': ticker.bid, 'ask': ticker.ask, 'last': ticker.last,
                'time': pd.to_datetime(ticker.time_msc, unit='ms')}

    @retry(max_retries=3, delay=1.0, exceptions=(MT5ConnectionError, TimeoutError, OperationError))
    @safe_operation("get_positions")
    def get_open_positions(self, symbol: Optional[str] = None, bypass_kill_switch: bool = False) -> Tuple[List[Dict[str, Any]], None, None]: # Matched main_bot call
        if not bypass_kill_switch and self.is_kill_switch_active():
            logger.warning("Kill switch is active. get_open_positions operation aborted.")
            return [], None, None
        self._ensure_connection()
        if symbol: positions = mt5.positions_get(symbol=symbol)
        else: positions = mt5.positions_get()
        if positions is None:
            logger.warning(f"No positions found or error occurred: {mt5.last_error()}")
            return [], None, None
        return [position._asdict() for position in positions], None, None

    def place_order(self, *args, **kwargs) -> Dict[str, Any]:
        self._ensure_connection()
        if not self.trading:
            logger.error("Trading operations module not available (connection may have failed).")
            raise MT5ConnectionError("Trading operations not initialized due to MT5 connection issue.")
        return self.trading.open_position(*args, **kwargs)

    def close_position(self, ticket: int, volume: Optional[float] = None, comment: str = "", bypass_kill_switch: bool = False) -> Dict[str, Any]:
        self._ensure_connection()
        if not self.trading:
            logger.error("Trading operations module not available (connection may have failed).")
            raise MT5ConnectionError("Trading operations not initialized due to MT5 connection issue.")
        return self.trading.close_position(ticket, volume, comment, bypass_kill_switch=bypass_kill_switch)

    def modify_position(self, *args, **kwargs) -> Dict[str, Any]:
        self._ensure_connection()
        if not self.trading:
            logger.error("Trading operations module not available (connection may have failed).")
            raise MT5ConnectionError("Trading operations not initialized due to MT5 connection issue.")
        return self.trading.modify_position(*args, **kwargs)

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if MT5_AVAILABLE and self.initialized: # Only shutdown if initialized
             logger.info("Shutting down MT5 connection...")
             mt5.shutdown()
        self.initialized = False
        self.connected = False
        logger.info("Disconnected from MT5 (logical status).")
    
    def __del__(self):
        self.disconnect()

```
