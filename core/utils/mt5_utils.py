"""
Utility functions for MetaTrader 5 operations.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, cast, Tuple, List, Union
import json

# Import MT5 with error handling
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 module not found. MT5 functionality will be disabled.")

from .error_handler import ConnectionError, TimeoutError, OperationError, retry

logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

class MT5ConnectionManager:
    """Manager class for MetaTrader 5 connection."""
    
    def __init__(self, auto_reconnect: bool = True, reconnect_interval: int = 30):
        """
        Initialize the connection manager.
        
        Args:
            auto_reconnect: Whether to automatically reconnect
            reconnect_interval: Interval between reconnection attempts in seconds
        """
        self.initialized = False
        self.connected = False
        self.connection_params = {}
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.reconnect_thread = None
        self.should_stop = threading.Event()
        self.lock = threading.RLock()
    
    def initialize(self, path: Optional[str] = None) -> bool:
        """
        Initialize the MT5 terminal.
        
        Args:
            path: Path to MetaTrader 5 terminal executable
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If MT5 is not available or initialization fails
        """
        if not MT5_AVAILABLE:
            raise ConnectionError("MetaTrader5 module not available")
            
        try:
            with self.lock:
                # Shutdown first if already initialized
                if self.initialized:
                    mt5.shutdown()
                    self.initialized = False
                    self.connected = False
                    logger.debug("Shutting down existing MT5 connection before reinitializing")
                
                if path:
                    result = mt5.initialize(path=path)
                else:
                    result = mt5.initialize()
                    
                if result:
                    self.initialized = True
                    logger.info("MT5 initialized successfully")
                else:
                    error = mt5.last_error()
                    logger.error(f"MT5 initialization failed: {error}")
                    raise ConnectionError(f"MT5 initialization failed: {error}")
                    
                return result
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            self.initialized = False
            raise ConnectionError(f"Error initializing MT5: {str(e)}")
            
    def connect(self, server: str, login: int, password: str) -> bool:
        """
        Connect to the MT5 account.
        
        Args:
            server: MT5 server name
            login: Account login ID
            password: Account password
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ConnectionError: If connection fails
        """
        if not self.initialized:
            raise ConnectionError("MT5 not initialized")
            
        try:
            with self.lock:
                result = mt5.login(login=login, password=password, server=server)
                
                if result:
                    self.connected = True
                    self.connection_params = {
                        'server': server,
                        'login': login,
                        'password': password
                    }
                    logger.info(f"Connected to MT5 account {login}@{server}")
                    
                    # Start reconnection thread if auto_reconnect is enabled
                    if self.auto_reconnect and (self.reconnect_thread is None or not self.reconnect_thread.is_alive()):
                        self.should_stop.clear()
                        self.reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
                        self.reconnect_thread.start()
                        logger.debug("Started auto-reconnect thread")
                else:
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    raise ConnectionError(f"MT5 login failed: {error}")
                    
                return result
        except Exception as e:
            logger.error(f"Error connecting to MT5: {str(e)}")
            self.connected = False
            raise ConnectionError(f"Error connecting to MT5: {str(e)}")
    
    def _reconnect_loop(self) -> None:
        """
        Background thread for automatic reconnection.
        """
        logger.debug("Reconnect loop started")
        while not self.should_stop.is_set():
            # Check connection every reconnect_interval seconds
            self.should_stop.wait(self.reconnect_interval)
            
            if self.should_stop.is_set():
                break
                
            try:
                # Check if we need to reconnect
                if not self.ensure_connection(force_check=True):
                    logger.warning("Connection check failed, attempting to reconnect")
                    self._try_reconnect()
            except Exception as e:
                logger.error(f"Error in reconnect loop: {str(e)}")
        
        logger.debug("Reconnect loop stopped")
            
    def _try_reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5.
        
        Returns:
            True if reconnection was successful, False otherwise
        """
        if not self.connection_params:
            logger.warning("No connection parameters available for reconnection")
            return False
            
        try:
            # Try to initialize and reconnect
            self.initialize()
            return self.connect(
                server=self.connection_params['server'],
                login=self.connection_params['login'],
                password=self.connection_params['password']
            )
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {str(e)}")
            return False
            
    def ensure_connection(self, force_check: bool = False) -> bool:
        """
        Ensure that the connection to MT5 is active.
        
        Args:
            force_check: Force a connection check even if flags indicate connected
        
        Returns:
            True if connected, False otherwise
        """
        if not force_check and (not self.initialized or not self.connected):
            logger.warning("MT5 not initialized or connected")
            return False
            
        # Check if terminal is still connected
        try:
            with self.lock:
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    logger.warning("MT5 terminal not responding")
                    self.connected = False
                    return False
                    
                # Check if account is still connected
                account_info = mt5.account_info()
                if account_info is None:
                    logger.warning("MT5 account not connected")
                    self.connected = False
                    return False
                    
                # All checks passed
                return True
        except Exception as e:
            logger.warning(f"Connection check failed: {str(e)}")
            self.connected = False
            return False
        
    def disconnect(self) -> None:
        """
        Disconnect from the MT5 terminal.
        """
        # Stop reconnect thread
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            self.should_stop.set()
            self.reconnect_thread.join(timeout=2.0)
            
            with self.lock:
                if self.initialized:
                    mt5.shutdown()
                    self.initialized = False
                    self.connected = False
                    logger.info("Disconnected from MT5")
            
    def __del__(self):
        """Cleanup on object destruction."""
        self.disconnect()

# Create singleton instance
mt5_connection = MT5ConnectionManager()

def with_mt5_connection(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to ensure MT5 connection before executing a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if not mt5_connection.ensure_connection():
            raise ConnectionError("MT5 not connected")
        return func(*args, **kwargs)
    return wrapper

def initialize_mt5(path: Optional[str] = None) -> bool:
    """
    Initialize MT5 terminal.
    
    Args:
        path: Path to MT5 terminal executable
        
    Returns:
        True if successful, False otherwise
    """
    return mt5_connection.initialize(path)

def connect_mt5(server: str, login: int, password: str) -> bool:
    """
    Connect to MT5 account.
    
    Args:
        server: MT5 server name
        login: Account login ID
        password: Account password
        
    Returns:
        True if successful, False otherwise
    """
    return mt5_connection.connect(server, login, password)

@retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
@with_mt5_connection
def get_account_info() -> Dict[str, Any]:
    """
    Get account information with retry logic.
    
    Returns:
        Account information as a dictionary
        
    Raises:
        ConnectionError: If connection fails
        OperationError: If operation fails
    """
    try:
        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            raise OperationError(f"Failed to get account info: {error}")
            
        # Convert to dictionary
        return account_info._asdict()
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        raise OperationError(f"Error getting account info: {str(e)}")

@retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
@with_mt5_connection
def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get information about a trading symbol with retry logic.
    
    Args:
        symbol: Trading symbol name
        
    Returns:
        Symbol information as a dictionary
        
    Raises:
        ConnectionError: If connection fails
        OperationError: If operation fails
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error = mt5.last_error()
            raise OperationError(f"Failed to get symbol info for {symbol}: {error}")
            
        # Convert to dictionary
        return symbol_info._asdict()
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
        raise OperationError(f"Error getting symbol info for {symbol}: {str(e)}")

@retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
@with_mt5_connection
def get_symbols() -> List[str]:
    """
    Get all available trading symbols with retry logic.
    
    Returns:
        List of symbol names
        
    Raises:
        ConnectionError: If connection fails
        OperationError: If operation fails
    """
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            error = mt5.last_error()
            raise OperationError(f"Failed to get symbols: {error}")
            
        return [symbol.name for symbol in symbols]
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        raise OperationError(f"Error getting symbols: {str(e)}")

@retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
@with_mt5_connection
def check_connection_state() -> Dict[str, Any]:
    """
    Check MT5 connection state with retry logic.
    
    Returns:
        Dictionary with connection information
        
    Raises:
        ConnectionError: If connection fails
    """
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            error = mt5.last_error()
            raise ConnectionError(f"Failed to get terminal info: {error}")
            
        # Convert to dictionary
        terminal_info_dict = terminal_info._asdict()
        
        # Get account info if connected
        try:
            account_info = mt5.account_info()
            if account_info is not None:
                terminal_info_dict['account'] = account_info._asdict()
        except:
            pass
            
        return terminal_info_dict
    except Exception as e:
        logger.error(f"Error checking connection state: {str(e)}")
        raise ConnectionError(f"Error checking connection state: {str(e)}")

def disconnect_mt5() -> None:
    """Disconnect from MT5 terminal."""
    mt5_connection.disconnect() 