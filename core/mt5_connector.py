"""
Module for interacting with MetaTrader 5.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pytz
from datetime import datetime, timedelta

# Import from utility modules
from .utils.error_handler import retry, safe_operation, ConnectionError, TimeoutError, OperationError, ValidationError
from .utils.mt5_utils import (
    mt5_connection, with_mt5_connection, initialize_mt5, connect_mt5, disconnect_mt5,
    get_account_info, get_symbol_info, get_symbols, check_connection_state
)
from .config_manager import ConfigManager
from .trading_operations import MT5TradingOperations

# Import MT5 with error handling
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 module not found. MT5 functionality will be disabled.")

logger = logging.getLogger(__name__)

class MT5Connector:
    """Class for interacting with MetaTrader 5."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the MT5 connector.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.initialized = False
        self.connected = False
        
        # Get MT5 configuration
        self.mt5_config = self.config.get_mt5_config()
        
        # Cache for frequently accessed data
        self._cache = {}
        
        # Try to initialize and connect
        self._initialize()
        
        # Initialize trading operations
        if self.initialized and self.connected:
            self.trading = MT5TradingOperations(self)
    
    def _initialize(self) -> bool:
        """
        Initialize connection to MetaTrader 5.
        
        Returns:
            True if successful, False otherwise
        """
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 module not available")
            return False
            
        try:
            # Initialize MT5
            terminal_path = self.mt5_config.get("path")
            if initialize_mt5(terminal_path):
                self.initialized = True
                
                # Try to connect if initialization succeeded
                server = self.mt5_config.get("server")
                login = int(self.mt5_config.get("login"))
                password = self.mt5_config.get("password")
                
                if connect_mt5(server, login, password):
                    self.connected = True
                    logger.info(f"Connected to MT5 account {login}@{server}")
                    return True
                else:
                    logger.error("Failed to connect to MT5 account")
            else:
                logger.error("Failed to initialize MT5 terminal")
                
            return False
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False
    
    def _ensure_connection(self) -> bool:
        """
        Ensure connection to MT5 is active.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.initialized or not self.connected:
            return self._initialize()
            
        # Check connection state
        try:
            check_connection_state()
            return True
        except ConnectionError:
            logger.warning("MT5 connection lost, trying to reconnect")
            return self._initialize()
    
    @safe_operation("get_account_info")
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get information about the current account.
        
        Returns:
            Dictionary with account information
            
        Raises:
            ConnectionError: If not connected to MT5
            OperationError: If operation fails
        """
        self._ensure_connection()
        return get_account_info()
    
    @safe_operation("get_symbol_info")
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a trading symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Dictionary with symbol information
            
        Raises:
            ConnectionError: If not connected to MT5
            OperationError: If operation fails
        """
        self._ensure_connection()
        
        # Check cache first
        cache_key = f"symbol_info_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
            
        # Get fresh data
        symbol_info = get_symbol_info(symbol)
        
        # Cache result
        self._cache[cache_key] = symbol_info.copy()
        
        return symbol_info
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("get_history_data")
    def get_history_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        count: int = 500
    ) -> pd.DataFrame:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'M15', 'H1', 'D1')
            start_time: Start time (optional)
            end_time: End time (optional)
            count: Number of bars to fetch (used if start_time not provided)
            
        Returns:
            DataFrame with historical data
            
        Raises:
            ConnectionError: If not connected to MT5
            ValidationError: If invalid parameters
            OperationError: If operation fails
        """
        self._ensure_connection()
        
        # Convert timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        if timeframe not in timeframe_map:
            raise ValidationError(f"Invalid timeframe: {timeframe}")
            
        mt5_timeframe = timeframe_map[timeframe]
        
        # Set timezone to UTC
        timezone = pytz.timezone("UTC")
        
        # Set end time to now if not provided
        if end_time is None:
            end_time = datetime.now(timezone)
        elif not isinstance(end_time, datetime):
            end_time = pd.to_datetime(end_time)
            
        if not end_time.tzinfo:
            end_time = timezone.localize(end_time)
            
        # Set start time if provided
        if start_time is not None:
            if not isinstance(start_time, datetime):
                start_time = pd.to_datetime(start_time)
                
            if not start_time.tzinfo:
                start_time = timezone.localize(start_time)
                
            # Fetch data using time range
            try:
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_time, end_time)
            except Exception as e:
                logger.error(f"Error fetching history data for {symbol}/{timeframe}: {str(e)}")
                raise OperationError(f"Error fetching history data: {str(e)}")
        else:
            # Fetch data using count
            try:
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, end_time, count)
            except Exception as e:
                logger.error(f"Error fetching history data for {symbol}/{timeframe}: {str(e)}")
                raise OperationError(f"Error fetching history data: {str(e)}")
                
        if rates is None or len(rates) == 0:
            logger.warning(f"No history data found for {symbol}/{timeframe}")
            return pd.DataFrame()
            
        # Create DataFrame
        df = pd.DataFrame(rates)
        
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to standard OHLCV names
        df.rename(columns={
            'time': 'Time',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread',
            'real_volume': 'RealVolume'
        }, inplace=True)
        
        # Set index to time
        df.set_index('Time', inplace=True)
        
        return df
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("get_symbol_price")
    def get_symbol_price(self, symbol: str) -> Dict[str, float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Dictionary with bid and ask prices
            
        Raises:
            ConnectionError: If not connected to MT5
            OperationError: If operation fails
        """
        self._ensure_connection()
        
        try:
            ticker = mt5.symbol_info_tick(symbol)
            if ticker is None:
                error = mt5.last_error()
                raise OperationError(f"Failed to get ticker for {symbol}: {error}")
                
            return {
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'time': pd.to_datetime(ticker.time_msc, unit='ms')
            }
        except Exception as e:
            logger.error(f"Error getting symbol price for {symbol}: {str(e)}")
            raise OperationError(f"Error getting symbol price: {str(e)}")
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("get_positions")
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Args:
            symbol: Trading symbol to filter by (optional)
            
        Returns:
            List of open positions
            
        Raises:
            ConnectionError: If not connected to MT5
            OperationError: If operation fails
        """
        self._ensure_connection()
        
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                error = mt5.last_error()
                logger.warning(f"No positions found: {error}")
                return []
                
            # Convert to list of dictionaries
            return [position._asdict() for position in positions]
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise OperationError(f"Error getting positions: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        disconnect_mt5()
        self.initialized = False
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.disconnect()
