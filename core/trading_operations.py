"""
Module for trading operations with MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Any, Union

# Import from utility modules
from .utils.error_handler import retry, safe_operation, ConnectionError, TimeoutError, OperationError, ValidationError

# Import MT5 with error handling
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 module not found. MT5 functionality will be disabled.")

logger = logging.getLogger(__name__)

class MT5TradingOperations:
    """Class for performing trading operations with MetaTrader 5."""
    
    def __init__(self, connector):
        """
        Initialize trading operations.
        
        Args:
            connector: MT5Connector instance
        """
        self.connector = connector
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("open_position")
    def open_position(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            volume: Position volume in lots
            price: Price to open at (market price if None)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            Dictionary with order result
            
        Raises:
            ConnectionError: If not connected to MT5
            ValidationError: If invalid parameters
            OperationError: If operation fails
        """
        self.connector._ensure_connection()
        
        # Validate parameters
        if order_type not in ['BUY', 'SELL']:
            raise ValidationError(f"Invalid order type: {order_type}")
            
        if volume <= 0:
            raise ValidationError(f"Invalid volume: {volume}")
            
        # Determine order type
        if order_type == 'BUY':
            mt5_order_type = mt5.ORDER_TYPE_BUY
            price_type = mt5.SYMBOL_TRADE_EXECUTION_MARKET
        else:  # SELL
            mt5_order_type = mt5.ORDER_TYPE_SELL
            price_type = mt5.SYMBOL_TRADE_EXECUTION_MARKET
            
        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)
        
        # Get current price if not provided
        if price is None:
            current_price = self.connector.get_symbol_price(symbol)
            price = current_price['ask'] if order_type == 'BUY' else current_price['bid']
            
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        try:
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                raise OperationError(f"Failed to send order: {error}")
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise OperationError(f"Order failed: {result.retcode} - {result.comment}")
                
            # Format the result
            order_result = {
                'ticket': result.order,
                'retcode': result.retcode,
                'comment': result.comment,
                'request': request
            }
            
            logger.info(f"Order opened successfully: {order_result['ticket']}")
            return order_result
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            raise OperationError(f"Error opening position: {str(e)}")
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("close_position")
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (if None, closes entire position)
            comment: Order comment
            
        Returns:
            Dictionary with order result
            
        Raises:
            ConnectionError: If not connected to MT5
            ValidationError: If invalid parameters
            OperationError: If operation fails
        """
        self.connector._ensure_connection()
        
        # Get position details
        positions = mt5.positions_get(ticket=ticket)
        
        if positions is None or len(positions) == 0:
            raise ValidationError(f"Position not found: {ticket}")
            
        position = positions[0]._asdict()
        
        # Determine volume to close
        close_volume = volume if volume is not None else position['volume']
        
        if close_volume > position['volume']:
            raise ValidationError(f"Close volume ({close_volume}) exceeds position volume ({position['volume']})")
            
        # Determine order type (opposite of position type)
        if position['type'] == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
        else:
            order_type = mt5.ORDER_TYPE_BUY
            
        # Get price
        symbol_info = self.connector.get_symbol_info(position['symbol'])
        
        current_price = self.connector.get_symbol_price(position['symbol'])
        price = current_price['bid'] if position['type'] == mt5.POSITION_TYPE_BUY else current_price['ask']
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position['symbol'],
            "volume": close_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": position['magic'],
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        try:
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                raise OperationError(f"Failed to close position: {error}")
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise OperationError(f"Close position failed: {result.retcode} - {result.comment}")
                
            # Format the result
            close_result = {
                'ticket': result.order,
                'retcode': result.retcode,
                'comment': result.comment,
                'request': request
            }
            
            logger.info(f"Position closed successfully: {ticket}")
            return close_result
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise OperationError(f"Error closing position: {str(e)}")
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("modify_position")
    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify an existing position.
        
        Args:
            ticket: Position ticket
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            Dictionary with order result
            
        Raises:
            ConnectionError: If not connected to MT5
            ValidationError: If invalid parameters
            OperationError: If operation fails
        """
        self.connector._ensure_connection()
        
        # Get position details
        positions = mt5.positions_get(ticket=ticket)
        
        if positions is None or len(positions) == 0:
            raise ValidationError(f"Position not found: {ticket}")
            
        position = positions[0]._asdict()
        
        # Set stop loss and take profit
        sl = stop_loss if stop_loss is not None else position['sl']
        tp = take_profit if take_profit is not None else position['tp']
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position['symbol'],
            "position": ticket,
            "sl": sl,
            "tp": tp
        }
        
        # Send order
        try:
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                raise OperationError(f"Failed to modify position: {error}")
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise OperationError(f"Modify position failed: {result.retcode} - {result.comment}")
                
            # Format the result
            modify_result = {
                'ticket': ticket,
                'retcode': result.retcode,
                'comment': result.comment,
                'request': request
            }
            
            logger.info(f"Position modified successfully: {ticket}")
            return modify_result
        except Exception as e:
            logger.error(f"Error modifying position: {str(e)}")
            raise OperationError(f"Error modifying position: {str(e)}") 