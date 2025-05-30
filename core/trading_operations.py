"""
Module for trading operations with MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Any, Union

# Import from utility modules
from .utils.error_handler import retry, safe_operation, ConnectionError, TimeoutError, OperationError, ValidationError
from . import constants as C # Import constants
from datetime import datetime # For paper trading ticket generation

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
    
    def __init__(self, connector, is_kill_switch_active_func: Callable[[], bool]):
        """
        Initialize trading operations.
        
        Args:
            connector: MT5Connector instance
            is_kill_switch_active_func: Callable that returns True if kill switch is active.
        """
        self.connector = connector
        self.is_kill_switch_active = is_kill_switch_active_func # Store the function
        # Access ConfigManager through the connector
        self.config_manager = self.connector.config
        global_settings = self.config_manager.get_global_settings()
        self.paper_trading = global_settings.get(C.CONFIG_PAPER_TRADING, C.DEFAULT_PAPER_TRADING)
        if self.paper_trading:
            logger.info(f"{C.PAPER_TRADE_COMMENT_PREFIX.upper()} MODE ENABLED: No real orders will be sent to MT5.")
    
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
        if self.is_kill_switch_active():
            logger.critical("Kill switch is active. Open position operation aborted.")
            # Return a structure similar to a failed trade or a specific kill switch status
            return {
                C.POSITION_TICKET: 0,
                'retcode': -1, # Custom retcode for kill switch
                C.REQUEST_COMMENT: "Operation aborted by kill switch",
                'request': {}
            }

        # Validate parameters (common for both paper and real trading)
        if order_type not in [C.ORDER_TYPE_BUY, C.ORDER_TYPE_SELL]:
            raise ValidationError(f"Invalid order type: {order_type}")
            
        if volume <= 0:
            raise ValidationError(f"Invalid volume: {volume}")

        # Determine MT5 order type for request structure (even for paper)
        mt5_order_type_enum = mt5.ORDER_TYPE_BUY if order_type == C.ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL

        # Get current price if not provided (common for logging in paper mode too)
        # In a real scenario, for paper trading, we might want to simulate price fetching too.
        # For now, assume price fetching is fine or use provided price.
        current_market_price = price
        if current_market_price is None:
            # This part still needs MT5 connection for price, or needs to be simulated too.
            # For simplicity, let's assume self.connector.get_symbol_price is okay for paper mode,
            # or a price is provided.
            try:
                fetched_price_info = self.connector.get_symbol_price(symbol)
                current_market_price = fetched_price_info['ask'] if order_type == C.ORDER_TYPE_BUY else fetched_price_info['bid']
            except Exception as e:
                logger.warning(f"{C.PAPER_TRADE_COMMENT_PREFIX}: Could not fetch live price for {symbol} due to {e}. Using 0.0.")
                current_market_price = 0.0

        request_params = {
            C.REQUEST_SYMBOL: symbol,
            C.REQUEST_VOLUME: volume,
            C.REQUEST_TYPE: order_type,
            C.REQUEST_PRICE: current_market_price,
            C.REQUEST_SL: stop_loss,
            C.REQUEST_TP: take_profit,
            C.REQUEST_COMMENT: comment,
            C.REQUEST_MAGIC: self.config_manager.get_global_settings().get(C.CONFIG_MAGIC_NUMBER, C.DEFAULT_MAGIC_NUMBER),
        }

        if self.paper_trading:
            simulated_ticket = int(datetime.now().timestamp() * 1000)
            log_msg = (f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating OPEN position: "
                       f"Symbol={symbol}, Type={order_type}, Vol={volume}, Price={current_market_price}, "
                       f"SL={stop_loss}, TP={take_profit}, Ticket={simulated_ticket}, Comment='{comment}'")
            logger.info(log_msg)
            return {
                C.POSITION_TICKET: simulated_ticket,
                'retcode': C.RETCODE_DONE,
                C.REQUEST_COMMENT: f"{C.PAPER_TRADE_COMMENT_PREFIX} executed successfully",
                'request': {**request_params, C.REQUEST_TYPE: mt5_order_type_enum, C.REQUEST_PRICE: current_market_price}
            }

        self.connector._ensure_connection()
        
        global_settings = self.config_manager.get_global_settings()
        mt5_request = {
            C.REQUEST_ACTION: mt5.TRADE_ACTION_DEAL,
            C.REQUEST_SYMBOL: symbol,
            C.REQUEST_VOLUME: volume,
            C.REQUEST_TYPE: mt5_order_type_enum,
            C.REQUEST_PRICE: current_market_price,
            C.REQUEST_SL: stop_loss if stop_loss is not None else 0.0,
            C.REQUEST_TP: take_profit if take_profit is not None else 0.0,
            C.REQUEST_DEVIATION: global_settings.get(C.CONFIG_MAX_SLIPPAGE_POINTS, C.DEFAULT_MAX_SLIPPAGE_POINTS),
            C.REQUEST_MAGIC: global_settings.get(C.CONFIG_MAGIC_NUMBER, C.DEFAULT_MAGIC_NUMBER),
            C.REQUEST_COMMENT: comment,
            C.REQUEST_TYPE_TIME: mt5.ORDER_TIME_GTC,
            C.REQUEST_TYPE_FILLING: mt5.ORDER_FILLING_FOK,
        }
        
        try:
            result = mt5.order_send(mt5_request)
            if result is None:
                error_code = mt5.last_error()
                raise OperationError(f"Failed to send order (result is None): MT5 Error Code {error_code}")
                
            if result.retcode != C.RETCODE_DONE: # Use constant for retcode
                raise OperationError(f"Order failed: Code={result.retcode}, Comment='{result.comment}', Request Volume={result.request.volume if result.request else 'N/A'}")
                
            order_result = {
                C.POSITION_TICKET: result.order,
                'retcode': result.retcode,
                C.REQUEST_COMMENT: result.comment,
                'request': mt5_request
            }
            
            logger.info(f"{C.LOG_MSG_ORDER_OPENED}: Ticket={order_result[C.POSITION_TICKET]}, Symbol={symbol}, Type={order_type}, Vol={volume}")
            return order_result
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {str(e)}")
            raise OperationError(f"Error opening position for {symbol}: {str(e)}")
    
    @retry(max_retries=3, delay=1.0, exceptions=(ConnectionError, TimeoutError, OperationError))
    @safe_operation("close_position")
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: str = "",
        bypass_kill_switch: bool = False # New parameter
    ) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (if None, closes entire position)
            comment: Order comment
            bypass_kill_switch: If True, allows operation even if kill switch is active.
            
        Returns:
            Dictionary with order result
            
        Raises:
            ConnectionError: If not connected to MT5
            ValidationError: If invalid parameters
            OperationError: If operation fails
        """
        if not bypass_kill_switch and self.is_kill_switch_active():
            logger.critical(f"Kill switch is active. Close position operation for ticket {ticket} aborted.")
            return {
                C.POSITION_TICKET: ticket, # Original ticket
                'retcode': -1, # Custom retcode for kill switch
                C.REQUEST_COMMENT: "Operation aborted by kill switch",
                'request': {}
            }

        if self.paper_trading:
            log_msg = (f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating CLOSE position: "
                       f"Ticket={ticket}, Volume={volume}, Comment='{comment}'")
            logger.info(log_msg)
            return {
                C.POSITION_TICKET: int(datetime.now().timestamp() * 1000),
                'retcode': C.RETCODE_DONE,
                C.REQUEST_COMMENT: f"{C.PAPER_TRADE_COMMENT_PREFIX} close executed successfully",
                'request': {'action': 'close', C.POSITION_TICKET: ticket, C.REQUEST_VOLUME: volume, C.REQUEST_COMMENT: comment}
            }

        self.connector._ensure_connection()
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            # Check if it was already closed or never existed
            history_orders = mt5.history_deals_get(position=ticket) # ticket is position ID
            history_orders = mt5.history_deals_get(position=ticket)
            if history_orders and len(history_orders) > 0:
                 logger.warning(f"Position {ticket} may have already been closed or is a historical order.")
                 return {
                    C.POSITION_TICKET: ticket,
                    'retcode': C.RETCODE_DONE,
                    C.REQUEST_COMMENT: "Position already closed or historical",
                    # Using actual string keys for 'request' dict as it's for info/logging
                    'request': {'action': 'close', 'ticket': ticket}
                 }
            raise ValidationError(f"Position with ticket {ticket} not found for closing.")
            
        position_data = positions[0]._asdict()
        
        close_volume = volume if volume is not None else position_data[C.POSITION_VOLUME]
        
        if close_volume > position_data[C.POSITION_VOLUME]:
            raise ValidationError(f"Close volume ({close_volume}) exceeds position volume ({position_data[C.POSITION_VOLUME]}) for ticket {ticket}")
            
        mt5_order_type_to_close = mt5.ORDER_TYPE_SELL if position_data[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
        price_info = self.connector.get_symbol_price(position_data[C.POSITION_SYMBOL])
        closing_price = price_info['bid'] if position_data[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else price_info['ask']
        
        global_settings = self.config_manager.get_global_settings()
        mt5_request = {
            C.REQUEST_ACTION: mt5.TRADE_ACTION_DEAL,
            C.REQUEST_SYMBOL: position_data[C.POSITION_SYMBOL],
            C.REQUEST_VOLUME: close_volume,
            C.REQUEST_TYPE: mt5_order_type_to_close,
            C.REQUEST_POSITION: ticket,
            C.REQUEST_PRICE: closing_price,
            C.REQUEST_DEVIATION: global_settings.get(C.CONFIG_MAX_SLIPPAGE_POINTS, C.DEFAULT_MAX_SLIPPAGE_POINTS),
            C.REQUEST_MAGIC: position_data[C.POSITION_MAGIC],
            C.REQUEST_COMMENT: comment,
            C.REQUEST_TYPE_TIME: mt5.ORDER_TIME_GTC,
            C.REQUEST_TYPE_FILLING: mt5.ORDER_FILLING_FOK,
        }
        
        try:
            result = mt5.order_send(mt5_request)
            if result is None:
                error_code = mt5.last_error()
                raise OperationError(f"Failed to close position {ticket} (result is None): MT5 Error Code {error_code}")
                
            if result.retcode != C.RETCODE_DONE:
                raise OperationError(f"Close position {ticket} failed: Code={result.retcode}, Comment='{result.comment}'")
                
            close_result = {
                C.POSITION_TICKET: result.order,
                'original_ticket': ticket,
                'retcode': result.retcode,
                C.REQUEST_COMMENT: result.comment,
                'request': mt5_request
            }
            
            logger.info(f"Position {ticket} closed successfully by order {result.order}.")
            return close_result
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {str(e)}")
            raise OperationError(f"Error closing position {ticket}: {str(e)}")
    
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
        if self.is_kill_switch_active():
            logger.critical(f"Kill switch is active. Modify position operation for ticket {ticket} aborted.")
            return {
                C.POSITION_TICKET: ticket,
                'retcode': -1, # Custom retcode for kill switch
                C.REQUEST_COMMENT: "Operation aborted by kill switch",
                'request': {}
            }

        if self.paper_trading:
            log_msg = (f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating MODIFY position: "
                       f"Ticket={ticket}, SL={stop_loss}, TP={take_profit}")
            logger.info(log_msg)
            return {
                C.POSITION_TICKET: ticket,
                'retcode': C.RETCODE_DONE,
                C.REQUEST_COMMENT: f"{C.PAPER_TRADE_COMMENT_PREFIX} modify executed successfully",
                'request': {'action': 'modify', C.POSITION_TICKET: ticket, C.REQUEST_SL: stop_loss, C.REQUEST_TP: take_profit}
            }

        self.connector._ensure_connection()
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise ValidationError(f"Position with ticket {ticket} not found for modification.")
            
        position_data = positions[0]._asdict()
        
        sl_to_set = stop_loss if stop_loss is not None else position_data[C.POSITION_SL]
        tp_to_set = take_profit if take_profit is not None else position_data[C.POSITION_TP]

        if sl_to_set is None: sl_to_set = 0.0
        if tp_to_set is None: tp_to_set = 0.0

        mt5_request = {
            C.REQUEST_ACTION: mt5.TRADE_ACTION_SLTP,
            C.REQUEST_SYMBOL: position_data[C.POSITION_SYMBOL],
            C.REQUEST_POSITION: ticket,
            C.REQUEST_SL: sl_to_set,
            C.REQUEST_TP: tp_to_set
        }
        
        try:
            result = mt5.order_send(mt5_request)
            if result is None:
                error_code = mt5.last_error()
                raise OperationError(f"Failed to modify position {ticket} (result is None): MT5 Error Code {error_code}")
                
            if result.retcode != C.RETCODE_DONE:
                raise OperationError(f"Modify position {ticket} failed: Code={result.retcode}, Comment='{result.comment}'")
                
            modify_result = {
                C.POSITION_TICKET: ticket,
                'retcode': result.retcode,
                C.REQUEST_COMMENT: result.comment,
                'request': mt5_request
            }
            
            logger.info(f"Position {ticket} modified successfully: SL={sl_to_set}, TP={tp_to_set}")
            return modify_result
        except Exception as e:
            logger.error(f"Error modifying position {ticket}: {str(e)}")
            raise OperationError(f"Error modifying position {ticket}: {str(e)}")