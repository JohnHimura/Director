"""
Module for trading operations with MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable # Added Callable

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
        log_extras = {'symbol': symbol} # Base extras for this operation
        if self.is_kill_switch_active():
            logger.critical("Kill switch is active. Open position operation aborted.", extra=log_extras)
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
                logger.warning(f"{C.PAPER_TRADE_COMMENT_PREFIX}: Could not fetch live price for {symbol} due to {e}. Using 0.0.", extra=log_extras)
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
                       f"Type={order_type}, Vol={volume}, Price={current_market_price}, "
                       f"SL={stop_loss}, TP={take_profit}, Ticket={simulated_ticket}, Comment='{comment}'")
            logger.info(log_msg, extra={'symbol': symbol, 'ticket': simulated_ticket})
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
            log_extras_trade = {**log_extras, 'ticket': result.order}
            logger.info(f"{C.LOG_MSG_ORDER_OPENED}: Type={order_type}, Vol={volume}", extra=log_extras_trade)
            return order_result
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}", extra=log_extras)
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
        log_extras_ticket = {'ticket': ticket} # Base extras for this operation, symbol added later if available
        if not bypass_kill_switch and self.is_kill_switch_active():
            logger.critical(f"Kill switch is active. Close position operation for ticket {ticket} aborted.", extra=log_extras_ticket)
            return {
                C.POSITION_TICKET: ticket,
                'retcode': -1,
                C.REQUEST_COMMENT: "Operation aborted by kill switch",
                'request': {}
            }

        if self.paper_trading:
            log_msg = (f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating CLOSE position: "
                       f"Volume={volume}, Comment='{comment}'")
            logger.info(log_msg, extra=log_extras_ticket)
            return {
                C.POSITION_TICKET: int(datetime.now().timestamp() * 1000), # New simulated ticket for the close operation itself
                'retcode': C.RETCODE_DONE,
                C.REQUEST_COMMENT: f"{C.PAPER_TRADE_COMMENT_PREFIX} close executed successfully",
                'request': {'action': 'close', C.POSITION_TICKET: ticket, C.REQUEST_VOLUME: volume, C.REQUEST_COMMENT: comment}
            }

        self.connector._ensure_connection()
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            # Check if it was already closed or never existed
            history_orders = mt5.history_deals_get(position=ticket)
            if history_orders and len(history_orders) > 0:
                 logger.warning(f"Position {ticket} may have already been closed or is a historical order.", extra=log_extras_ticket)
                 return {
                    C.POSITION_TICKET: ticket,
                    'retcode': C.RETCODE_DONE,
                    C.REQUEST_COMMENT: "Position already closed or historical",
                    # Using actual string keys for 'request' dict as it's for info/logging
                    'request': {'action': 'close', 'ticket': ticket}
                 }
            raise ValidationError(f"Position with ticket {ticket} not found for closing.") # This will be caught by @safe_operation
            
        position_data = positions[0]._asdict()
        log_extras_ticket['symbol'] = position_data[C.REQUEST_SYMBOL] # Add symbol to extras now that we have it
        
        close_volume = volume if volume is not None else position_data[C.POSITION_VOLUME]
        
        if close_volume > position_data[C.POSITION_VOLUME]:
            # Add extra to ValidationError for context if it's caught by safe_operation
            raise ValidationError(f"Close volume ({close_volume}) exceeds position volume ({position_data[C.POSITION_VOLUME]}) for ticket {ticket}")
            
        mt5_order_type_to_close = mt5.ORDER_TYPE_SELL if position_data[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
        price_info = self.connector.get_symbol_price(position_data[C.REQUEST_SYMBOL])
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
                
            closed_position_profit = 0.0
            try:
                # Attempt to fetch deal details for the closing order to get profit
                # This might need a small delay or retry if deal is not immediately available.
                # For simplicity, trying once.
                time.sleep(0.5) # Small delay to allow deal processing by server
                deals = mt5.history_deals_get(order=result.order)
                if deals:
                    for deal in deals:
                        closed_position_profit += deal.profit + deal.commission + deal.swap
                else: # Fallback: try to get deals by position ID if order linking fails
                    deals_by_pos = mt5.history_deals_get(position=ticket)
                    if deals_by_pos:
                        # Filter deals that are "out" or "in/out" and match the volume, symbol, etc.
                        # This can be complex; a simpler sum of profits for the position ID might be an approximation.
                        for deal in deals_by_pos:
                            # This logic is simplified: it assumes all deals for this position ID contribute to the P&L
                            # of the part being closed. A more accurate way is to sum profits of deals
                            # whose volume matches the closed volume and occur after the position opening.
                            # For now, sum all profits for deals associated with this position ticket that are "closing" deals.
                            # A common way is to identify deals that reduce or close the position.
                            if deal.order == result.order:
                                closed_position_profit += deal.profit + deal.commission + deal.swap
                    else:
                        logger.warning(f"Could not fetch deals for closing order {result.order} or position {ticket} to determine profit.", extra=log_extras_ticket)
            except Exception as deal_ex:
                logger.warning(f"Could not determine profit for closed position {ticket} due to: {deal_ex}", extra=log_extras_ticket)

            close_result = {
                C.POSITION_TICKET: result.order,
                'original_ticket': ticket,
                'retcode': result.retcode,
                C.REQUEST_COMMENT: result.comment,
                'request': mt5_request,
                'profit': closed_position_profit
            }
            
            logger.info(f"Position closed successfully by order {result.order}. Realized P&L for this closure: {closed_position_profit:.2f}", extra=log_extras_ticket)
            return close_result
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}", extra=log_extras_ticket) # Use updated log_extras_ticket
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
        log_extras_ticket = {'ticket': ticket} # Base extras, symbol added later
        if self.is_kill_switch_active():
            logger.critical(f"Kill switch is active. Modify position operation for ticket {ticket} aborted.", extra=log_extras_ticket)
            return {
                C.POSITION_TICKET: ticket,
                'retcode': -1,
                C.REQUEST_COMMENT: "Operation aborted by kill switch",
                'request': {}
            }

        if self.paper_trading:
            log_msg = (f"[{C.PAPER_TRADE_COMMENT_PREFIX.upper()}] Simulating MODIFY position: "
                       f"SL={stop_loss}, TP={take_profit}") # Ticket is in log_extras
            logger.info(log_msg, extra=log_extras_ticket)
            return {
                C.POSITION_TICKET: ticket,
                'retcode': C.RETCODE_DONE,
                C.REQUEST_COMMENT: f"{C.PAPER_TRADE_COMMENT_PREFIX} modify executed successfully",
                'request': {'action': 'modify', C.POSITION_TICKET: ticket, C.REQUEST_SL: stop_loss, C.REQUEST_TP: take_profit}
            }

        self.connector._ensure_connection()
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise ValidationError(f"Position with ticket {ticket} not found for modification.") # Caught by @safe_operation
            
        position_data = positions[0]._asdict()
        log_extras_ticket['symbol'] = position_data[C.REQUEST_SYMBOL] # Add symbol to extras
        
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
            
            logger.info(f"Position modified successfully: SL={sl_to_set}, TP={tp_to_set}", extra=log_extras_ticket)
            return modify_result
        except Exception as e:
            logger.error(f"Error modifying position: {str(e)}", extra=log_extras_ticket)
            raise OperationError(f"Error modifying position {ticket}: {str(e)}")