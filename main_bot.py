"""
Main trading bot module.
"""

import logging
import time
import sys
import json
import uuid # Import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import MetaTrader5 as mt5

# Add the current directory to the path
import os
from pathlib import Path # Import Path
# import sys # sys is already imported above
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config_manager import ConfigManager
from core.logging_setup import setup_logging, get_log_context # Import get_log_context
from core.mt5_connector import MT5Connector
from core.strategy_engine import StrategyEngine, SignalType
from core.risk_manager import RiskManager
from core import constants as C

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the trading bot.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_manager = ConfigManager(config_path)
        
        # Kill switch state
        self.kill_switch_activated = False
        self._kill_switch_file_path_str = self.config_manager.get_global_settings().get(C.CONFIG_KILL_SWITCH_FILE_PATH, "KILL_SWITCH.txt")
        self._kill_switch_file = Path(self._kill_switch_file_path_str)
        self._kill_switch_close_positions = self.config_manager.get_global_settings().get(C.CONFIG_KILL_SWITCH_CLOSE_POSITIONS, True)


        # Initialize components
        self.mt5 = MT5Connector(self.config_manager, self.is_kill_switch_active) # Pass callable
        self.strategy = StrategyEngine(self.config_manager)
        
        # Get active symbols
        self.symbols = self._get_active_symbols() # This will use self.config_manager
        
        if not self.symbols:
            raise ValueError("No active symbols found in configuration")
            
        # Use the first symbol as default for risk manager
        default_symbol = next(iter(self.symbols.keys()))
            
        self.risk_manager = RiskManager(
            config=self.config_manager.get_risk_params(default_symbol),
            account_info=self._get_account_info(),
            config_manager=self.config_manager, # Pass config_manager
            symbol=default_symbol
        )
        
        # Trading state
        self.is_running = False
        self.last_update = None
        self.timeframes = self.config_manager.get_timeframes() # Use renamed config_manager
        
        # Initialize data cache
        self.data_cache = {}
        self.last_reset_date = None
        
        # Mapeo de strings de timeframe a constantes de MT5
        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M2": mt5.TIMEFRAME_M2,
            "M3": mt5.TIMEFRAME_M3,
            "M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5,
            "M6": mt5.TIMEFRAME_M6,
            "M10": mt5.TIMEFRAME_M10,
            "M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15,
            "M20": mt5.TIMEFRAME_M20,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H2": mt5.TIMEFRAME_H2,
            "H3": mt5.TIMEFRAME_H3,
            "H4": mt5.TIMEFRAME_H4,
            "H6": mt5.TIMEFRAME_H6,
            "H8": mt5.TIMEFRAME_H8,
            "H12": mt5.TIMEFRAME_H12,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
    
    def _get_account_info(self) -> Dict[str, float]:
        """Get account information from MT5."""
        if not self.mt5.initialized:
            return {
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0,
                'margin_level': 0.0
            }
        
        account_info = mt5.account_info()
        if account_info is None:
            return {
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0,
                'margin_level': 0.0
            }
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level
        }
    
    def _get_active_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Get dictionary of active trading symbols and their configurations.
        
        Returns:
            Dict where keys are symbol names and values are their configurations
        """
        active_symbols = {}
        # Use renamed config_manager and C.CONFIG_ENABLED
        for symbol, config_data in self.config_manager.get_active_symbols().items():
            if config_data.get(C.CONFIG_ENABLED, False):
                active_symbols[symbol] = config_data
        return active_symbols
    
    def initialize(self) -> bool:
        """
        Initialize the trading bot.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Initialize MT5 connection
            if not self.mt5.initialize():
                logger.error("Failed to initialize MT5 connection")
                return False
            
            logger.info("Trading bot initialized successfully")
            return True
            
        except Exception as e:
            logger.exception("Error initializing trading bot")
            return False
    
    def start(self) -> None:
        """Start the trading bot."""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        logger.info("Starting trading bot...")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    self._run_iteration()
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, stopping...")
                    self.stop()
                    break
                except Exception as e:
                    logger.exception("Error in main loop")
                    # Add a small delay to prevent tight error loops
                    time.sleep(5)
                
                # Sleep for a bit before the next iteration
                loop_interval_val = self.config_manager.get_global_settings().get(C.CONFIG_LOOP_INTERVAL, 1)
                time.sleep(loop_interval_val)
                
        finally:
            self.stop()

    def _check_for_config_reload(self) -> None:
        """Checks for configuration changes and applies them if necessary."""
        # This function was added in a previous step, ensure its internal strings are constants
        was_reloaded, changes = self.config_manager.check_and_reload_config()
        if was_reloaded:
            logger.info("Configuration reloaded. Applying changes...")
            self._apply_reloaded_config(changes)

    def _apply_reloaded_config(self, changes: Dict[str, Any]) -> None:
        """Applies changes from a reloaded configuration."""
        # This function was added in a previous step, ensure its internal strings are constants
        restart_recommended = False

        if C.RELOAD_CHANGES_LOGGING_LEVEL in changes:
            new_level_str = changes[C.RELOAD_CHANGES_LOGGING_LEVEL]
            setup_logging(config_manager=self.config_manager) # Re-setup with new config
            logger.info(f"Logging setup re-initialized with new level: {new_level_str}")

        if C.RELOAD_CHANGES_LOGGING_FILE_REQUIRES_RESTART in changes:
            logger.warning("Logging file path changed. Restart is required to apply this change.")
            restart_recommended = True

        if C.RELOAD_CHANGES_GLOBAL_SETTINGS in changes:
            logger.info(f"Global settings changed: {changes[C.RELOAD_CHANGES_GLOBAL_SETTINGS]}. RiskManager will be updated.")
            if hasattr(self.risk_manager, 'update_config') and callable(getattr(self.risk_manager, 'update_config')):
                 self.risk_manager.update_config()

        if C.RELOAD_CHANGES_SYMBOLS in changes:
            changed_symbols_data = changes[C.RELOAD_CHANGES_SYMBOLS]
            logger.info(f"Symbol configurations changed: {json.dumps(changed_symbols_data)}")
            active_symbols_refreshed = False
            for symbol, sym_changes in changed_symbols_data.items():
                if C.CONFIG_ENABLED in sym_changes:
                    logger.info(f"Symbol {symbol} enabled status changed to {sym_changes[C.CONFIG_ENABLED]}.")
                    self.symbols = self._get_active_symbols()
                    active_symbols_refreshed = True
                    if not sym_changes[C.CONFIG_ENABLED]:
                        logger.info(f"Symbol {symbol} disabled. No new trades will be opened for it.")

                if C.CONFIG_RISK in sym_changes:
                    logger.info(f"Risk parameters updated for {symbol}.")
                    if hasattr(self.risk_manager, 'update_config') and callable(getattr(self.risk_manager, 'update_config')):
                        self.risk_manager.update_config(symbol=symbol)

                if C.CONFIG_INDICATORS in sym_changes:
                    logger.info(f"Indicator parameters updated for {symbol}. Strategy will use new values on next analysis.")
                if C.CONFIG_SR in sym_changes:
                    logger.info(f"SR parameters updated for {symbol}. Strategy will use new values on next analysis.")

            if active_symbols_refreshed:
                 logger.info(f"Active symbols list refreshed: {list(self.symbols.keys())}")

        if hasattr(self.strategy, 'update_strategy_config'):
            self.strategy.update_strategy_config()

        critical_change_keys = [k for k in changes if "requires_restart" in k and k != C.RELOAD_CHANGES_LOGGING_FILE_REQUIRES_RESTART]
        if critical_change_keys:
            for key in critical_change_keys:
                logger.warning(f"Critical configuration change detected: '{key}'. A manual bot restart is required to apply this.")
            restart_recommended = True

        if restart_recommended:
            logger.warning("One or more configuration changes require a manual bot restart to take full effect.")
        else:
            logger.info("Hot-reload changes applied successfully.")

    def stop(self) -> None:
        """Stop the trading bot."""
        if not self.is_running:
            return
            
        logger.info("Stopping trading bot...")
        self.is_running = False
        self.mt5.disconnect()
    
    def _run_iteration(self) -> None:
        """Run one iteration of the trading loop."""
        # Set correlation ID for this iteration
        log_ctx = get_log_context()
        log_ctx.correlation_id = str(uuid.uuid4())

        try:
            self._check_for_config_reload()
            self._check_kill_switch() # Check kill switch status

            if self.kill_switch_activated:
                logger.critical(f"KILL SWITCH ACTIVE. File: {self._kill_switch_file_path_str}. No new trades will be initiated. Bot may be in a safe, monitoring-only mode or will stop if configured.")
                # Optional: Implement logic to stop the bot or enter a minimal monitoring loop here
                # For now, it will prevent new trades via _check_for_entries and trading_operations checks.
                # If kill_switch_close_positions was true, positions would have been closed by _check_kill_switch.
                return # Skip the rest of the iteration's trading logic

            current_date = datetime.now().date()
            if self.last_reset_date != current_date:
                logger.info("New trading day detected. Resetting daily risk statistics.")
                if hasattr(self.risk_manager, 'reset_daily_stats'):
                    self.risk_manager.reset_daily_stats()
                self.last_reset_date = current_date

            # Update account info
            account_info = self._get_account_info()
            if hasattr(self.risk_manager, 'update_account_info'):
                self.risk_manager.update_account_info(account_info)

            # Check if we can trade - ensure risk_manager is updated with latest config for limits
            can_trade, reason = True, "" # Default
            if hasattr(self.risk_manager, 'check_daily_limits'):
                can_trade, reason = self.risk_manager.check_daily_limits()

            if not can_trade:
                logger.warning("Trading limited: %s", reason)
                return

            # Process each symbol
            for symbol_name, symbol_config_dict in self.symbols.items():
                # Use C.CONFIG_ENABLED
                if symbol_config_dict.get(C.CONFIG_ENABLED, False):
                    try:
                        self._process_symbol(symbol_name, symbol_config_dict)
                    except Exception as e:
                        logger.exception("Error processing symbol %s", symbol_name)
        finally:
            # Clear correlation_id at the end of the iteration (optional, good practice)
            if hasattr(log_ctx, 'correlation_id'):
                del log_ctx.correlation_id

    def is_kill_switch_active(self) -> bool:
        """Returns the current state of the kill switch."""
        return self.kill_switch_activated

    def _check_kill_switch(self) -> None:
        """Checks for the kill switch file and takes action if found."""
        if self.kill_switch_activated: # Already activated, no need to re-check file or re-close
            return

        if self._kill_switch_file.exists():
            self.kill_switch_activated = True
            logger.critical(f"KILL SWITCH ACTIVATED by file: {self._kill_switch_file_path_str}")

            if self._kill_switch_close_positions:
                logger.info("Kill switch: Attempting to close all open positions.")
                try:
                    # Pass a special flag to allow closing even if kill switch is generally blocking operations
                    open_positions, _, _ = self.mt5.get_open_positions(bypass_kill_switch=True)
                    if open_positions:
                        for position_dict in open_positions:
                            ticket = position_dict[C.POSITION_TICKET]
                            symbol = position_dict[C.POSITION_SYMBOL]
                            logger.info(f"Kill switch: Closing position {ticket} for {symbol}.")
                            # Pass bypass_kill_switch=True to ensure close operation goes through
                            close_result = self.mt5.close_position(ticket, bypass_kill_switch=True)
                            if close_result and close_result.get('retcode') == C.RETCODE_DONE:
                                logger.info(f"Kill switch: Successfully closed position {ticket}.")
                            else:
                                logger.error(f"Kill switch: Failed to close position {ticket}. Result: {close_result}")
                    else:
                        logger.info("Kill switch: No open positions found to close.")
                except Exception as e:
                    logger.exception(f"Kill switch: Error during closing of positions: {e}")

            # Optional: Add logic here to stop the bot completely if desired
            # self.stop()
            # self.is_running = False # To break the main loop in start()

    def _old_run_iteration_content(self) -> None:
        current_date = datetime.now().date()
        if self.last_reset_date != current_date:
            logger.info("New trading day detected. Resetting daily risk statistics.")
            if hasattr(self.risk_manager, 'reset_daily_stats'):
                self.risk_manager.reset_daily_stats()
            self.last_reset_date = current_date
            
        # Update account info
        account_info = self._get_account_info()
        self.risk_manager.update_account_info(account_info)
        
        # Check if we can trade
        can_trade, reason = self.risk_manager.check_daily_limits()
        if not can_trade:
            logger.warning("Trading limited: %s", reason)
            return
        
        # Process each symbol
        for symbol_name, symbol_config_dict in self.symbols.items():
            # Use C.CONFIG_ENABLED
            if symbol_config_dict.get(C.CONFIG_ENABLED, False):
                try:
                    # Call _check_for_config_reload() before processing each symbol or less frequently
                    # For now, it's in the main loop, called once per bot iteration.
                    self._process_symbol(symbol_name, symbol_config_dict)
                except Exception as e:
                    logger.exception("Error processing symbol %s", symbol_name)
    
    def _process_symbol(self, symbol: str, symbol_config: Dict[str, Any]) -> None:
        """
        Process a single symbol.
        
        Args:
            symbol: Symbol to process
            symbol_config: Symbol configuration
        """
        logger.debug("Processing symbol: %s", symbol)
        
        # Get current positions for this symbol
        positions_list, _, _ = self.mt5.get_open_positions()
        
        # Get market data for all required timeframes
        data = self._get_market_data(symbol)
        if not data:
            logger.warning("No market data for symbol: %s", symbol)
            return
        
        # Update data cache
        self.data_cache[symbol] = data
        
        # Check if we should close any positions
        if positions_list:
            self._manage_positions(positions_list, data)
        
        # Check if we should open new positions
        # Obtener todas las posiciones abiertas para el RiskManager
        current_open_positions_list, _, _ = self.mt5.get_open_positions()
        
        if not positions_list:
            if self.kill_switch_activated:
                logger.warning(f"Kill switch active, skipping new entry check for {symbol}.")
            else:
                self._check_for_entries(symbol, data, symbol_config, current_open_positions_list)
    
    def _get_market_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get market data for a symbol across all timeframes.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Dictionary of DataFrames keyed by timeframe
        """
        data = {}
        
        # Iterar sobre los timeframes configurados (strings)
        for tf_str in self.timeframes:
            # Obtener la constante entera de MT5 para el timeframe
            timeframe_mt5 = self.timeframe_map.get(tf_str)
            
            if timeframe_mt5 is None:
                logger.warning("Invalid timeframe configured: %s. Skipping.", tf_str)
                continue # Saltar este timeframe si no se encuentra en el mapeo
                
            try:
                # Get data from MT5 usando la constante entera
                df = self.mt5.get_data(
                    symbol=symbol,
                    timeframe=timeframe_mt5, # Usar la constante entera
                    count=1000  # Get enough bars for indicators
                )
                
                # Guardar los datos usando el string original del timeframe como clave
                if df is not None and not df.empty:
                    data[tf_str] = df # Usar el string original del timeframe
                
            except Exception as e:
                logger.exception("Error getting data for %s %s", symbol, tf_str)
        
        return data
    
    def _manage_positions(self, positions: List[Dict[str, Any]], data: Dict[str, pd.DataFrame]) -> None:
        """
        Manage open positions.
        
        Args:
            positions: List of open positions (dictionaries)
            data: Market data for the symbol
        """
        for position in positions:
            try:
                self._manage_position(position, data)
            except Exception as e:
                position_ticket = position.get(C.POSITION_TICKET, 'N/A') # Use C.POSITION_TICKET
                logger.exception("Error managing position %s", position_ticket)
    
    def _manage_position(self, position: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> None:
        """
        Manage a single open position.
        
        Args:
            position: Position to manage (dictionary from MT5 position._asdict())
            data: Market data for the symbol
        """
        symbol = position[C.POSITION_SYMBOL]
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("Could not get symbol info for %s", symbol)
            return

        # position[C.POSITION_TYPE] should be mt5.POSITION_TYPE_BUY (0) or mt5.POSITION_TYPE_SELL (1)
        current_price = symbol_info.ask if position[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else symbol_info.bid
        
        # Check if we should close the position
        close_signal = self._check_exit_signals(position, data, current_price)
        
        if close_signal:
            self._close_position(position, "Exit signal: " + close_signal)
        else:
            # Check if we should move stop loss to break even
            self._check_move_to_break_even(position, current_price)
            
            # Check if we should trail stop loss
            self._check_trailing_stop(position, current_price, data)
    
    def _check_exit_signals(
        self, 
        position: Dict[str, Any], # Esperar un diccionario
        data: Dict[str, pd.DataFrame],
        current_price: float
    ) -> Optional[str]:
        """
        Check if we should exit a position.
        
        Args:
            position: Position to check (dictionary)
            data: Market data for the symbol
            current_price: Current market price
            
        Returns:
            Reason for exit, or None if should not exit
        """
        # Check stop loss and take profit
        # position[C.POSITION_TYPE] is 0 for BUY, 1 for SELL (MT5 constants)
        # position[C.POSITION_SL] and position[C.POSITION_TP]
        if position[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY:
            if (position[C.POSITION_SL] != 0 and current_price <= position[C.POSITION_SL]) or \
               (position[C.POSITION_TP] != 0 and current_price >= position[C.POSITION_TP]):
                return "SL/TP hit"
        elif position[C.POSITION_TYPE] == mt5.POSITION_TYPE_SELL:
            if (position[C.POSITION_SL] != 0 and current_price >= position[C.POSITION_SL]) or \
               (position[C.POSITION_TP] != 0 and current_price <= position[C.POSITION_TP]):
                return "SL/TP hit"
        
        # Check strategy exit signals
        # Map MT5 position type to strategy's expected SignalType if they differ
        # Assuming strategy expects SignalType.BUY (1) or SignalType.SELL (-1)
        # MT5 POSITION_TYPE_BUY is 0, POSITION_TYPE_SELL is 1. This needs careful mapping.
        # For now, assuming strategy's position_info['position_type'] takes these MT5 int constants.
        # This was handled in StrategyEngine._check_exit_signal to compare with SignalType enum.
        # Let's ensure the position_info dict passed to analyze() uses consistent types.
        # The current MACDStrategy's _check_exit_signal expects position_info['position_type'] to be SignalType.BUY/SELL

        strat_pos_type = SignalType.BUY if position[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else SignalType.SELL

        analysis = self.strategy.analyze(
            symbol=position[C.POSITION_SYMBOL],
            data=data,
            position_info={
                C.POSITION_TYPE: strat_pos_type, # Pass SignalType enum value
                C.POSITION_OPEN_PRICE: position[C.POSITION_OPEN_PRICE],
                C.POSITION_SL: position[C.POSITION_SL],
                C.POSITION_TP: position[C.POSITION_TP],
                'current_price': current_price # This key is fine as string for strategy internal use
            }
        )
        
        # analysis['signal'] is from SignalType enum
        if (strat_pos_type == SignalType.BUY and analysis['signal'] == SignalType.SELL) or \
           (strat_pos_type == SignalType.SELL and analysis['signal'] == SignalType.BUY):
            return analysis.get('message', 'Strategy exit signal') # 'message' is fine as string key
        
        return None
    
    def _close_position(self, position: Dict[str, Any], reason: str) -> None:
        """
        Close a position.
        
        Args:
            position: Position to close (dictionary from MT5 position._asdict())
            reason: Reason for closing
        """
        position_ticket = position[C.POSITION_TICKET]
        logger.info(f"Closing position {position_ticket}: {reason}")
        
        result = self.mt5.close_position(position_ticket)
        
        if result.get('retcode') != C.RETCODE_DONE:
            logger.error(f"Failed to close position {position_ticket}: {result.get(C.REQUEST_COMMENT)}")
    
    def _check_move_to_break_even(self, position: Dict[str, Any], current_price: float) -> None:
        """
        Check if we should move stop loss to break even.
        
        Args:
            position: Position to check (dictionary from MT5 position._asdict())
            current_price: Current market price
        """
        symbol = position[C.POSITION_SYMBOL]
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not get symbol info for {symbol} in break even check")
            return
        
        position_type = position[C.POSITION_TYPE] # MT5 constant
        open_price = position[C.POSITION_OPEN_PRICE]
        current_sl = position[C.POSITION_SL]

        # TODO: Make '50' points and spread factor configurable through constants or config
        points_threshold = 50

        if position_type == mt5.POSITION_TYPE_BUY:
            price_move = (current_price - open_price) / symbol_info.point
            if price_move >= points_threshold:
                new_sl = open_price + (symbol_info.spread * symbol_info.point)
                if new_sl > current_sl:
                    self._modify_position(position, sl=new_sl)
        else: # SELL
            price_move = (open_price - current_price) / symbol_info.point
            if price_move >= points_threshold:
                new_sl = open_price - (symbol_info.spread * symbol_info.point)
                if new_sl < current_sl or current_sl == 0:
                    self._modify_position(position, sl=new_sl)
    
    def _check_trailing_stop(
        self, 
        position: Dict[str, Any], # Esperar un diccionario
        current_price: float,
        data: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Check if we should trail the stop loss.
        
        Args:
            position: Position to check (dictionary)
            current_price: Current market price
            data: Market data for the symbol
        """
        # Get ATR for trailing stop distance
        # TODO: Make ATR multiplier configurable
        atr_multiplier_for_trailing = 2.0
        atr = self._get_atr(data) # ATR period is default 14 in _get_atr
        if atr is None or atr == 0: # Also check if ATR is zero
            return
        
        position_type = position[C.POSITION_TYPE] # MT5 constant
        current_sl = position[C.POSITION_SL]

        if position_type == mt5.POSITION_TYPE_BUY:
            new_sl = current_price - (atr * atr_multiplier_for_trailing)
            if new_sl > current_sl:
                self._modify_position(position, sl=new_sl)
        else: # SELL
            new_sl = current_price + (atr * atr_multiplier_for_trailing)
            if new_sl < current_sl or current_sl == 0:
                self._modify_position(position, sl=new_sl)
    
    def _get_atr(self, data: Dict[str, pd.DataFrame], period: int = 14) -> Optional[float]:
        """
        Get ATR value from market data.
        
        Args:
            data: Market data for the symbol
            period: ATR period, default from risk config or global if not passed
            
        Returns:
            ATR value or None if not available
        """
        # Try to get ATR from the primary timeframe
        # TODO: Make primary_tf selection more robust, perhaps from config
        timeframes_list = list(self.config_manager.get_timeframes().keys())
        if not timeframes_list:
            logger.warning("No timeframes configured for ATR.")
            return None
        primary_tf = timeframes_list[0]

        if primary_tf in data and not data[primary_tf].empty:
            df = data[primary_tf]
            # Ensure indicator_period is valid
            atr_period_from_config = self.config_manager.get_indicator_params(df.name if hasattr(df,'name') else '').get(C.CONFIG_INDICATOR_ATR_PERIOD, period)

            if len(df) >= atr_period_from_config + 1:
                # Calculate ATR if not already present
                if C.INDICATOR_ATR not in df.columns: # Use constant
                    # Ensure column names are lowercase as per IndicatorCalculator convention potentially
                    high_col = 'high' if 'high' in df.columns else 'High'
                    low_col = 'low' if 'low' in df.columns else 'Low'
                    close_col = 'close' if 'close' in df.columns else 'Close'

                    high = df[high_col]
                    low = df[low_col]
                    close = df[close_col]
                    
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr_series = tr.rolling(window=atr_period_from_config).mean()
                    df[C.INDICATOR_ATR] = atr_series # Use constant
                
                return df[C.INDICATOR_ATR].iloc[-1]
        
        return None
    
    def _modify_position(self, position: Dict[str, Any], sl: Optional[float] = None, tp: Optional[float] = None) -> None:
        """
        Modify a position's stop loss or take profit.
        
        Args:
            position: Position to modify (dictionary)
            sl: New stop loss price (None to keep current)
            tp: New take profit price (None to keep current)
        """
        # If no changes, do nothing
        if sl is None and tp is None:
            return
        
        position_ticket = position[C.POSITION_TICKET]
        # Use current values if not provided
        new_sl = sl if sl is not None else position[C.POSITION_SL]
        new_tp = tp if tp is not None else position[C.POSITION_TP]
        
        # Modify the position
        result = self.mt5.modify_position(
            ticket=position_ticket,
            sl=new_sl,
            tp=new_tp
        )
        
        if result.get('retcode') != C.RETCODE_DONE:
            logger.error(f"Failed to modify position {position_ticket}: {result.get(C.REQUEST_COMMENT)}")
        else:
            logger.info(f"Modified position {position_ticket}: SL={new_sl}, TP={new_tp}")
    
    def _check_for_entries(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        symbol_config: Dict[str, Any],
        current_positions: List[Dict[str, Any]] # Recibir la lista completa de posiciones
    ) -> None:
        """
        Check for new entry signals.
        
        Args:
            symbol: Symbol to check
            data: Market data for the symbol
            symbol_config: Symbol configuration
            current_positions: List of current positions
        """
        # Check if we can trade this symbol
        # Pasar la lista completa de posiciones al risk_manager
        can_trade, reason = self.risk_manager.check_market_conditions(
            symbol=symbol,
            data=data,
            current_positions=current_positions
        )
        
        if not can_trade:
            logger.debug("Skipping %s: %s", symbol, reason)
            return
        
        # Analyze the market
        # Pass None for position_info as we are checking for new entries
        analysis = self.strategy.analyze(symbol, data, position_info=None)
        
        signal_type = analysis.get('signal', SignalType.NONE) # 'signal' is fine as string key
        if signal_type == SignalType.NONE:
            return
        
        # Get current price
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not get symbol info for {symbol} in entry check")
            return
        
        current_price = symbol_info.ask if signal_type == SignalType.BUY else symbol_info.bid
        
        # Calculate position size
        # analysis.get('stop_loss') uses string key 'stop_loss' from StrategyResult.to_dict()
        stop_loss_price = analysis.get(C.POSITION_SL, 0.0) # Use constant if defined for StrategyResult keys

        position_size_details = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=current_price,
            stop_loss=stop_loss_price,
            risk_amount=None
        )
        
        calculated_lot_size = position_size_details.get(C.LOT_SIZE, 0.0) # Use constant
        if calculated_lot_size <= 0:
            logger.warning(f"Invalid position size ({calculated_lot_size}) for {symbol}")
            return
        
        order_type_str = C.ORDER_TYPE_BUY if signal_type == SignalType.BUY else C.ORDER_TYPE_SELL
        
        # analysis.get('take_profit') uses string key 'take_profit'
        take_profit_price = analysis.get(C.POSITION_TP, 0.0) # Use constant
        message = analysis.get('message', '') # 'message' is fine as string key

        result = self.mt5.place_order( # This calls MT5Connector -> MT5TradingOperations
            symbol=symbol,
            order_type=order_type_str, # Pass 'BUY' or 'SELL' string
            lot_size=calculated_lot_size,
            sl=stop_loss_price,
            tp=take_profit_price,
            comment=f"Auto {order_type_str.capitalize()}: {message}"
        )
        
        if result.get('retcode') == C.RETCODE_DONE: # Use constant for retcode
            logger.info(f"Placed {order_type_str.upper()} order for {symbol}: "
                       f"Lots={calculated_lot_size:.2f}, SL={stop_loss_price:.5f}, TP={take_profit_price:.5f}")
            
            if hasattr(self.risk_manager, 'update_trade_count'):
                 self.risk_manager.update_trade_count()
        else:
            logger.error(f"Failed to place {order_type_str.upper()} order for {symbol}: {result.get(C.REQUEST_COMMENT)}")

def main():
    """Main entry point for the trading bot."""
    # Logging is setup after ConfigManager is initialized in TradingBot
    # logger.info("Starting trading bot...") # Moved after logging setup
    
    bot = None
    try:
        bot = TradingBot()
        # Logging setup is now inside TradingBot's _apply_reloaded_config or after its init
        # For initial setup before any potential reload, it's done in TradingBot constructor phase
        # if not hasattr(bot, 'config_manager') or bot.config_manager is None:
        #      logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #      logger.critical("ConfigManager not available in TradingBot for main. Logging setup with basic config.")
        # else:
        #      setup_logging(config_manager=bot.config_manager)
        # logger.info("Trading bot application started.") # Now logger is configured
        
        if not bot.initialize():
            logger.error("Failed to initialize trading bot")
            return 1
        
        bot.start()
        
    except Exception as e:
        logger.exception("Fatal error in trading bot")
        return 1
    
    logger.info("Trading bot stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
