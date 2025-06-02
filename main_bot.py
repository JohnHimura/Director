"""
Main trading bot module.
"""

import logging
import time
import sys
import json
import uuid
from datetime import datetime, timedelta, timezone, date # Ensure date is imported
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import MetaTrader5 as mt5

# Add the current directory to the path
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config_manager import ConfigManager
from core.logging_setup import setup_logging, get_log_context
from core.mt5_connector import MT5Connector
from core.utils.error_handler import ConnectionError as MT5ConnectionError # Import specific error
from core.strategy_engine import StrategyEngine, SignalType
from core.risk_manager import RiskManager
from core.state_manager import StateManager # Import StateManager
from core import constants as C
from core.constants import CONFIG_ENABLE_NEWS_FILTER, DEFAULT_ENABLE_NEWS_FILTER

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
        self.state_manager = StateManager() # Initialize StateManager
        
        global_settings = self.config_manager.get_global_settings()

        # Kill switch state (file-based manual override)
        self.kill_switch_activated = False # This will be OR'ed with account_kill_switch_hit later
        self._kill_switch_file_path_str = global_settings.get(C.CONFIG_KILL_SWITCH_FILE_PATH, "KILL_SWITCH.txt")
        self._kill_switch_file = Path(self._kill_switch_file_path_str)
        self._kill_switch_close_positions = global_settings.get(C.CONFIG_KILL_SWITCH_CLOSE_POSITIONS, True)

        # Load persisted state variables
        self.peak_account_equity = self.state_manager.load_variable(C.STATE_PEAK_EQUITY, 0.0)
        self.account_kill_switch_hit = self.state_manager.load_variable("account_kill_switch_hit", False)

        self.initial_daily_balance_for_drawdown = self.state_manager.load_variable(
            C.STATE_INITIAL_DAILY_BALANCE, 0.0
        )
        self.daily_pnl_realized = self.state_manager.load_variable(
            C.STATE_DAILY_PNL_REALIZED, 0.0
        )
        last_reset_date_str = self.state_manager.load_variable(C.STATE_LAST_RESET_DATE)
        if last_reset_date_str:
            try:
                self.last_reset_date = date.fromisoformat(last_reset_date_str)
            except ValueError:
                logger.warning(f"Could not parse persisted last_reset_date: {last_reset_date_str}. Will reset daily stats.")
                self.last_reset_date = None
        else:
            self.last_reset_date = None

        self.daily_drawdown_limit_hit_today = False # This is session-based, but reset if date changes

        # Account Drawdown Kill-Switch config
        self.enable_account_drawdown_kill_switch = global_settings.get(
            C.CONFIG_ENABLE_ACCOUNT_DRAWDOWN_KILL_SWITCH,
            C.DEFAULT_ENABLE_ACCOUNT_DRAWDOWN_KILL_SWITCH
        )
        if self.account_kill_switch_hit: # If loaded state says it was hit
            self.kill_switch_activated = True
            logger.critical("Account Kill Switch was previously activated. Bot remains in a critical state.")

        self.max_account_drawdown_percentage = global_settings.get(
            C.CONFIG_MAX_ACCOUNT_DRAWDOWN_PERCENTAGE,
            C.DEFAULT_MAX_ACCOUNT_DRAWDOWN_PERCENTAGE
        )
        self.close_positions_on_account_kill_switch = global_settings.get(
            C.CONFIG_CLOSE_POSITIONS_ON_ACCOUNT_KILL_SWITCH,
            C.DEFAULT_CLOSE_POSITIONS_ON_ACCOUNT_KILL_SWITCH
        )

        # Daily Drawdown config
        self.enable_daily_drawdown_limit = global_settings.get(C.CONFIG_ENABLE_DAILY_DRAWDOWN_LIMIT, C.DEFAULT_ENABLE_DAILY_DRAWDOWN_LIMIT)
        self.max_daily_drawdown_percentage_config = global_settings.get(C.CONFIG_MAX_DAILY_DRAWDOWN_PERCENTAGE, C.DEFAULT_MAX_DAILY_DRAWDOWN_PERCENTAGE) # Renamed to avoid clash
        self.close_positions_on_dd_limit = global_settings.get(C.CONFIG_CLOSE_POSITIONS_ON_DAILY_DRAWDOWN_LIMIT, C.DEFAULT_CLOSE_POSITIONS_ON_DAILY_DRAWDOWN_LIMIT)

        # News Filter state
        self.enable_news_filter = global_settings.get(CONFIG_ENABLE_NEWS_FILTER, DEFAULT_ENABLE_NEWS_FILTER) # Direct import to avoid AttributeError
        self.high_impact_news_windows_config = global_settings.get(C.CONFIG_HIGH_IMPACT_NEWS_WINDOWS, [])
        self.parsed_news_windows = []
        self._parse_news_windows(self.high_impact_news_windows_config)

        # Initialize components
        self.mt5 = MT5Connector(self.config_manager, self.is_kill_switch_active)
        self.strategy = StrategyEngine(self.config_manager)
        self.symbols = self._get_active_symbols()
        
        if not self.symbols:
            raise ValueError("No active symbols found in configuration")
            
        default_symbol_for_rm = next(iter(self.symbols.keys())) if self.symbols else "EURUSD" # Fallback if no symbols
            
        self.risk_manager = RiskManager(
            config=self.config_manager.get_risk_params(default_symbol_for_rm), # Use a valid symbol or default
            account_info=self._get_account_info(), # Fetches live or default if not connected
            config_manager=self.config_manager,
            symbol=default_symbol_for_rm
        )
        
        self.is_running = False
        self.timeframes = self.config_manager.get_timeframes()
        self.data_cache = {}

        self.heartbeat_interval_seconds = global_settings.get("mt5_heartbeat_interval_seconds", 300)
        self.last_heartbeat_time = time.time()
        
        self.timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,"M2": mt5.TIMEFRAME_M2,"M3": mt5.TIMEFRAME_M3,"M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5,"M6": mt5.TIMEFRAME_M6,"M10": mt5.TIMEFRAME_M10,"M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15,"M20": mt5.TIMEFRAME_M20,"M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,"H2": mt5.TIMEFRAME_H2,"H3": mt5.TIMEFRAME_H3,"H4": mt5.TIMEFRAME_H4,
            "H6": mt5.TIMEFRAME_H6,"H8": mt5.TIMEFRAME_H8,"H12": mt5.TIMEFRAME_H12,
            "D1": mt5.TIMEFRAME_D1,"W1": mt5.TIMEFRAME_W1,"MN1": mt5.TIMEFRAME_MN1
        }

    def _get_account_info(self) -> Dict[str, float]:
        if not self.mt5.connected: # Use connected flag which is set after successful init & login
            logger.warning("MT5 not connected, returning default account info for initialization.")
            return {'balance': 10000.0, 'equity': 10000.0, 'margin': 0.0, 'free_margin': 10000.0, 'margin_level': 0.0}
        try:
            return self.mt5.get_account_info() # This now uses direct mt5.account_info()
        except (MT5ConnectionError, Exception) as e: # Catch potential errors if called when disconnected
            logger.error(f"Could not retrieve account info from MT5: {e}. Returning defaults.")
            return {'balance': 10000.0, 'equity': 10000.0, 'margin': 0.0, 'free_margin': 10000.0, 'margin_level': 0.0}

    def _get_active_symbols(self) -> Dict[str, Dict[str, Any]]:
        active_symbols = {}
        for symbol, config_data in self.config_manager.get_active_symbols().items():
            if config_data.get(C.CONFIG_ENABLED, False):
                active_symbols[symbol] = config_data
        return active_symbols
    
    def initialize(self) -> bool:
        try:
            if not self.mt5.check_connection_and_reconnect():
                logger.error("Failed to initialize MT5 connection after retries. Bot cannot start.")
                return False
            
            current_account_info = self._get_account_info()
            current_equity = current_account_info.get(C.ACCOUNT_EQUITY, 0.0)

            if self.peak_account_equity == 0.0 and current_equity > 0:
                self.peak_account_equity = current_equity
                self.state_manager.save_variable(C.STATE_PEAK_EQUITY, self.peak_account_equity)
                logger.info(f"Peak account equity initialized to current equity: {self.peak_account_equity:.2f}")

            if self.initial_daily_balance_for_drawdown == 0.0 and current_equity > 0:
                self.initial_daily_balance_for_drawdown = current_equity
                self.state_manager.save_variable(C.STATE_INITIAL_DAILY_BALANCE, self.initial_daily_balance_for_drawdown)
                logger.info(f"Initial daily balance for drawdown initialized to current equity: {self.initial_daily_balance_for_drawdown:.2f}")

            self._synchronize_trades_on_startup() # Renamed from _reconcile_positions_on_startup
            logger.info("Trading bot initialized successfully")
            return True
        except Exception as e:
            logger.exception("Error initializing trading bot")
            return False

    def _reconcile_positions_on_startup(self) -> None:
        logger.info("Reconciling positions on startup...")
        try:
            db_positions = self.state_manager.load_open_positions()
            live_positions_raw, _, _ = self.mt5.get_open_positions()

            live_positions_map = {p[C.POSITION_TICKET]: p for p in live_positions_raw}
            db_positions_map = {p[C.POSITION_TICKET]: p for p in db_positions}

            live_tickets = set(live_positions_map.keys())
            db_tickets = set(db_positions_map.keys())

            for ticket in db_tickets - live_tickets:
                closed_pos = db_positions_map[ticket]
                logger.info(f"Position {ticket} ({closed_pos[C.POSITION_SYMBOL]}) loaded from DB but not live. Assuming closed offline.",
                            extra={'symbol': closed_pos[C.POSITION_SYMBOL], 'ticket': ticket})
                self.state_manager.remove_position(ticket)

            for ticket in live_tickets - db_tickets:
                new_live_pos = live_positions_map[ticket]
                logger.info(f"Position {ticket} ({new_live_pos[C.POSITION_SYMBOL]}) is live but not in DB. Adding to state.",
                            extra={'symbol': new_live_pos[C.POSITION_SYMBOL], 'ticket': ticket})
                self.state_manager.add_position(new_live_pos)

            for ticket in live_tickets.intersection(db_tickets):
                live_pos = live_positions_map[ticket]
                db_pos = db_positions_map[ticket]
                if live_pos.get(C.POSITION_SL) != db_pos.get(C.POSITION_SL) or \
                   live_pos.get(C.POSITION_TP) != db_pos.get(C.POSITION_TP):
                    logger.info(f"Position {ticket} ({live_pos[C.POSITION_SYMBOL]}) SL/TP differs between live and DB. Updating DB.",
                                extra={'symbol': live_pos[C.POSITION_SYMBOL], 'ticket': ticket})
                    self.state_manager.update_position_sl_tp(ticket, live_pos.get(C.POSITION_SL), live_pos.get(C.POSITION_TP))
            logger.info("Position reconciliation complete.")
        except MT5ConnectionError as e:
            logger.error(f"MT5 Connection error during position reconciliation: {e}. State may be inconsistent.")
        except Exception as e:
            logger.exception("Unexpected error during position reconciliation.")

    # Renamed from _reconcile_positions_on_startup to better reflect its purpose for this subtask
    def _synchronize_trades_on_startup(self) -> None:
        logger.info("Synchronizing trades on startup...")
        try:
            db_positions = self.state_manager.load_open_positions()
            # Ensure MT5 is connected before trying to get live positions
            if not self.mt5.connected:
                logger.error("MT5 not connected. Cannot synchronize trades from live data.")
                # Potentially load from DB only and assume they are still open if bot was down briefly
                # For now, we'll just log and continue, relying on later checks if connection resumes.
                return

            live_positions_raw, _, _ = self.mt5.get_open_positions()

            live_positions_map = {p[C.POSITION_TICKET]: p for p in live_positions_raw}
            db_positions_map = {p[C.POSITION_TICKET]: p for p in db_positions} # Assuming tickets are stored with C.POSITION_TICKET

            live_tickets = set(live_positions_map.keys())
            db_tickets = set(db_positions_map.keys())

            # Trades closed while bot was offline (in DB, not live)
            for ticket in db_tickets - live_tickets:
                closed_pos = db_positions_map[ticket]
                logger.info(f"Position {ticket} ({closed_pos.get(C.POSITION_SYMBOL, 'N/A')}) in DB but not live. Removing from state.",
                            extra={'symbol': closed_pos.get(C.POSITION_SYMBOL, 'N/A'), 'ticket': ticket})
                self.state_manager.remove_position(ticket)
                # Note: Realized P&L for such trades is not updated here as we don't have close price/time.

            # Trades opened while bot was offline or not tracked (live, not in DB)
            for ticket in live_tickets - db_tickets:
                new_live_pos = live_positions_map[ticket]
                logger.info(f"Position {ticket} ({new_live_pos[C.POSITION_SYMBOL]}) live but not in DB. Adding to state.",
                            extra={'symbol': new_live_pos[C.POSITION_SYMBOL], 'ticket': ticket})
                self.state_manager.add_position(new_live_pos) # add_position uses INSERT OR REPLACE

            # Trades still open: update SL/TP in DB if different from live
            for ticket in live_tickets.intersection(db_tickets):
                live_pos = live_positions_map[ticket]
                db_pos = db_positions_map[ticket]
                # Ensure keys exist before comparison
                live_sl = live_pos.get(C.POSITION_SL)
                db_sl = db_pos.get(C.POSITION_SL)
                live_tp = live_pos.get(C.POSITION_TP)
                db_tp = db_pos.get(C.POSITION_TP)

                if live_sl != db_sl or live_tp != db_tp:
                    logger.info(f"Position {ticket} ({live_pos[C.POSITION_SYMBOL]}) SL/TP differs. Live: SL={live_sl}, TP={live_tp}. DB: SL={db_sl}, TP={db_tp}. Updating DB.",
                                extra={'symbol': live_pos[C.POSITION_SYMBOL], 'ticket': ticket})
                    self.state_manager.update_position_sl_tp(ticket, live_sl, live_tp)

            logger.info("Trade synchronization complete.")
        except MT5ConnectionError as e:
            logger.error(f"MT5 Connection error during trade synchronization: {e}. State may be inconsistent.")
        except Exception as e:
            logger.exception("Unexpected error during trade synchronization.")


    def start(self) -> None:
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        logger.info("Starting trading bot...")
        self.is_running = True
        
        try:
            # Persist initial state variables that might have been set/loaded in __init__ or initialize
            if self.peak_account_equity > 0 :
                 self.state_manager.save_variable(C.STATE_PEAK_EQUITY, self.peak_account_equity)
            if self.initial_daily_balance_for_drawdown > 0:
                 self.state_manager.save_variable(C.STATE_INITIAL_DAILY_BALANCE, self.initial_daily_balance_for_drawdown)
            if self.last_reset_date:
                 self.state_manager.save_variable(C.STATE_LAST_RESET_DATE, self.last_reset_date.isoformat())
            self.state_manager.save_variable("account_kill_switch_hit", self.account_kill_switch_hit)

            while self.is_running:
                try:
                    self._run_iteration()
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, stopping...")
                    self.stop()
                    break
                except Exception as e:
                    logger.exception("Error in main loop")
                    time.sleep(5)
                
                loop_interval_val = self.config_manager.get_global_settings().get(C.CONFIG_LOOP_INTERVAL, 1)
                time.sleep(loop_interval_val)
        finally:
            self.stop() # stop() will now handle final state saving

    def stop(self) -> None:
        if not self.is_running and not (self.mt5 and self.mt5.connected):
            logger.info("Bot already stopped or was not fully started.")
            return

        logger.info("Stopping trading bot and saving final state...")
        self.is_running = False

        try:
            if self.mt5 and self.mt5.connected:
                try:
                    live_positions_on_shutdown, _, _ = self.mt5.get_open_positions(bypass_kill_switch=True) # Bypass KS for final save
                    if live_positions_on_shutdown:
                        # Overwrite open_positions table with the definitive list from MT5
                        self.state_manager.save_open_positions(live_positions_on_shutdown)
                        logger.info(f"Saved/Replaced {len(live_positions_on_shutdown)} open positions from MT5 to database on shutdown.")
                    else:
                        # If no positions are live, ensure the DB table is also empty
                        self.state_manager.save_open_positions([])
                        logger.info("No live positions on MT5. Cleared open positions in database on shutdown.")
                except Exception as e:
                     logger.exception("Error fetching or saving live positions during shutdown.")
            else:
                logger.warning("MT5 not connected during shutdown. Cannot save definitive live positions. Database may not reflect true final state if positions were closed manually while bot was disconnected.")

            # Save other state variables
            self.state_manager.save_variable(C.STATE_PEAK_EQUITY, self.peak_account_equity)
            self.state_manager.save_variable(C.STATE_INITIAL_DAILY_BALANCE, self.initial_daily_balance_for_drawdown)
            self.state_manager.save_variable(C.STATE_DAILY_PNL_REALIZED, self.daily_pnl_realized)
            if self.last_reset_date:
                self.state_manager.save_variable(C.STATE_LAST_RESET_DATE, self.last_reset_date.isoformat())
            self.state_manager.save_variable("account_kill_switch_hit", self.account_kill_switch_hit)
            logger.info("Final bot state saved.")
        except Exception as e:
            logger.exception("Error saving state during shutdown.")
        finally:
            if self.mt5:
                self.mt5.disconnect()
    
    def _run_iteration(self) -> None:
        log_ctx = get_log_context()
        log_ctx.correlation_id = str(uuid.uuid4())

        try:
            # Check and reload config if modified
            reloaded, changes = self.config_manager.check_and_reload_config()
            if reloaded:
                # Handle config changes if necessary (e.g., update logging level, strategy params, etc.)
                # This part might need more specific implementation based on what config changes require immediate action
                if C.RELOAD_CHANGES_LOGGING_LEVEL in changes:
                    setup_logging(config_manager=self.config_manager)
                    logger.info("Logging level updated due to config reload.")
                # Example: if symbol-specific params change, potentially update strategy engine or risk manager if they hold cached config
                # For now, we assume components fetch latest config via config_manager.get_* methods when needed.

            self._check_kill_switch()

            if self.kill_switch_activated and not self.account_kill_switch_hit: # File-based KS
                 logger.critical(f"FILE KILL SWITCH ACTIVE: {self._kill_switch_file_path_str}. No new trades.")
                 return
            if self.account_kill_switch_hit: # Account KS is more severe
                 logger.critical("ACCOUNT KILL SWITCH IS ACTIVE. No further trading actions will be performed.")
                 # Potentially self.is_running = False to stop the bot entirely
                 return

            current_date_obj = datetime.now().date() # Use date object for comparison
            if self.last_reset_date != current_date_obj:
                logger.info("New trading day detected. Resetting daily P&L and drawdown status.")
                account_info_new_day = self._get_account_info()
                self.initial_daily_balance_for_drawdown = account_info_new_day.get(C.ACCOUNT_EQUITY, 0.0)
                self.daily_pnl_realized = 0.0
                self.daily_drawdown_limit_hit_today = False
                self.last_reset_date = current_date_obj

                self.state_manager.save_variable(C.STATE_INITIAL_DAILY_BALANCE, self.initial_daily_balance_for_drawdown)
                self.state_manager.save_variable(C.STATE_DAILY_PNL_REALIZED, self.daily_pnl_realized)
                self.state_manager.save_variable(C.STATE_LAST_RESET_DATE, self.last_reset_date.isoformat())
                logger.info(f"Initial balance for drawdown today: {self.initial_daily_balance_for_drawdown:.2f}. Persisted state.")

                if hasattr(self.risk_manager, 'reset_daily_stats'):
                    self.risk_manager.reset_daily_stats()

            account_info = self._get_account_info()
            current_equity = account_info.get(C.ACCOUNT_EQUITY, 0.0)

            if hasattr(self.risk_manager, 'update_account_info'):
                self.risk_manager.update_account_info(account_info)

            if self.peak_account_equity == 0.0 and current_equity > 0:
                self.peak_account_equity = current_equity
                logger.info(f"Initial peak account equity set to: {self.peak_account_equity:.2f}")
                self.state_manager.save_variable(C.STATE_PEAK_EQUITY, self.peak_account_equity)
            elif current_equity > self.peak_account_equity:
                old_peak = self.peak_account_equity
                self.peak_account_equity = current_equity
                logger.info(f"New peak account equity reached: {current_equity:.2f} (previous: {old_peak:.2f})")
                self.state_manager.save_variable(C.STATE_PEAK_EQUITY, self.peak_account_equity)

            if self.enable_account_drawdown_kill_switch and not self.account_kill_switch_hit:
                if self.peak_account_equity > 0:
                    drawdown_value = self.peak_account_equity - current_equity
                    drawdown_percentage = (drawdown_value / self.peak_account_equity) * 100.0 if self.peak_account_equity != 0 else 0

                    if drawdown_value > 0 and drawdown_percentage >= self.max_account_drawdown_percentage:
                        logger.critical(
                            f"ACCOUNT KILL SWITCH ACTIVATED! Peak Equity: {self.peak_account_equity:.2f}, "
                            f"Current Equity: {current_equity:.2f}, Drawdown: {drawdown_percentage:.2f}% "
                            f"(Threshold: {self.max_account_drawdown_percentage:.2f}%)."
                        )
                        self.account_kill_switch_hit = True
                        self.kill_switch_activated = True
                        self.state_manager.save_variable("account_kill_switch_hit", self.account_kill_switch_hit)

                        if self.close_positions_on_account_kill_switch:
                            logger.info("Account Kill Switch: Attempting to close all open positions.")
                            open_positions, _, _ = self.mt5.get_open_positions(bypass_kill_switch=True)
                            for pos_dict in open_positions:
                                self._close_position(pos_dict, "Account Kill Switch Activated", bypass_dd_check=True)

            if self.account_kill_switch_hit:
                logger.critical("Account Kill Switch is active. No further trading actions will be performed in this session.")
                return

            if self.enable_daily_drawdown_limit and not self.daily_drawdown_limit_hit_today:
                max_loss_amount = self.initial_daily_balance_for_drawdown * (self.max_daily_drawdown_percentage_config / 100.0) # Use renamed config var
                if self.daily_pnl_realized < 0 and abs(self.daily_pnl_realized) >= max_loss_amount:
                    self.daily_drawdown_limit_hit_today = True
                    logger.critical(
                        f"DAILY DRAWDOWN LIMIT REACHED! Realized P&L: {self.daily_pnl_realized:.2f}, "
                        f"Max Loss Amount: {max_loss_amount:.2f}, Initial Daily Balance: {self.initial_daily_balance_for_drawdown:.2f}"
                    )
                    if self.close_positions_on_dd_limit:
                        logger.info("Closing all open positions due to daily drawdown limit.")
                        open_positions, _, _ = self.mt5.get_open_positions(bypass_kill_switch=True)
                        for pos_dict in open_positions:
                            self._close_position(pos_dict, "Daily drawdown limit closure", bypass_dd_check=True)

            if self.daily_drawdown_limit_hit_today:
                logger.warning("Daily drawdown limit hit. No new trades will be initiated for the rest of the day.")
                return

            can_trade_risk_mgr, reason_risk_mgr = True, ""
            if hasattr(self.risk_manager, 'check_daily_limits'):
                can_trade_risk_mgr, reason_risk_mgr = self.risk_manager.check_daily_limits()

            if not can_trade_risk_mgr:
                logger.warning(f"Trading limited by RiskManager: {reason_risk_mgr}")
                return

            for symbol_name, symbol_config_dict in self.symbols.items():
                if symbol_config_dict.get(C.CONFIG_ENABLED, False):
                    if self.daily_drawdown_limit_hit_today:
                        logger.info(f"Skipping symbol {symbol_name} as daily drawdown limit is hit.", extra={'symbol': symbol_name})
                        continue
                    try:
                        self._process_symbol(symbol_name, symbol_config_dict)
                    except MT5ConnectionError as ce:
                        logger.error(f"MT5 Connection error during processing of {symbol_name}: {ce}. Will attempt reconnect on next cycle if needed.")
                    except Exception as e:
                        logger.exception(f"Error processing symbol {symbol_name}", extra={'symbol': symbol_name})

            if time.time() - self.last_heartbeat_time > self.heartbeat_interval_seconds:
                logger.info("Performing periodic MT5 connection check (heartbeat)...")
                if not self.mt5.check_connection_and_reconnect():
                    logger.error("Periodic MT5 connection check failed and could not reconnect.")
                else:
                    logger.info("Periodic MT5 connection check successful (or reconnected).")
                self.last_heartbeat_time = time.time()

        finally:
            if hasattr(log_ctx, 'correlation_id'):
                del log_ctx.correlation_id

    def is_kill_switch_active(self) -> bool:
        return self.kill_switch_activated # This flag is now a combination of file KS and account KS

    def _check_kill_switch(self) -> None:
        """
        Checks for the file-based kill switch and updates self.kill_switch_activated.
        If account_kill_switch_hit is True, self.kill_switch_activated is forced to True.
        """
        if self.account_kill_switch_hit:
            if not self.kill_switch_activated:
                self.kill_switch_activated = True
                # This log is important if account_kill_switch_hit was loaded from state
                # and kill_switch_activated wasn't set immediately in __init__ based on it.
                logger.critical("ACCOUNT KILL SWITCH is active; ensuring general kill_switch_activated is also true.")
            return # Account kill switch overrides file-based one for this session.

        # If account_kill_switch_hit is False, proceed with file-based logic
        file_exists = self._kill_switch_file.exists()

        if file_exists:
            if not self.kill_switch_activated:
                self.kill_switch_activated = True
                logger.critical(f"FILE-BASED KILL SWITCH ACTIVATED by file: {self._kill_switch_file_path_str}")
                if self._kill_switch_close_positions:
                    logger.info("File Kill switch: Attempting to close all open positions.")
                    try:
                        open_positions, _, _ = self.mt5.get_open_positions(bypass_kill_switch=True)
                        if open_positions:
                            for position_dict in open_positions:
                                ticket = position_dict[C.POSITION_TICKET]
                                symbol_of_position = position_dict[C.POSITION_SYMBOL]
                                logger.info(f"File Kill switch: Closing position {ticket} for {symbol_of_position}.", extra={'symbol': symbol_of_position, 'ticket': ticket})
                                close_result = self.mt5.close_position(ticket, bypass_kill_switch=True)
                                if close_result and close_result.get('retcode') == C.RETCODE_DONE:
                                    logger.info(f"File Kill switch: Successfully closed position {ticket}.", extra={'symbol': symbol_of_position, 'ticket': ticket})
                                    self.state_manager.remove_position(ticket)
                                else:
                                    logger.error(f"File Kill switch: Failed to close position {ticket}. Result: {close_result}", extra={'symbol': symbol_of_position, 'ticket': ticket})
                        else:
                            logger.info("File Kill switch: No open positions found to close.")
                    except Exception as e:
                        logger.exception(f"File Kill switch: Error during closing of positions: {e}")
            # If file exists and kill_switch_activated is already true, do nothing, it's already on.
        else: # File does not exist
            if self.kill_switch_activated: # And we know account_kill_switch_hit is false here
                logger.info(f"File-based Kill Switch file ({self._kill_switch_file_path_str}) removed. Deactivating general kill switch.")
                self.kill_switch_activated = False

    def _process_symbol(self, symbol: str, symbol_config: Dict[str, Any]) -> None:
        logger.debug(f"Processing symbol: {symbol}", extra={'symbol': symbol})
        
        positions_list_all, _, _ = self.mt5.get_open_positions()
        positions_list_for_symbol = [p for p in positions_list_all if p[C.POSITION_SYMBOL] == symbol]

        data = self._get_market_data(symbol)
        if not data:
            return
        
        self.data_cache[symbol] = data
        
        if positions_list_for_symbol:
            self._manage_positions(positions_list_for_symbol, data)
        
        if not positions_list_for_symbol:
            if self.kill_switch_activated: # This now covers both file and account KS
                logger.warning(f"Kill switch active, skipping new entry check for {symbol}.", extra={'symbol': symbol})
            elif self.daily_drawdown_limit_hit_today:
                logger.warning(f"Daily drawdown limit hit, skipping new entry check for {symbol}.", extra={'symbol': symbol})
            else:
                self._check_for_entries(symbol, data, symbol_config, positions_list_all)
    
    def _parse_news_windows(self, news_windows_config: List[List[str]]):
        self.parsed_news_windows = []
        for window_entry in news_windows_config:
            if len(window_entry) == 3:
                str_start, str_end, event_name = window_entry
                try:
                    start_dt = datetime.strptime(str_start, "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(str_end, "%Y-%m-%d %H:%M:%S")
                    if start_dt >= end_dt:
                        logger.error(f"Malformed news window: start time {str_start} is not before end time {str_end} for event '{event_name}'. Skipping.")
                        continue
                    self.parsed_news_windows.append((start_dt, end_dt, event_name))
                except ValueError as e:
                    logger.error(f"Malformed date string in news window entry: {window_entry}. Error: {e}. Skipping.")
            else:
                logger.error(f"Malformed news window entry (expected 3 items): {window_entry}. Skipping.")
        logger.info(f"Parsed {len(self.parsed_news_windows)} news windows.")

    def _is_within_news_blackout_period(self) -> Tuple[bool, Optional[str]]:
        current_dt = datetime.now()
        for start_dt, end_dt, event_name in self.parsed_news_windows:
            if start_dt <= current_dt <= end_dt:
                logger.info(f"Current time {current_dt.strftime('%Y-%m-%d %H:%M:%S')} is within news blackout for event: '{event_name}' ({start_dt.strftime('%Y-%m-%d %H:%M:%S')} - {end_dt.strftime('%Y-%m-%d %H:%M:%S')})")
                return True, event_name
        return False, None

    def _get_market_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data = {}
        for tf_str in self.timeframes:
            timeframe_mt5 = self.timeframe_map.get(tf_str)
            if timeframe_mt5 is None:
                logger.warning(f"Invalid timeframe configured: {tf_str} for symbol {symbol}. Skipping.", extra={'symbol': symbol})
                continue
            try:
                df = self.mt5.get_data(symbol=symbol,timeframe=timeframe_mt5, count=1000)
                if df is not None and not df.empty: data[tf_str] = df
                else: logger.warning(f"No market data returned for {symbol} on timeframe {tf_str}.", extra={'symbol': symbol, 'timeframe': tf_str})
            except Exception as e:
                logger.exception(f"Error getting data for {symbol} {tf_str}", extra={'symbol': symbol, 'timeframe': tf_str})
        if not data: logger.warning(f"No market data successfully fetched for any timeframe for symbol: {symbol}", extra={'symbol': symbol})
        return data
    
    def _manage_positions(self, positions: List[Dict[str, Any]], data: Dict[str, pd.DataFrame]) -> None:
        for position in positions:
            try:
                self._manage_position(position, data)
            except Exception as e:
                position_ticket = position.get(C.POSITION_TICKET, 'N/A')
                symbol_of_position = position.get(C.POSITION_SYMBOL, 'N/A')
                logger.exception(f"Error managing position {position_ticket}", extra={'symbol': symbol_of_position, 'ticket': position_ticket})
    
    def _manage_position(self, position: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> None:
        symbol = position[C.POSITION_SYMBOL]
        ticket = position.get(C.POSITION_TICKET)
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Could not get symbol info for {symbol}", extra={'symbol': symbol, 'ticket': ticket})
            return

        current_price = symbol_info.ask if position[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else symbol_info.bid
        
        close_signal = self._check_exit_signals(position, data, current_price)
        if close_signal:
            self._close_position(position, "Exit signal: " + close_signal)
            return

        if self._apply_time_based_exit(position, current_price, symbol_info):
            self.state_manager.remove_position(position[C.POSITION_TICKET]) # Position closed, remove from DB
            return
            
        original_sl = position[C.POSITION_SL]
        self._apply_breakeven_stop(position, current_price, symbol_info)
        if position[C.POSITION_SL] != original_sl:
             self.state_manager.update_position_sl_tp(position[C.POSITION_TICKET], position[C.POSITION_SL], position[C.POSITION_TP])
             original_sl = position[C.POSITION_SL]

        self._apply_trailing_stop_loss(position, current_price, symbol_info)
        if position[C.POSITION_SL] != original_sl:
            self.state_manager.update_position_sl_tp(position[C.POSITION_TICKET], position[C.POSITION_SL], position[C.POSITION_TP])

    def _apply_time_based_exit(self, position: Dict[str, Any], current_market_price: float, symbol_info: Any) -> bool:
        symbol = position[C.POSITION_SYMBOL]
        ticket = position.get(C.POSITION_TICKET)
        try:
            symbol_config_data = self.config_manager.get_symbol_config(symbol)
            strategy_params = symbol_config_data.get(C.CONFIG_STRATEGY_PARAMS, {})
        except Exception as e:
            logger.warning(f"Could not fetch strategy_params for {symbol} for Time-Based Exit: {e}. Using defaults.", extra={'symbol': symbol, 'ticket': ticket})
            strategy_params = {}

        enable_time_exit = strategy_params.get(C.CONFIG_ENABLE_TIME_BASED_EXIT, C.DEFAULT_ENABLE_TIME_BASED_EXIT)
        if not enable_time_exit: return False

        max_duration_hours_config = strategy_params.get(C.CONFIG_MAX_TRADE_DURATION_HOURS, C.DEFAULT_MAX_TRADE_DURATION_HOURS)
        position_open_datetime = datetime.fromtimestamp(position[C.POSITION_TIME], tz=timezone.utc)
        current_datetime = datetime.now(timezone.utc)
        duration_hours = (current_datetime - position_open_datetime).total_seconds() / 3600

        if duration_hours >= max_duration_hours_config:
            reason = f"Time-based exit after {duration_hours:.2f} hours (max: {max_duration_hours_config}h)"
            logger.info(f"Closing position {ticket} for {symbol} due to: {reason}", extra={'symbol': symbol, 'ticket': ticket})
            self._close_position(position, reason)
            return True
        return False

    def _apply_breakeven_stop(self, position: Dict[str, Any], current_market_price: float, symbol_info: Any) -> None:
        symbol = position[C.POSITION_SYMBOL]
        ticket = position.get(C.POSITION_TICKET)
        try:
            symbol_config_data = self.config_manager.get_symbol_config(symbol)
            strategy_params = symbol_config_data.get(C.CONFIG_STRATEGY_PARAMS, {})
        except Exception as e:
            logger.warning(f"Could not fetch strategy_params for {symbol} for BreakEven: {e}. Using defaults.", extra={'symbol': symbol, 'ticket': ticket})
            strategy_params = {}

        enable_be_stop = strategy_params.get(C.CONFIG_ENABLE_BREAKEVEN_STOP, C.DEFAULT_ENABLE_BREAKEVEN_STOP)
        if not enable_be_stop: return

        breakeven_pips_profit = strategy_params.get(C.CONFIG_BREAKEVEN_PIPS_PROFIT, C.DEFAULT_BREAKEVEN_PIPS_PROFIT)
        breakeven_extra_pips = strategy_params.get(C.CONFIG_BREAKEVEN_EXTRA_PIPS, C.DEFAULT_BREAKEVEN_EXTRA_PIPS)
        point_value = symbol_info.point
        if point_value == 0:
            logger.warning(f"Point value for {symbol} is 0. Cannot calculate pips for break-even stop.", extra={'symbol': symbol, 'ticket': ticket})
            return

        position_type = position[C.POSITION_TYPE]; entry_price = position[C.POSITION_OPEN_PRICE]
        current_sl = position[C.POSITION_SL]; position_ticket = position[C.POSITION_TICKET]
        current_profit_pips = 0
        if position_type == mt5.POSITION_TYPE_BUY: current_profit_pips = (current_market_price - entry_price) / point_value
        elif position_type == mt5.POSITION_TYPE_SELL: current_profit_pips = (entry_price - current_market_price) / point_value
        else: logger.warning(f"Unknown position type {position_type} for ticket {ticket}. Cannot apply break-even.", extra={'symbol': symbol, 'ticket': ticket}); return

        if current_profit_pips >= breakeven_pips_profit:
            breakeven_sl_price = 0.0; log_extras = {'symbol': symbol, 'ticket': ticket}
            if position_type == mt5.POSITION_TYPE_BUY:
                breakeven_sl_price = entry_price + (breakeven_extra_pips * point_value)
                if breakeven_sl_price > current_sl:
                    logger.info(f"Applying Break-Even SL for BUY. Profit {current_profit_pips:.2f} pips. Entry: {entry_price:.5f}, CurrSL: {current_sl:.5f}, NewSL: {breakeven_sl_price:.5f}", extra=log_extras)
                    modify_result = self.mt5.modify_position(ticket=position_ticket, sl=breakeven_sl_price)
                    if modify_result.get('retcode') == C.RETCODE_DONE: position[C.POSITION_SL] = breakeven_sl_price; logger.info(f"Successfully applied Break-Even SL for BUY. New SL: {breakeven_sl_price:.5f}", extra=log_extras)
                    else: logger.error(f"Failed to apply Break-Even SL for BUY. Result: {modify_result}", extra=log_extras)
                else: logger.debug(f"BE BUY: Conditions not met for SL update. Profit: {current_profit_pips:.2f}, Pot.SL: {breakeven_sl_price:.5f}, CurrSL: {current_sl:.5f}", extra=log_extras)
            elif position_type == mt5.POSITION_TYPE_SELL:
                breakeven_sl_price = entry_price - (breakeven_extra_pips * point_value)
                if breakeven_sl_price < current_sl or current_sl == 0.0:
                    logger.info(f"Applying Break-Even SL for SELL. Profit {current_profit_pips:.2f} pips. Entry: {entry_price:.5f}, CurrSL: {current_sl:.5f}, NewSL: {breakeven_sl_price:.5f}", extra=log_extras)
                    modify_result = self.mt5.modify_position(ticket=position_ticket, sl=breakeven_sl_price)
                    if modify_result.get('retcode') == C.RETCODE_DONE: position[C.POSITION_SL] = breakeven_sl_price; logger.info(f"Successfully applied Break-Even SL for SELL. New SL: {breakeven_sl_price:.5f}", extra=log_extras)
                    else: logger.error(f"Failed to apply Break-Even SL for SELL. Result: {modify_result}", extra=log_extras)
                else: logger.debug(f"BE SELL: Conditions not met for SL update. Profit: {current_profit_pips:.2f}, Pot.SL: {breakeven_sl_price:.5f}, CurrSL: {current_sl:.5f}", extra=log_extras)

    def _apply_trailing_stop_loss(self, position: Dict[str, Any], current_market_price: float, symbol_info: Any) -> None:
        symbol = position[C.POSITION_SYMBOL]; ticket = position.get(C.POSITION_TICKET)
        try:
            symbol_config_data = self.config_manager.get_symbol_config(symbol)
            strategy_params = symbol_config_data.get(C.CONFIG_STRATEGY_PARAMS, {})
        except Exception as e:
            logger.warning(f"Could not fetch strategy_params for {symbol} for TSL: {e}. Using defaults.", extra={'symbol': symbol, 'ticket': ticket})
            strategy_params = {}
        enable_tsl = strategy_params.get(C.CONFIG_ENABLE_TRAILING_STOP, C.DEFAULT_ENABLE_TRAILING_STOP)
        if not enable_tsl: return
        trailing_start_pips = strategy_params.get(C.CONFIG_TRAILING_START_PIPS_PROFIT, C.DEFAULT_TRAILING_START_PIPS_PROFIT)
        trailing_step_pips = strategy_params.get(C.CONFIG_TRAILING_STEP_PIPS, C.DEFAULT_TRAILING_STEP_PIPS)
        activation_dist_pips = strategy_params.get(C.CONFIG_TRAILING_ACTIVATION_PRICE_DISTANCE_PIPS, C.DEFAULT_TRAILING_ACTIVATION_PRICE_DISTANCE_PIPS)
        point_value = symbol_info.point; position_type = position[C.POSITION_TYPE]
        entry_price = position[C.POSITION_OPEN_PRICE]; current_sl = position[C.POSITION_SL]
        position_ticket = position[C.POSITION_TICKET]; current_profit_pips = 0
        if position_type == mt5.POSITION_TYPE_BUY: current_profit_pips = (current_market_price - entry_price) / point_value
        else: current_profit_pips = (entry_price - current_market_price) / point_value
        if current_profit_pips >= trailing_start_pips:
            potential_new_sl = 0.0; activation_buffer_price = activation_dist_pips * point_value; log_extras = {'symbol': symbol, 'ticket': ticket}
            if position_type == mt5.POSITION_TYPE_BUY:
                potential_new_sl = current_market_price - (trailing_step_pips * point_value)
                if potential_new_sl > (entry_price + activation_buffer_price) and potential_new_sl > current_sl:
                    logger.info(f"Trailing SL for BUY. Profit {current_profit_pips:.2f} pips. CP: {current_market_price:.5f}, CS: {current_sl:.5f}, NSL: {potential_new_sl:.5f}", extra=log_extras)
                    modify_result = self.mt5.modify_position(ticket=position_ticket, sl=potential_new_sl)
                    if modify_result.get('retcode') == C.RETCODE_DONE: position[C.POSITION_SL] = potential_new_sl; logger.info(f"Successfully applied TSL for BUY. New SL: {potential_new_sl:.5f}", extra=log_extras)
                    else: logger.error(f"Failed to apply TSL for BUY. Result: {modify_result}", extra=log_extras)
                else: logger.debug(f"TSL BUY: Conditions not met. Profit: {current_profit_pips:.2f}, Pot.SL: {potential_new_sl:.5f}, E+Act: {(entry_price + activation_buffer_price):.5f}, CS: {current_sl:.5f}", extra=log_extras)
            elif position_type == mt5.POSITION_TYPE_SELL:
                potential_new_sl = current_market_price + (trailing_step_pips * point_value)
                if potential_new_sl < (entry_price - activation_buffer_price) and (potential_new_sl < current_sl or current_sl == 0.0):
                    logger.info(f"Trailing SL for SELL. Profit {current_profit_pips:.2f} pips. CP: {current_market_price:.5f}, CS: {current_sl:.5f}, NSL: {potential_new_sl:.5f}", extra=log_extras)
                    modify_result = self.mt5.modify_position(ticket=position_ticket, sl=potential_new_sl)
                    if modify_result.get('retcode') == C.RETCODE_DONE: position[C.POSITION_SL] = potential_new_sl; logger.info(f"Successfully applied TSL for SELL. New SL: {potential_new_sl:.5f}", extra=log_extras)
                    else: logger.error(f"Failed to apply TSL for SELL. Result: {modify_result}", extra=log_extras)
                else: logger.debug(f"TSL SELL: Conditions not met. Profit: {current_profit_pips:.2f}, Pot.SL: {potential_new_sl:.5f}, E-Act: {(entry_price - activation_buffer_price):.5f}, CS: {current_sl:.5f}", extra=log_extras)

    def _check_exit_signals(self, position: Dict[str, Any], data: Dict[str, pd.DataFrame], current_price: float) -> Optional[str]:
        if position[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY:
            if (position[C.POSITION_SL] != 0 and current_price <= position[C.POSITION_SL]) or \
               (position[C.POSITION_TP] != 0 and current_price >= position[C.POSITION_TP]): return "SL/TP hit"
        elif position[C.POSITION_TYPE] == mt5.POSITION_TYPE_SELL:
            if (position[C.POSITION_SL] != 0 and current_price >= position[C.POSITION_SL]) or \
               (position[C.POSITION_TP] != 0 and current_price <= position[C.POSITION_TP]): return "SL/TP hit"

        strat_pos_type = SignalType.BUY if position[C.POSITION_TYPE] == mt5.POSITION_TYPE_BUY else SignalType.SELL
        analysis = self.strategy.analyze(symbol=position[C.POSITION_SYMBOL], data=data,
            position_info={ C.POSITION_TYPE: strat_pos_type, C.POSITION_OPEN_PRICE: position[C.POSITION_OPEN_PRICE],
                            C.POSITION_SL: position[C.POSITION_SL], C.POSITION_TP: position[C.POSITION_TP],
                            'current_price': current_price })
        if (strat_pos_type == SignalType.BUY and analysis['signal'] == SignalType.SELL) or \
           (strat_pos_type == SignalType.SELL and analysis['signal'] == SignalType.BUY):
            return analysis.get('message', 'Strategy exit signal')
        return None
    
    def _close_position(self, position: Dict[str, Any], reason: str, bypass_dd_check: bool = False) -> None: # Added bypass_dd_check
        position_ticket = position[C.POSITION_TICKET]; symbol = position[C.POSITION_SYMBOL]
        log_extras = {'symbol': symbol, 'ticket': position_ticket}
        logger.info(f"Attempting to close position {position_ticket}: {reason}", extra=log_extras)
        is_dd_closure = "Daily drawdown limit closure" in reason or "Account Kill Switch Activated" in reason

        result = self.mt5.close_position(position_ticket, bypass_kill_switch=is_dd_closure)
        if result.get('retcode') == C.RETCODE_DONE:
            closed_profit = result.get('profit', 0.0)
            self.daily_pnl_realized += closed_profit
            self.state_manager.remove_position(position_ticket) # Remove from DB after successful close
            self.state_manager.save_variable(C.STATE_DAILY_PNL_REALIZED, self.daily_pnl_realized) # Persist P&L
            logger.info(f"Successfully closed position. Profit: {closed_profit:.2f}. Updated Daily P&L: {self.daily_pnl_realized:.2f}", extra=log_extras)
        else:
            logger.error(f"Failed to close position. Result: {result.get(C.REQUEST_COMMENT)}", extra=log_extras)

    def _check_move_to_break_even(self, position: Dict[str, Any], current_price: float) -> None: # Obsolete, kept for structure if needed
        symbol = position[C.POSITION_SYMBOL]; ticket = position.get(C.POSITION_TICKET)
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info: logger.error(f"Could not get symbol info for {symbol} in break even check", extra={'symbol': symbol, 'ticket': ticket}); return
    
    def _check_trailing_stop(self, position: Dict[str, Any], current_price: float, data: Dict[str, pd.DataFrame]) -> None: # Obsolete
        symbol = position.get(C.POSITION_SYMBOL, "N/A"); ticket = position.get(C.POSITION_TICKET)
        atr = self._get_atr(data, symbol_for_log=symbol, ticket_for_log=ticket)
        if atr is None or atr == 0: return
    
    def _get_atr(self, data: Dict[str, pd.DataFrame], period: int = 14, symbol_for_log: Optional[str] = None, ticket_for_log: Optional[int] = None) -> Optional[float]:
        log_extras = {'symbol': symbol_for_log, 'ticket': ticket_for_log}
        timeframes_list = list(self.config_manager.get_timeframes().keys())
        if not timeframes_list: logger.warning("No timeframes configured for ATR.", extra=log_extras); return None
        primary_tf = timeframes_list[0]
        if primary_tf in data and not data[primary_tf].empty:
            df = data[primary_tf]
            config_symbol_context = symbol_for_log if symbol_for_log else (df.name if hasattr(df,'name') else '')
            atr_period_from_config = self.config_manager.get_indicator_params(config_symbol_context).get(C.CONFIG_INDICATOR_ATR_PERIOD, period)
            if len(df) >= atr_period_from_config + 1:
                if C.INDICATOR_ATR not in df.columns:
                    high_col='high';low_col='low';close_col='close' # Assume lowercase now
                    tr1 = df[high_col] - df[low_col]; tr2 = abs(df[high_col] - df[close_col].shift()); tr3 = abs(df[low_col] - df[close_col].shift())
                    df[C.INDICATOR_ATR] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=atr_period_from_config).mean()
                return df[C.INDICATOR_ATR].iloc[-1]
        return None
    
    def _modify_position(self, position: Dict[str, Any], sl: Optional[float] = None, tp: Optional[float] = None) -> None:
        symbol = position.get(C.POSITION_SYMBOL, "N/A"); position_ticket = position[C.POSITION_TICKET]
        log_extras = {'symbol': symbol, 'ticket': position_ticket}
        if sl is None and tp is None: return
        new_sl = sl if sl is not None else position[C.POSITION_SL]; new_tp = tp if tp is not None else position[C.POSITION_TP]
        result = self.mt5.modify_position(ticket=position_ticket, sl=new_sl, tp=new_tp)
        if result.get('retcode') != C.RETCODE_DONE:
            logger.error(f"Failed to modify position. Result: {result.get(C.REQUEST_COMMENT)}", extra=log_extras)
        else:
            logger.info(f"Modified position: SL={new_sl}, TP={new_tp}", extra=log_extras)
            # Persist SL/TP change to DB
            self.state_manager.update_position_sl_tp(position_ticket, new_sl, new_tp)
    
    def _check_for_entries(self, symbol: str, data: Dict[str, pd.DataFrame], symbol_config: Dict[str, Any], current_positions: List[Dict[str, Any]]) -> None:
        log_extras_symbol = {'symbol': symbol}
        if self.enable_news_filter:
            is_blackout, event_name = self._is_within_news_blackout_period()
            if is_blackout: logger.warning(f"Skipping new entry check for {symbol} due to news event: {event_name}.", extra=log_extras_symbol); return
        can_trade, reason = self.risk_manager.check_market_conditions(symbol=symbol, data=data, current_positions=current_positions)
        if not can_trade: logger.debug(f"Skipping {symbol} due to market conditions or risk limits: {reason}", extra=log_extras_symbol); return
        analysis = self.strategy.analyze(symbol, data, position_info=None)
        signal_type = analysis.get('signal', SignalType.NONE)
        if signal_type == SignalType.NONE: return
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info: logger.error(f"Could not get symbol info for {symbol} in entry check", extra=log_extras_symbol); return
        current_price = symbol_info.ask if signal_type == SignalType.BUY else symbol_info.bid
        stop_loss_price = analysis.get(C.POSITION_SL, 0.0)
        
        # Verificar que el stop_loss_price sea válido antes de continuar
        if stop_loss_price == 0.0:
            logger.warning(f"Stop loss price is 0.0 for {symbol}. Skipping trade entry. Analysis message: {analysis.get('message')}", extra=log_extras_symbol)
            return
            
        position_size_details = self.risk_manager.calculate_position_size(symbol=symbol, entry_price=current_price, stop_loss=stop_loss_price, risk_amount=None)
        calculated_lot_size = position_size_details.get(C.LOT_SIZE, 0.0)
        if calculated_lot_size <= 0: logger.warning(f"Invalid position size ({calculated_lot_size}) for {symbol}", extra=log_extras_symbol); return
        order_type_str = C.ORDER_TYPE_BUY if signal_type == SignalType.BUY else C.ORDER_TYPE_SELL
        take_profit_price = analysis.get(C.POSITION_TP, 0.0); message = analysis.get('message', '')
        result = self.mt5.place_order(symbol=symbol, order_type=order_type_str, lot_size=calculated_lot_size, sl=stop_loss_price, tp=take_profit_price, comment=f"Auto {order_type_str.capitalize()}: {message}")

        log_extras_trade = {**log_extras_symbol, 'ticket': result.get(C.POSITION_TICKET)}
        if result.get('retcode') == C.RETCODE_DONE and result.get(C.POSITION_TICKET) > 0:
            logger.info(f"Placed {order_type_str.upper()} order: Lots={calculated_lot_size:.2f}, SL={stop_loss_price:.5f}, TP={take_profit_price:.5f}", extra=log_extras_trade)
            
            time.sleep(0.5)
            new_positions, _, _ = self.mt5.get_open_positions(symbol=symbol)
            added_to_db = False
            for pos in new_positions:
                # MT5 order ticket from result.order is sometimes different from position ticket.
                # Matching by magic number and symbol is more robust if order comment isn't unique.
                # For now, assuming ticket in result is the position ticket or that we find it.
                # A better way would be to get order details then deal details then position ticket.
                # For simplicity, if only one new position for symbol, assume it's ours.
                # This part is tricky without reliable direct position ticket from order_send result.
                # Using a placeholder logic: if a position matches key params.
                # This might need refinement based on how MT5 returns order vs position tickets.
                # For now, just adding the first new position if specific ticket matching fails.
                # This is a common issue; MT5's order ticket from order_send is not always the position ticket.
                # A robust way is to query positions by symbol and find one that wasn't there before,
                # or one with a matching magic number and recent open time.
                 # The result.get(C.POSITION_TICKET) from place_order is the order ticket.
                 # We need to find the position opened by this order.
                 # This can be complex due to partial fills or if order ticket != position ticket.
                 # For simplicity, assume order ticket can be used to find the position, or use magic number.
                 # A more robust way is to query mt5.history_deals_get(order=order_ticket)
                 # then mt5.positions_get(ticket=deal.position_id).

                 # Simple approach: query positions for the symbol and find the one matching the order.
                 # This assumes that `place_order` returns the order ticket in `result[C.POSITION_TICKET]`
                 order_ticket = result.get(C.POSITION_TICKET)
                 # Wait a brief moment for the position to be reflected by MT5
                 time.sleep(1.0) # Adjust as needed, or implement a retry loop for fetching position

                 # Attempt to find the position by the order ticket (if MT5 links them directly, which is not always true)
                 # or by magic number if order ticket is not the position ticket.
                 # For now, we'll assume the 'ticket' in the result is the position ticket.
                 # This is a known simplification.
                 # A more robust method would be to iterate `new_positions` and match by magic number, symbol, and open time.

                 # Fetch position details using the order ticket if that's how MT5 works, or find by other means.
                 # For this example, we'll assume the result[C.POSITION_TICKET] IS the position ticket.
                 # If not, this logic needs to be much more robust (e.g., find by magic number and symbol).
                 # The current mt5.place_order returns result.order as the ticket.
                 # We need to fetch the position details using this ticket.

                 # Simplistic: Try to find the position using the order ticket directly.
                 # This might not be the actual position ticket.
                 # A better way: after order_send, query positions by symbol and magic number,
                 # and find the newest one or match by volume if only one such position exists.

                 # For now, we assume self.mt5.get_positions can take a ticket argument that is an order ticket
                 # to find the resulting position(s). This is not standard MT5 behavior.
                 # The most reliable way is to iterate all open positions for the symbol and match
                 # by magic number and time, or to use deal/order history to link.

                 # Let's refine: after placing an order, query all positions for the symbol.
                 # If there's a new one matching the magic number, consider it ours.
                 all_positions_after_order, _, _ = self.mt5.get_open_positions(symbol=symbol)
                 found_new_position = None
                 # A very basic way to find the new position:
                 # Iterate `all_positions_after_order`, compare with `current_positions` (passed to _check_for_entries)
                 # This is still complex. For now, we will just log and rely on startup reconciliation or shutdown save.
                 # A proper implementation would be:
                 # 1. Get order ticket from `result`.
                 # 2. Use `mt5.history_deals_get(order=order_ticket)` to get deal(s).
                 # 3. Use `deal.position_id` to get the actual position ticket.
                 # 4. Use `mt5.positions_get(ticket=position_id)` to get full position details.
                 # 5. Then save THAT to state_manager.

                 # For now, we'll add a placeholder log and improve this if direct position info isn't available from result.
                 # The current `place_order` in `trading_operations.py` returns `result.order` as `C.POSITION_TICKET`.
                 # This is the order ticket. We need the position ticket.
                 # This is a known difficulty with MT5 API.
                 # For this iteration, we will not save immediately after opening due to this complexity,
                 # and will rely on the save during shutdown or next reconciliation.
                 # This means if bot crashes before shutdown, newly opened trade might not be in DB.
                 logger.info(f"Order {order_ticket} placed. Position details will be saved by reconciliation/shutdown.", extra=log_extras_trade)
                 # To properly implement:
                 # new_pos_details = self.mt5.get_position_details_by_order_ticket(order_ticket) # This method needs to be created
                 # if new_pos_details:
                 #    self.state_manager.add_position(new_pos_details)


            if hasattr(self.risk_manager, 'update_trade_count'):
                 self.risk_manager.update_trade_count()
        else:
            logger.error(f"Failed to place {order_type_str.upper()} order. Result: {result.get(C.REQUEST_COMMENT)}", extra=log_extras_trade)

def main():
    bot = None
    try:
        # Setup logging early if possible, but ConfigManager needs to load first for full config
        # BasicConfig can be used if complex setup fails
        try:
            # Attempt to setup logging with default path first, TradingBot will re-setup if path differs
            temp_config_for_log = ConfigManager("config.json") # Temporary for initial log setup
            setup_logging(config_manager=temp_config_for_log)
        except Exception as log_e:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logging.error(f"Initial logging setup failed: {log_e}. Using basic logging.")

        bot = TradingBot()
        if not bot.initialize():
            logger.error("Failed to initialize trading bot. Exiting.")
            sys.exit(1) # Use sys.exit for proper exit codes
        
        bot.start()
        
    except Exception as e:
        logger.exception("Fatal error in trading bot")
        sys.exit(1) # Use sys.exit
    
    logger.info("Trading bot stopped")
    sys.exit(0) # Use sys.exit

if __name__ == "__main__":
    main() # Call main directly

