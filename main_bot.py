"""
Main trading bot module.
"""

import logging
import time
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import MetaTrader5 as mt5

# Add the current directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config_manager import ConfigManager
from core.logging_setup import setup_logging
from core.mt5_connector import MT5Connector
from core.strategy_engine import StrategyEngine, SignalType
from core.risk_manager import RiskManager

# Initialize logging
logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the trading bot.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = ConfigManager(config_path)
        
        # Initialize components
        self.mt5 = MT5Connector(self.config)
        self.strategy = StrategyEngine(self.config)
        
        # Get active symbols
        self.symbols = self._get_active_symbols()
        
        if not self.symbols:
            raise ValueError("No active symbols found in configuration")
            
        # Use the first symbol as default for risk manager
        default_symbol = next(iter(self.symbols.keys()))
            
        self.risk_manager = RiskManager(
            config=self.config.get_risk_params(default_symbol),
            account_info=self._get_account_info()
        )
        
        # Trading state
        self.is_running = False
        self.last_update = None
        self.timeframes = self.config.get_timeframes()
        
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
        for symbol, config in self.config.get_active_symbols().items():
            if config.get('enabled', False):
                active_symbols[symbol] = config
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
                time.sleep(self.config.get_global_settings().get('loop_interval', 1))
                
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the trading bot."""
        if not self.is_running:
            return
            
        logger.info("Stopping trading bot...")
        self.is_running = False
        self.mt5.disconnect()
    
    def _run_iteration(self) -> None:
        """Run one iteration of the trading loop."""
        # Reset daily stats if it's a new day
        current_date = datetime.now().date()
        if self.last_reset_date != current_date:
            logger.info("New trading day detected. Resetting daily risk statistics.")
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
            if symbol_config_dict.get('enabled', False):
                try:
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
        
        if not positions_list:  # Solo abrir nuevas posiciones si no hay abiertas para este símbolo
            # Pasamos la lista completa de posiciones abiertas al risk_manager
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
                # Acceder a ticket como clave de diccionario para el log
                position_ticket = position.get('ticket', 'N/A')
                logger.exception("Error managing position %s", position_ticket)
    
    def _manage_position(self, position: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> None:
        """
        Manage a single open position.
        
        Args:
            position: Position to manage (dictionary)
            data: Market data for the symbol
        """
        # Get current price
        symbol = position['symbol'] # Acceder a symbol como clave
        
        # Acceder a type como clave
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("Could not get symbol info for %s", symbol)
            return
            
        current_price = symbol_info.ask if position['type'] == 'BUY' else symbol_info.bid
        
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
        # Acceder a type, sl, tp como claves
        if position['type'] == 'BUY':
            if current_price <= position['sl'] or current_price >= position['tp']:
                return "SL/TP hit"
        else:  # sell
            if current_price >= position['sl'] or current_price <= position['tp']:
                return "SL/TP hit"
        
        # Check strategy exit signals
        analysis = self.strategy.analyze(
            symbol=position['symbol'], # Acceder a symbol como clave
            data=data,
            position_info={
                'position_type': position['type'], # Acceder a type como clave
                'entry_price': position['open_price'], # Acceder a open_price como clave
                'stop_loss': position['sl'], # Acceder a sl como clave
                'take_profit': position['tp'], # Acceder a tp como clave
                'current_price': current_price
            }
        )
        
        # Acceder a type como clave
        if (position['type'] == 'BUY' and analysis['signal'] == SignalType.SELL) or \
           (position['type'] == 'SELL' and analysis['signal'] == SignalType.BUY):
            return analysis.get('message', 'Strategy exit signal')
        
        return None
    
    def _close_position(self, position: Dict[str, Any], reason: str) -> None:
        """
        Close a position.
        
        Args:
            position: Position to close (dictionary)
            reason: Reason for closing
        """
        # Acceder a ticket como clave para el log y la llamada a close_position
        logger.info("Closing position %s: %s", position['ticket'], reason)
        
        # Close the position
        result = self.mt5.close_position(position['ticket']) # Acceder a ticket como clave
        
        if result.retcode != 10009:  # MT5.TRADE_RETCODE_DONE
            # Acceder a ticket como clave para el log
            logger.error("Failed to close position %s: %s", position['ticket'], result.comment)
    
    def _check_move_to_break_even(self, position: Dict[str, Any], current_price: float) -> None:
        """
        Check if we should move stop loss to break even.
        
        Args:
            position: Position to check (dictionary)
            current_price: Current market price
        """
        # Get symbol info for point value
        # Acceder a symbol como clave
        symbol_info = mt5.symbol_info(position['symbol'])
        if not symbol_info:
            logger.error("Could not get symbol info for %s in break even check", position['symbol'])
            return
        
        # Calculate price move in points
        # Acceder a type, price_open, sl como claves
        if position['type'] == 'BUY':
            price_move = (current_price - position['open_price']) / symbol_info.point
            if price_move >= 50:  # 50 points in profit
                # Move SL to break even + spread
                new_sl = position['open_price'] + (symbol_info.spread * symbol_info.point)
                if new_sl > position['sl']:  # Only move SL up. Acceder a sl como clave
                    self._modify_position(position, sl=new_sl)
        else:  # sell
            price_move = (position['open_price'] - current_price) / symbol_info.point
            if price_move >= 50:  # 50 points in profit
                # Move SL to break even - spread
                new_sl = position['open_price'] - (symbol_info.spread * symbol_info.point)
                if new_sl < position['sl'] or position['sl'] == 0:  # Only move SL down. Acceder a sl como clave
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
        atr = self._get_atr(data)
        if atr is None:
            return
        
        # Calculate new stop loss
        # Acceder a type, sl como claves
        if position['type'] == 'BUY':
            new_sl = current_price - (atr * 2)  # 2 * ATR trailing stop
            if new_sl > position['sl']:  # Only move SL up. Acceder a sl como clave
                self._modify_position(position, sl=new_sl)
        else:  # sell
            new_sl = current_price + (atr * 2)  # 2 * ATR trailing stop
            if new_sl < position['sl'] or position['sl'] == 0:  # Only move SL down. Acceder a sl como clave
                    self._modify_position(position, sl=new_sl)
    
    def _get_atr(self, data: Dict[str, pd.DataFrame], period: int = 14) -> Optional[float]:
        """
        Get ATR value from market data.
        
        Args:
            data: Market data for the symbol
            period: ATR period
            
        Returns:
            ATR value or None if not available
        """
        # Try to get ATR from the primary timeframe
        primary_tf = self.config.get_timeframes()[0]  # First timeframe is primary
        if primary_tf in data and not data[primary_tf].empty:
            df = data[primary_tf]
            if len(df) >= period + 1:
                # Calculate ATR if not already present
                if 'ATR' not in df.columns:
                    high = df['high']
                    low = df['low']
                    close = df['close']
                    
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=period).mean()
                    df['ATR'] = atr
                
                return df['ATR'].iloc[-1]
        
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
        
        # Use current values if not provided
        # Acceder a sl, tp como claves
        if sl is None:
            sl = position['sl']
        if tp is None:
            tp = position['tp']
        
        # Modify the position
        # Acceder a ticket como clave
        result = self.mt5.modify_position(
            ticket=position['ticket'],
            sl=sl,
            tp=tp
        )
        
        if result.retcode != 10009:  # MT5.TRADE_RETCODE_DONE
            # Acceder a ticket como clave para el log
            logger.error("Failed to modify position %s: %s", position['ticket'], result.comment)
        else:
            # Acceder a ticket como clave para el log
            logger.info("Modified position %s: SL=%.5f, TP=%.5f", position['ticket'], sl, tp)
    
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
        analysis = self.strategy.analyze(symbol, data)
        
        # Check for entry signals
        if analysis['signal'] == SignalType.NONE:
            return
        
        # Get current price
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error("Could not get symbol info for %s in entry check", symbol)
            return
        
        current_price = symbol_info.ask if analysis['signal'] == SignalType.BUY else symbol_info.bid
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=current_price,
            # La stop loss se obtiene del resultado del análisis de estrategia
            stop_loss=analysis.get('stop_loss', 0),
            risk_amount=None  # Usar porcentaje de riesgo por defecto
        )
        
        if position_size['lot_size'] <= 0:
            logger.warning("Invalid position size for %s", symbol)
            return
        
        # Place the order
        order_type = 'buy' if analysis['signal'] == SignalType.BUY else 'sell'
        
        result = self.mt5.place_order(
            symbol=symbol,
            order_type=order_type,
            lot_size=position_size['lot_size'],
            # La stop loss y take profit se obtienen del resultado del análisis de estrategia
            sl=analysis.get('stop_loss', 0),
            tp=analysis.get('take_profit', 0),
            comment=f"Auto {order_type.capitalize()}: {analysis.get('message', '')}"
        )
        
        if result.retcode == 10009:  # MT5.TRADE_RETCODE_DONE
            logger.info("Placed %s order for %s: Lots=%.2f, SL=%.5f, TP=%.5f", 
                       order_type.upper(), symbol, position_size['lot_size'], 
                       analysis.get('stop_loss', 0), analysis.get('take_profit', 0))
            
            # Update risk manager
            self.risk_manager.update_trade_count()
        else:
            logger.error("Failed to place %s order for %s: %s", 
                        order_type.upper(), symbol, result.comment)

def main():
    """Main entry point for the trading bot."""
    # Set up logging
    setup_logging()
    
    logger.info("Starting trading bot...")
    
    try:
        # Create and start the bot
        bot = TradingBot()
        
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
