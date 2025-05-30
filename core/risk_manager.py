"""
Risk management module for the trading bot.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from . import constants as C # Import constants

logger = logging.getLogger(__name__)

class RiskManager:
    """Handles risk management and position sizing."""
    
    def __init__(self, config: Dict[str, Any], account_info: Dict[str, float],
                 config_manager: Optional[Any] = None, symbol: Optional[str] = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary with risk parameters for a specific symbol or defaults.
            account_info: Dictionary with account information (balance, equity, etc.)
            config_manager: Optional ConfigManager instance for fetching updated configs.
            symbol: Optional symbol this RiskManager instance is primarily for.
        """
        self.config_manager = config_manager
        self.symbol = symbol # Symbol this instance is primarily for, if any
        self.account_info = account_info
        self._apply_config(config) # Apply initial configuration
        
        # Track daily performance
        self.daily_high_watermark = self.account_info.get('balance', 10000)
        self.daily_drawdown = 0.0
        self.daily_trades = 0
        # self.max_daily_trades is set in _apply_config

    def _apply_config(self, config: Dict[str, Any]):
        """Helper to apply risk parameters from a config dictionary."""
        self.config = config # Store current config
        self.risk_per_trade = self.config.get(C.CONFIG_RISK_PER_TRADE, 0.01)
        self.max_risk_per_trade = self.config.get(C.CONFIG_MAX_RISK_PER_TRADE, 0.02)
        self.max_daily_drawdown = self.config.get('max_daily_drawdown', 0.05) # Assuming 'max_daily_drawdown' is a specific key
        self.max_open_trades = self.config.get('max_open_trades', 5)       # Assuming 'max_open_trades' is a specific key
        self.max_position_size = self.config.get('max_position_size', 0.1)   # Assuming 'max_position_size' is a specific key
        self.max_daily_trades = self.config.get('max_daily_trades', 10)     # Assuming 'max_daily_trades' is a specific key
        logger.info(f"RiskManager config applied. RPT: {self.risk_per_trade}, Max RPT: {self.max_risk_per_trade}, Max DD: {self.max_daily_drawdown}")

    def update_config(self, symbol: Optional[str] = None):
        """
        Updates the risk manager's configuration.
        If symbol is provided and matches self.symbol, it fetches specific config for that symbol.
        Otherwise, it might refresh based on defaults if self.symbol is None.
        """
        if self.config_manager is None:
            logger.warning("RiskManager has no ConfigManager instance, cannot update config dynamically.")
            return

        target_symbol = symbol if symbol is not None else self.symbol

        if target_symbol:
            try:
                new_risk_params = self.config_manager.get_risk_params(target_symbol)
                self._apply_config(new_risk_params)
                logger.info(f"RiskManager configuration updated for symbol {target_symbol}.")
            except Exception as e:
                logger.error(f"Failed to update RiskManager config for symbol {target_symbol}: {e}")
        else:
            # If no specific symbol, this RiskManager might be using global defaults.
            # Re-fetch default risk parameters.
            try:
                default_risk_params = self.config_manager.get_defaults().get(C.CONFIG_RISK, {}) # Use C.CONFIG_RISK
                if default_risk_params:
                    self._apply_config(default_risk_params)
                    logger.info("RiskManager configuration updated using default risk parameters.")
                else:
                    logger.warning("Could not find default risk parameters to update RiskManager.")
            except Exception as e:
                logger.error(f"Failed to update RiskManager config with defaults: {e}")

    def calculate_position_size(
        self,
        symbol: str, # Symbol for which to calculate, might differ from self.symbol
        entry_price: float,
        stop_loss: float,
        risk_amount: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol for which calculation is being done.
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_amount: Optional fixed risk amount (overrides risk_per_trade if provided)
            
        Returns:
            Dictionary with position size and risk details
        """
        if entry_price <= 0 or stop_loss <= 0:
            logger.error("Invalid price or stop loss for position size calculation")
            return {'lot_size': 0.0, 'risk_amount': 0.0, 'risk_percent': 0.0}

        # Fetch the relevant risk parameters for the given symbol for this calculation
        # This ensures that even if RM instance is for 'default', it can calculate for specific symbol
        current_risk_params = self.config
        if self.config_manager and symbol != self.symbol:
            try:
                current_risk_params = self.config_manager.get_risk_params(symbol)
            except Exception as e:
                logger.warning(f"Could not fetch risk params for {symbol} during position sizing. Using RM's current config. Error: {e}")

        # Use constants for getting specific risk parameters
        risk_per_trade_val = current_risk_params.get(C.CONFIG_RISK_PER_TRADE, self.risk_per_trade)
        max_risk_per_trade_val = current_risk_params.get(C.CONFIG_MAX_RISK_PER_TRADE, self.max_risk_per_trade)
        # For max_position_size, assuming the key is 'max_position_size' directly in the risk dict
        max_position_size_pct = current_risk_params.get('max_position_size', self.max_position_size)

        balance = self.account_info.get('balance', 10000) # 'balance' is an AccountInfo object attribute name
        
        if risk_amount is None:
            risk_amount = balance * risk_per_trade_val # Use fetched value
        
        max_allowable_risk_amount = balance * max_risk_per_trade_val # Use fetched value
        risk_amount = min(risk_amount, max_allowable_risk_amount)
        
        if entry_price > stop_loss:  # Long position
            risk_per_share = entry_price - stop_loss
        else:  # Short position
            risk_per_share = stop_loss - entry_price
        
        if risk_per_share <= 0:
            logger.error(f"Invalid risk per share for {symbol} (entry: {entry_price}, stop_loss: {stop_loss})")
            return {'lot_size': 0.0, 'risk_amount': 0.0, 'risk_percent': 0.0}
        
        position_size_units = risk_amount / risk_per_share
        
        # TODO: Get contract size from MT5 or config instead of hardcoding 100,000
        lot_size = position_size_units / 100000.0
        
        max_lot_size_cap = (balance * max_position_size_pct) / entry_price / 100000.0
        lot_size = min(lot_size, max_lot_size_cap)
        
        actual_risk_amount = lot_size * 100000.0 * risk_per_share
        actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else 0
        
        # Use take_profit settings from the specific symbol's config if available
        tp_atr_multiplier = current_risk_params.get('tp_atr_multiplier', 3.0) # Default if not in config
        # Note: _calculate_take_profit might need ATR or other data, this is simplified
        # For now, it uses a fixed risk_reward_ratio.
        # A more advanced version would fetch ATR for the symbol and use tp_atr_multiplier.

        return {
            C.LOT_SIZE: round(lot_size, 2), # Use constant for key
            'risk_amount': actual_risk_amount, # These are calculated values, keys are fine
            'risk_percent': actual_risk_percent,
            'risk_per_share': risk_per_share,
            C.POSITION_SL: stop_loss, # Use constant for key
            C.POSITION_TP': self._calculate_take_profit(entry_price, stop_loss, # Use constant for key
                                                       C.ORDER_TYPE_BUY if entry_price > stop_loss else C.ORDER_TYPE_SELL,
                                                       risk_reward_ratio=tp_atr_multiplier)
        }
    
    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
            position_type: str, # Expects C.ORDER_TYPE_BUY or C.ORDER_TYPE_SELL
        risk_reward_ratio: float = 3.0
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_type: C.ORDER_TYPE_BUY or C.ORDER_TYPE_SELL string constant
            risk_reward_ratio: Desired risk-reward ratio (e.g., 3.0 for 1:3)
            
        Returns:
            Take profit price
        """
        if position_type == C.ORDER_TYPE_BUY:
            risk = entry_price - stop_loss
            return entry_price + (risk * risk_reward_ratio)
        elif position_type == C.ORDER_TYPE_SELL:
            risk = stop_loss - entry_price
            return entry_price - (risk * risk_reward_ratio)
        else: # Should not happen if called with constants
            logger.warning(f"Unknown position type '{position_type}' in _calculate_take_profit, returning entry price.")
            return entry_price
    
    def check_daily_limits(self) -> Tuple[bool, str]:
        """
        Check if daily trading limits have been reached.
        
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check daily drawdown
        current_equity = self.account_info.get('equity', 0)
        daily_pnl = current_equity - self.daily_high_watermark
        
        if daily_pnl < 0:
            self.daily_drawdown = abs(daily_pnl) / self.daily_high_watermark
            if self.daily_drawdown >= self.max_daily_drawdown:
                return False, f"Daily drawdown limit reached: {self.daily_drawdown:.2%} >= {self.max_daily_drawdown:.2%}"
        else:
            self.daily_high_watermark = current_equity
        
        # Check daily trade count
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades} >= {self.max_daily_trades}"
        
        return True, ""
    
    def update_trade_count(self) -> None:
        """Increment the daily trade counter."""
        self.daily_trades += 1
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_high_watermark = self.account_info.get('balance', 10000)
        self.daily_drawdown = 0.0
        self.daily_trades = 0
    
    def check_market_conditions(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        current_positions: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Check if market conditions are suitable for trading.
        
        Args:
            symbol: Trading symbol
            data: Dictionary of DataFrames for different timeframes
            current_positions: List of current positions
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if market is open (if using exchange data)
        # This is a placeholder - implement based on your market data source
        
        # Check if we've reached max open trades
        if len(current_positions) >= self.max_open_trades:
            return False, f"Max open trades reached: {len(current_positions)} >= {self.max_open_trades}"
        
        # Check if we already have a position in this symbol
        for position in current_positions: # position is a dict from MT5
            if position.get(C.POSITION_SYMBOL) == symbol: # Use constant
                return False, f"Already have an open position in {symbol}"
        
        # Check for high volatility (optional)
        if self._is_high_volatility(data):
            return False, "Market volatility is too high"
        
        return True, ""
    
    def _is_high_volatility(self, data: Dict[str, pd.DataFrame], threshold: float = 2.0) -> bool:
        """
        Check if market volatility is above threshold.
        
        Args:
            data: Dictionary of DataFrames for different timeframes
            threshold: Volatility threshold (multiple of average true range)
            
        Returns:
            True if volatility is high, False otherwise
        """
        # This is a simplified implementation - you might want to use ATR or other volatility measures
        if not data:
            return False
        
        # Use the primary timeframe for volatility check
        primary_tf = self.config.get('timeframes', {}).get('primary', 'M30')
        if primary_tf not in data:
            return False
        
        df = data[primary_tf].copy()
        if len(df) < 20:  # Need enough data for ATR
            return False
        
        # Calculate ATR (Average True Range)
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Check if current ATR is above threshold times the average
        current_atr = tr.iloc[-1]
        return current_atr > (atr * threshold)
    
    def update_account_info(self, account_info: Dict[str, float]) -> None:
        """
        Update account information.
        
        Args:
            account_info: Dictionary with account information
        """
        self.account_info.update(account_info)
        
        # Update daily high watermark
        equity = account_info.get('equity', 0)
        if equity > self.daily_high_watermark:
            self.daily_high_watermark = equity
