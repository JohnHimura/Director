"""
Risk management module for the trading bot.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """Handles risk management and position sizing."""
    
    def __init__(self, config: Dict[str, Any], account_info: Dict[str, float]):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary with risk parameters
            account_info: Dictionary with account information (balance, equity, etc.)
        """
        self.config = config
        self.account_info = account_info
        
        # Initialize risk parameters with defaults
        self.risk_per_trade = config.get('risk_per_trade', 0.01)  # 1% risk per trade
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% max risk per trade
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)  # 5% max daily drawdown
        self.max_open_trades = config.get('max_open_trades', 5)  # Max open trades
        self.max_position_size = config.get('max_position_size', 0.1)  # Max position size (10% of account)
        
        # Track daily performance
        self.daily_high_watermark = account_info.get('balance', 10000)
        self.daily_drawdown = 0.0
        self.daily_trades = 0
        self.max_daily_trades = config.get('max_daily_trades', 10)
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_amount: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_amount: Optional fixed risk amount (overrides risk_per_trade if provided)
            
        Returns:
            Dictionary with position size and risk details
        """
        if entry_price <= 0 or stop_loss <= 0:
            logger.error("Invalid price or stop loss for position size calculation")
            return {'lot_size': 0.0, 'risk_amount': 0.0, 'risk_percent': 0.0}
        
        # Get account balance
        balance = self.account_info.get('balance', 10000)
        
        # Calculate risk amount if not provided
        if risk_amount is None:
            risk_amount = balance * self.risk_per_trade
        
        # Ensure risk amount doesn't exceed max risk per trade
        max_risk_amount = balance * self.max_risk_per_trade
        risk_amount = min(risk_amount, max_risk_amount)
        
        # Calculate position size
        if entry_price > stop_loss:  # Long position
            risk_per_share = entry_price - stop_loss
        else:  # Short position
            risk_per_share = stop_loss - entry_price
        
        if risk_per_share <= 0:
            logger.error("Invalid risk per share (entry: %f, stop_loss: %f)", entry_price, stop_loss)
            return {'lot_size': 0.0, 'risk_amount': 0.0, 'risk_percent': 0.0}
        
        # Calculate number of shares (or lots)
        position_size = risk_amount / risk_per_share
        
        # Convert to lots (assuming 100,000 units per standard lot)
        lot_size = position_size / 100000.0
        
        # Ensure position size doesn't exceed max position size
        max_lot_size = (balance * self.max_position_size) / entry_price / 100000.0
        lot_size = min(lot_size, max_lot_size)
        
        # Calculate actual risk amount and percentage
        actual_risk_amount = lot_size * 100000.0 * risk_per_share
        actual_risk_percent = (actual_risk_amount / balance) * 100
        
        return {
            'lot_size': lot_size,
            'risk_amount': actual_risk_amount,
            'risk_percent': actual_risk_percent,
            'risk_per_share': risk_per_share,
            'stop_loss': stop_loss,
            'take_profit': self._calculate_take_profit(entry_price, stop_loss, position_type='BUY' if entry_price > stop_loss else 'SELL')
        }
    
    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        position_type: str,
        risk_reward_ratio: float = 3.0
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_type: 'BUY' or 'SELL'
            risk_reward_ratio: Desired risk-reward ratio (e.g., 3.0 for 1:3)
            
        Returns:
            Take profit price
        """
        if position_type.upper() == 'BUY':
            risk = entry_price - stop_loss
            return entry_price + (risk * risk_reward_ratio)
        else:  # SELL
            risk = stop_loss - entry_price
            return entry_price - (risk * risk_reward_ratio)
    
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
        for position in current_positions:
            if position.get('symbol') == symbol:
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
