"""
Strategy engine for the trading bot.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Protocol, Type
import pandas as pd
import numpy as np
import abc

from .indicator_calculator import IndicatorCalculator
from .sr_handler import SRHandler, SRLevel

logger = logging.getLogger(__name__)

class SignalType:
    """Enum for signal types."""
    NONE = 0
    BUY = 1
    SELL = -1

class StrategyResult:
    """Class to hold strategy analysis results."""
    
    def __init__(
        self,
        signal: int = SignalType.NONE,
        signal_strength: float = 0.0,
        message: str = "",
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        indicators: Dict[str, Any] = None,
        levels: Dict[str, List[float]] = None,
        trend_direction: int = 0
    ):
        self.signal = signal
        self.signal_strength = signal_strength
        self.message = message
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.indicators = indicators or {}
        self.levels = levels or {"support": [], "resistance": []}
        self.trend_direction = trend_direction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'message': self.message,
            'indicators': self.indicators,
            'levels': self.levels,
            'trend_direction': self.trend_direction
        }

class Strategy(abc.ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indicator_calc = IndicatorCalculator(config)
        self.sr_handler = SRHandler(config)
        
        # Initialize state
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
    
    @abc.abstractmethod
    def analyze(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        position_info: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """
        Analyze the market and generate trading signals.
        
        Args:
            symbol: Trading symbol
            data: Dictionary of DataFrames for different timeframes
            position_info: Information about current positions (if any)
            
        Returns:
            StrategyResult with analysis results and signals
        """
        pass
    
    def get_trend_direction(self, data: Dict[str, pd.DataFrame]) -> int:
        """
        Determine the overall trend direction from higher timeframes.
        
        Args:
            data: Dictionary of DataFrames for different timeframes
            
        Returns:
            1 for uptrend, -1 for downtrend, 0 for neutral
        """
        # Default to neutral
        trend = 0
        
        # Check H4 and D1 timeframes if available
        for tf in ['H4', 'D1']:
            if tf in data:
                df = data[tf].copy()
                
                # Normalize column names to lowercase
                df.columns = [col.lower() for col in df.columns]
                
                self.indicator_calc.calculate_all(df)
                
                # Simple trend detection using EMA crossover
                if 'ema_50' in df.columns and 'ema_200' in df.columns:
                    ema_50 = df['ema_50'].iloc[-1]
                    ema_200 = df['ema_200'].iloc[-1]
                    
                    if ema_50 > ema_200:
                        trend += 1
                    elif ema_50 < ema_200:
                        trend -= 1
        
        # Normalize trend
        if trend > 0:
            return 1
        elif trend < 0:
            return -1
        return 0

class MACDStrategy(Strategy):
    """MACD-based trading strategy."""
    
    def analyze(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        position_info: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """
        Analyze the market using MACD strategy.
        
        Args:
            symbol: Trading symbol
            data: Dictionary of DataFrames for different timeframes
            position_info: Information about current positions (if any)
            
        Returns:
            StrategyResult with analysis results
        """
        if not data:
            return StrategyResult(message="No data provided")
        
        # Get timeframes from config
        timeframes_config = self.config.get_timeframes()
        primary_tf = list(timeframes_config.keys())[0] if timeframes_config else 'M30'
        
        if primary_tf not in data:
            logger.error("Primary timeframe %s not found in data", primary_tf)
            return StrategyResult(message=f"Primary timeframe {primary_tf} not found")
        
        df = data[primary_tf].copy()
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate indicators
        indicator_params = self.config.get_indicator_params(symbol)
        df = self.indicator_calc.calculate_all(df, indicator_params)
        
        # Get support/resistance levels
        sr_levels = self.sr_handler.get_sr_levels(df)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Get trend direction from higher timeframes
        trend_direction = self.get_trend_direction(data)
        
        # Check for entry signals
        signal, signal_strength, message = self._check_entry_signal(
            df=df,
            current_price=current_price,
            trend_direction=trend_direction,
            sr_levels=sr_levels,
            position_info=position_info
        )
        
        # Check for exit signals if in a position
        if position_info and position_info.get('position_type'):
            exit_signal, exit_message = self._check_exit_signal(
                df=df,
                position_info=position_info,
                current_price=current_price
            )
            
            if exit_signal != SignalType.NONE:
                signal = exit_signal
                message = exit_message
        
        # Prepare indicators dict
        indicators_dict = {
            'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None,
            'macd': df['macd'].iloc[-1] if 'macd' in df.columns else None,
            'macd_signal': df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else None,
            'atr': df['atr'].iloc[-1] if 'atr' in df.columns else None,
        }
        
        # Prepare levels dict
        levels_dict = {
            'support': [lvl.price for lvl in sr_levels if lvl.type == 'support'],
            'resistance': [lvl.price for lvl in sr_levels if lvl.type == 'resistance'],
        }
        
        # Create and return result
        result = StrategyResult(
            signal=signal,
            signal_strength=signal_strength,
            message=message,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            indicators=indicators_dict,
            levels=levels_dict,
            trend_direction=trend_direction
        )
        
        return result
    
    def _check_entry_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        trend_direction: int,
        sr_levels: List[SRLevel],
        position_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, float, str]:
        """
        Check for entry signals based on MACD strategy rules.
        
        Args:
            df: DataFrame with indicator data
            current_price: Current market price
            trend_direction: Overall trend direction (1=up, -1=down, 0=neutral)
            sr_levels: List of support/resistance levels
            position_info: Information about current positions (if any)
            
        Returns:
            Tuple of (signal, strength, message)
        """
        # Skip if already in a position
        if position_info and position_info.get('position_type'):
            return SignalType.NONE, 0.0, "Already in a position"
        
        # Check if we have required indicators
        required_indicators = ['rsi', 'macd', 'macd_signal', 'atr']
        if not all(ind in df.columns for ind in required_indicators):
            return SignalType.NONE, 0.0, "Missing required indicators"
        
        # Get indicator values
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_prev = df['macd'].iloc[-2] if len(df) > 1 else 0
        macd_signal_prev = df['macd_signal'].iloc[-2] if len(df) > 1 else 0
        atr = df['atr'].iloc[-1]
        
        # Check for MACD crossover
        macd_cross_up = macd_prev < macd_signal_prev and macd > macd_signal
        macd_cross_down = macd_prev > macd_signal_prev and macd < macd_signal
        
        # Check RSI conditions
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # Check price relative to support/resistance
        support_levels = [lvl for lvl in sr_levels if lvl.type == 'support']
        resistance_levels = [lvl for lvl in sr_levels if lvl.type == 'resistance']
        
        # Find nearest support/resistance
        nearest_support = max((lvl for lvl in support_levels if lvl.price < current_price), 
                             key=lambda x: x.price, default=None)
        nearest_resistance = min((lvl for lvl in resistance_levels if lvl.price > current_price), 
                                key=lambda x: x.price, default=None)
        
        # Calculate distance to nearest levels
        support_dist = (current_price - nearest_support.price) / current_price * 100 if nearest_support else float('inf')
        resistance_dist = (nearest_resistance.price - current_price) / current_price * 100 if nearest_resistance else float('inf')
        
        # Check for buy signal
        buy_conditions = [
            macd_cross_up,
            rsi_oversold or (30 <= rsi <= 70 and trend_direction >= 0),
            support_dist < 0.5,  # Within 0.5% of support
            (trend_direction >= 0)  # Uptrend or neutral
        ]
        
        # Check for sell signal
        sell_conditions = [
            macd_cross_down,
            rsi_overbought or (30 <= rsi <= 70 and trend_direction <= 0),
            resistance_dist < 0.5,  # Within 0.5% of resistance
            (trend_direction <= 0)  # Downtrend or neutral
        ]
        
        # Calculate signal strength (0-1)
        buy_strength = sum(1 for cond in buy_conditions if cond) / len(buy_conditions)
        sell_strength = sum(1 for cond in sell_conditions if cond) / len(sell_conditions)
        
        # Only take signals with sufficient strength
        strategy_settings = self.config.get_global_settings().get('strategy', {})
        min_strength = strategy_settings.get('min_signal_strength', 0.7)
        
        if buy_strength >= min_strength and buy_strength > sell_strength:
            # Calculate risk/reward
            if nearest_support and atr > 0:
                stop_loss = min(nearest_support.price, current_price - 2 * atr)
                take_profit = current_price + 3 * (current_price - stop_loss)
                
                # Update instance variables
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                
                return SignalType.BUY, buy_strength, "Buy signal: MACD cross up, near support"
            
        elif sell_strength >= min_strength:
            # Calculate risk/reward
            if nearest_resistance and atr > 0:
                stop_loss = max(nearest_resistance.price, current_price + 2 * atr)
                take_profit = current_price - 3 * (stop_loss - current_price)
                
                # Update instance variables
                self.entry_price = current_price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                
                return SignalType.SELL, sell_strength, "Sell signal: MACD cross down, near resistance"
        
        return SignalType.NONE, 0.0, "No clear signal"
    
    def _check_exit_signal(
        self,
        df: pd.DataFrame,
        position_info: Dict[str, Any],
        current_price: float
    ) -> Tuple[int, str]:
        """
        Check for exit signals based on strategy rules.
        
        Args:
            df: DataFrame with indicator data
            position_info: Information about current position
            current_price: Current market price
            
        Returns:
            Tuple of (signal, message)
        """
        if not position_info or 'position_type' not in position_info:
            return SignalType.NONE, "No position information"
        
        position_type = position_info['position_type']
        entry_price = position_info.get('entry_price', 0)
        stop_loss = position_info.get('stop_loss', 0)
        take_profit = position_info.get('take_profit', 0)
        
        # Check stop loss and take profit
        if position_type == 'BUY':
            if current_price <= stop_loss:
                return SignalType.SELL, "Stop loss hit"
            elif current_price >= take_profit:
                return SignalType.SELL, "Take profit hit"
                
            # Check for exit conditions (e.g., RSI overbought, MACD cross down)
            if 'rsi' in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
                rsi = df['rsi'].iloc[-1]
                macd = df['macd'].iloc[-1]
                macd_prev = df['macd'].iloc[-2] if len(df) > 1 else 0
                macd_signal = df['macd_signal'].iloc[-1]
                macd_signal_prev = df['macd_signal'].iloc[-2] if len(df) > 1 else 0
                
                # Check for MACD cross down
                if macd_prev > macd_signal_prev and macd < macd_signal:
                    return SignalType.SELL, "Exit: MACD cross down"
                
                # Check for RSI overbought
                if rsi > 70:
                    return SignalType.SELL, "Exit: RSI overbought"
                    
        elif position_type == 'SELL':
            if current_price >= stop_loss:
                return SignalType.BUY, "Stop loss hit"
            elif current_price <= take_profit:
                return SignalType.BUY, "Take profit hit"
                
            # Check for exit conditions (e.g., RSI oversold, MACD cross up)
            if 'rsi' in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
                rsi = df['rsi'].iloc[-1]
                macd = df['macd'].iloc[-1]
                macd_prev = df['macd'].iloc[-2] if len(df) > 1 else 0
                macd_signal = df['macd_signal'].iloc[-1]
                macd_signal_prev = df['macd_signal'].iloc[-2] if len(df) > 1 else 0
                
                # Check for MACD cross up
                if macd_prev < macd_signal_prev and macd > macd_signal:
                    return SignalType.BUY, "Exit: MACD cross up"
                
                # Check for RSI oversold
                if rsi < 30:
                    return SignalType.BUY, "Exit: RSI oversold"
        
        return SignalType.NONE, "No exit signal"

class StrategyEngine:
    """Implements the trading strategy logic."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize state
        self.last_signal = SignalType.NONE
        
        # Load strategy
        strategy_name = config.get_global_settings().get('strategy', {}).get('type', 'MACD')
        self.strategy = self._create_strategy(strategy_name, config)
    
    def _create_strategy(self, strategy_name: str, config: Dict[str, Any]) -> Strategy:
        """
        Create a strategy instance based on name.
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary
            
        Returns:
            Strategy instance
        """
        # Strategy factory - add more strategies here as needed
        strategies = {
            'MACD': MACDStrategy,
        }
        
        strategy_class = strategies.get(strategy_name, MACDStrategy)
        return strategy_class(config)
    
    def analyze(
        self, 
        symbol: str, 
        data: Dict[str, pd.DataFrame],
        position_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the market and generate trading signals.
        
        Args:
            symbol: Trading symbol
            data: Dictionary of DataFrames for different timeframes
            position_info: Information about current positions (if any)
            
        Returns:
            Dictionary with analysis results and signals
        """
        # Delegate to strategy
        result = self.strategy.analyze(symbol, data, position_info)
        
        # Update last signal
        if result.signal != SignalType.NONE:
            self.last_signal = result.signal
        
        # Return as dictionary
        return result.to_dict()
