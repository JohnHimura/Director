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
from . import constants as C # Import constants

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
        self.levels = levels or {C.SR_SUPPORT: [], C.SR_RESISTANCE: []} # Use constants
        self.trend_direction = trend_direction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        # Using string literals for dict keys is standard for JSON/API responses.
        # Constants might be overkill here unless these keys are used programmatically elsewhere.
        # For now, keeping them as strings for API compatibility.
        return {
            'signal': self.signal, # This key is part of the API
            'signal_strength': self.signal_strength, # API
            'message': self.message, # API
            C.CONFIG_INDICATORS: self.indicators, # Use constant if this key is used internally
            'levels': self.levels, # API
            'trend_direction': self.trend_direction # API
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

    def update_config(self, symbol_specific_config: Dict[str, Any]):
        """
        Update the strategy's configuration, typically for a specific symbol's parameters.
        This method can be overridden by specific strategies if they need to re-initialize
        components or clear caches based on new config.

        Args:
            symbol_specific_config: The new symbol-specific part of the configuration.
        """
        # Default implementation: just update the relevant part of self.config
        # Strategies might have symbol-specific configs stored, or use a general one.
        # This assumes self.config holds the parameters used by indicator_calc and sr_handler.
        # If a strategy directly uses parts of self.config['symbols'][symbol], this needs adjustment.
        # For now, let's assume indicator_params and sr_params are what might change.

        # This is a simplified approach. A more robust way would be for StrategyEngine
        # to pass the FULL updated ConfigManager object or specific parts of it.
        # For now, we assume that the 'config' passed to Strategy init is the defaults,
        # and specific calls to analyze might use symbol-specific params from ConfigManager.
        # This update_config is more about internal state if a strategy caches something.

        # Let's refine this: The Strategy instance is usually generic.
        # Specific parameters for a symbol are fetched by ConfigManager.get_indicator_params(symbol)
        # So, IndicatorCalculator and SRHandler inside the strategy *should* be getting fresh
        # params when analyze is called, if ConfigManager is used correctly within analyze.
        # However, if the strategy itself caches symbol-specific params, this method is useful.
        # The current MACDStrategy re-fetches params in analyze via self.config.get_indicator_params(symbol)
        # where self.config is the ConfigManager instance. So, direct update might not be strictly needed
        # for MACDStrategy as written, IF its self.config is THE ConfigManager instance.

        # Let's assume for now that the 'config' object held by the strategy IS the ConfigManager.
        # If so, no specific action is needed here as ConfigManager itself is reloaded.
        # But if StrategyEngine gives it a *copy* of config, then it needs updating.

        # The current StrategyEngine.__init__ passes 'config' which is the ConfigManager instance.
        # self.strategy = self._create_strategy(strategy_name, config)
        # And MACDStrategy.analyze uses self.config.get_indicator_params(symbol), which is good.
        # So, for MACDStrategy, direct updates via a special method might not be strictly necessary
        # as long as its self.config object is the one being updated (which it is, via ConfigManager reload).

        # What *could* change are things like strategy-level settings if they were copied at init.
        # e.g. strategy_settings = self.config.get_global_settings().get('strategy', {})
        # If min_signal_strength is cached in __init__, it needs update.
        # MACDStrategy._check_entry_signal does this:
        # strategy_settings = self.config.get_global_settings().get('strategy', {})
        # min_strength = strategy_settings.get('min_signal_strength', 0.7)
        # This is fetched live, so it's fine.

        # Conclusion: For current MACDStrategy, no specific internal state needs updating
        # beyond what ConfigManager reload + cache clearing already handles,
        # because it fetches necessary params from ConfigManager live during analyze.
        # This method is a placeholder for strategies that *do* cache parameters.
        logger.info(f"Strategy {self.__class__.__name__} received config update. Current implementation relies on live fetch from ConfigManager.")
        # Example if it cached something:
        # self.min_strength = self.config.get_global_settings().get('strategy', {}).get('min_signal_strength', 0.7)

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
                if C.INDICATOR_EMA_50 in df.columns and C.INDICATOR_EMA_200 in df.columns:
                    ema_50 = df[C.INDICATOR_EMA_50].iloc[-1]
                    ema_200 = df[C.INDICATOR_EMA_200].iloc[-1]
                    
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
        timeframes_config = self.config.get_timeframes() # ConfigManager method
        # Use a more robust way to get primary_tf, perhaps from a specific config key or default.
        # For now, assuming the first one is primary if TF config is a dict.
        # If it's a list, then list(timeframes_config)[0] would be fine.
        # If it's a dict like {"M15": "mt5.TIMEFRAME_M15"}, then list(timeframes_config.keys())[0]
        primary_tf_keys = list(timeframes_config.keys())
        primary_tf = primary_tf_keys[0] if primary_tf_keys else C.TF_M15 if hasattr(C, 'TF_M15') else 'M15' # Fallback

        if primary_tf not in data:
            logger.error(f"Primary timeframe {primary_tf} not found in data for symbol {symbol}")
            return StrategyResult(message=f"Primary timeframe {primary_tf} not found for {symbol}")
        
        df = data[primary_tf].copy()
        
        # Normalize column names to lowercase - good practice
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate indicators
        indicator_params = self.config.get_indicator_params(symbol)
        df = self.indicator_calc.calculate_all(df, indicator_params)
        
        # Get support/resistance levels
        sr_levels = self.sr_handler.get_sr_levels(df)
        
        # Get current price
        current_price = df[C.INDICATOR_CLOSE_PRICE if hasattr(C, 'INDICATOR_CLOSE_PRICE') else 'close'].iloc[-1] # Assuming 'close' is always the key
        
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
        if position_info and position_info.get(C.POSITION_TYPE):
            exit_signal, exit_message = self._check_exit_signal(
                df=df,
                position_info=position_info,
                current_price=current_price
            )
            
            if exit_signal != SignalType.NONE: # SignalType is an enum, not string
                signal = exit_signal
                message = exit_message
        
        # Prepare indicators dict
        indicators_dict = {
            C.INDICATOR_RSI: df[C.INDICATOR_RSI].iloc[-1] if C.INDICATOR_RSI in df.columns else None,
            C.INDICATOR_MACD: df[C.INDICATOR_MACD].iloc[-1] if C.INDICATOR_MACD in df.columns else None,
            C.INDICATOR_MACD_SIGNAL_LINE: df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1] if C.INDICATOR_MACD_SIGNAL_LINE in df.columns else None,
            C.INDICATOR_ATR: df[C.INDICATOR_ATR].iloc[-1] if C.INDICATOR_ATR in df.columns else None,
        }
        
        # Prepare levels dict
        levels_dict = {
            C.SR_SUPPORT: [lvl.price for lvl in sr_levels if lvl.type == C.SR_SUPPORT],
            C.SR_RESISTANCE: [lvl.price for lvl in sr_levels if lvl.type == C.SR_RESISTANCE],
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
        if position_info and position_info.get(C.POSITION_TYPE):
            return SignalType.NONE, 0.0, "Already in a position"
        
        # Check if we have required indicators
        required_indicators = [C.INDICATOR_RSI, C.INDICATOR_MACD, C.INDICATOR_MACD_SIGNAL_LINE, C.INDICATOR_ATR]
        if not all(ind in df.columns for ind in required_indicators):
            # Log which indicators are missing for easier debugging
            missing_inds = [ind for ind in required_indicators if ind not in df.columns]
            logger.warning(f"Missing required indicators for entry signal: {missing_inds}. Available: {df.columns.tolist()}")
            return SignalType.NONE, 0.0, f"Missing required indicators: {missing_inds}"
        
        # Get indicator values
        rsi = df[C.INDICATOR_RSI].iloc[-1]
        macd = df[C.INDICATOR_MACD].iloc[-1]
        macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1] # Renamed for clarity vs signal variable
        macd_prev = df[C.INDICATOR_MACD].iloc[-2] if len(df) > 1 else 0
        macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if len(df) > 1 else 0 # Renamed
        atr = df[C.INDICATOR_ATR].iloc[-1]
        
        # Check for MACD crossover
        macd_cross_up = macd_prev < macd_s_line_prev and macd > macd_s_line
        macd_cross_down = macd_prev > macd_s_line_prev and macd < macd_s_line
        
        # Check RSI conditions
        rsi_oversold = rsi < 30 # Potentially make 30 a constant C.RSI_OVERSOLD_THRESHOLD
        rsi_overbought = rsi > 70 # Potentially make 70 a constant C.RSI_OVERBOUGHT_THRESHOLD
        
        # Check price relative to support/resistance
        support_levels = [lvl for lvl in sr_levels if lvl.type == C.SR_SUPPORT]
        resistance_levels = [lvl for lvl in sr_levels if lvl.type == C.SR_RESISTANCE]
        
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
        strategy_settings = self.config.get_global_settings().get(C.CONFIG_STRATEGY, {})
        min_strength = strategy_settings.get(C.CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH, 0.7)
        
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
        # Use constants for accessing position_info dictionary
        if not position_info or C.POSITION_TYPE not in position_info:
            return SignalType.NONE, "No position information"
        
        pos_type = position_info[C.POSITION_TYPE] # BUY or SELL (from SignalType usually)
        # entry_price = position_info.get(C.POSITION_OPEN_PRICE, 0) # If needed
        sl = position_info.get(C.POSITION_SL, 0)
        tp = position_info.get(C.POSITION_TP, 0)
        
        # Check stop loss and take profit
        if pos_type == SignalType.BUY: # Assuming position_type stores SignalType.BUY/SELL
            if current_price <= sl and sl != 0: # Ensure SL is set
                return SignalType.SELL, "Stop loss hit"
            elif current_price >= tp and tp != 0: # Ensure TP is set
                return SignalType.SELL, "Take profit hit"
                
            # Check for exit conditions (e.g., RSI overbought, MACD cross down)
            if C.INDICATOR_RSI in df.columns and C.INDICATOR_MACD in df.columns and C.INDICATOR_MACD_SIGNAL_LINE in df.columns:
                rsi = df[C.INDICATOR_RSI].iloc[-1]
                macd = df[C.INDICATOR_MACD].iloc[-1]
                macd_prev = df[C.INDICATOR_MACD].iloc[-2] if len(df) > 1 else 0
                macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1]
                macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if len(df) > 1 else 0
                
                if macd_prev > macd_s_line_prev and macd < macd_s_line: # MACD crosses below signal line
                    return SignalType.SELL, "Exit: MACD cross down"
                
                if rsi > 70: # C.RSI_OVERBOUGHT_THRESHOLD
                    return SignalType.SELL, "Exit: RSI overbought"
                    
        elif pos_type == SignalType.SELL:
            if current_price >= sl and sl != 0:
                return SignalType.BUY, "Stop loss hit"
            elif current_price <= tp and tp != 0:
                return SignalType.BUY, "Take profit hit"
                
            if C.INDICATOR_RSI in df.columns and C.INDICATOR_MACD in df.columns and C.INDICATOR_MACD_SIGNAL_LINE in df.columns:
                rsi = df[C.INDICATOR_RSI].iloc[-1]
                macd = df[C.INDICATOR_MACD].iloc[-1]
                macd_prev = df[C.INDICATOR_MACD].iloc[-2] if len(df) > 1 else 0
                macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1]
                macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if len(df) > 1 else 0
                
                if macd_prev < macd_s_line_prev and macd > macd_s_line: # MACD crosses above signal line
                    return SignalType.BUY, "Exit: MACD cross up"
                
                if rsi < 30: # C.RSI_OVERSOLD_THRESHOLD
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
        # Accessing config via self.config (which is ConfigManager instance)
        strategy_section = self.config.get_global_settings().get(C.CONFIG_STRATEGY, {})
        strategy_name = strategy_section.get(C.CONFIG_STRATEGY_TYPE, 'MACD') # Default to 'MACD'

        self.strategy = self._create_strategy(strategy_name, self.config) # Pass ConfigManager
        self.config_manager = self.config # Redundant if self.config is already config_manager, ensure consistency

    def update_strategy_config(self):
        """
        Called when the main configuration has been reloaded.
        The strategy instance can then update its internal state if needed.
        """
        # The strategy itself might have specific logic to update based on new config.
        # For example, if it caches certain parameters from the config at initialization.
        if hasattr(self.strategy, 'update_config') and callable(getattr(self.strategy, 'update_config')):
            # We need to decide what part of config to pass.
            # For now, the strategy's update_config is a general hook.
            # It can pull whatever it needs from self.config_manager.
            self.strategy.update_config(self.config_manager)
            logger.info("Called update_config() on the current strategy.")
        else:
            logger.info("Current strategy does not have a specific update_config() method.")

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
