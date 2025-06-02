"""
Strategy engine for the trading bot.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Protocol, Type
import pandas as pd
import numpy as np
import abc

import importlib # Moved import to top
from pathlib import Path # Moved import to top
from .indicator_calculator import IndicatorCalculator
# SRHandler might be initialized within specific strategies if not passed by engine
# from .sr_handler import SRHandler, SRLevel
from . import constants as C
from .config_manager import ConfigManager, ValidationError # Import ValidationError for get_strategy_instance
from strategies.base_strategy import BaseStrategy # Import new BaseStrategy

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
        self.levels = levels or {C.SR_SUPPORT: [], C.SR_RESISTANCE: []}
        self.trend_direction = trend_direction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'message': self.message,
            C.CONFIG_INDICATORS: self.indicators,
            'levels': self.levels,
            'trend_direction': self.trend_direction
        }

# The old Strategy ABC is removed. BaseStrategy is now in strategies.base_strategy

class StrategyEngine:
    """Implements the trading strategy logic by dynamically loading and managing strategies."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.strategies_available: Dict[str, Type[BaseStrategy]] = {} # Use BaseStrategy
        self._load_strategies_dynamically()
        # self.last_signal = SignalType.NONE # This seems more like a bot-level attribute, remove from engine

    def update_strategy_config(self): # TODO: Review if this is needed with dynamic instantiation
        # This method is tricky. If strategies are instantiated per call to analyze,
        # they will always get the latest config. If they are cached, they need update.
        # For now, dynamic loading implies strategies might be re-fetched or are short-lived per symbol.
        # If we cache strategy instances, they would need this.
        logger.info("StrategyEngine.update_strategy_config called. If strategies are cached, they should be updated/reloaded.")
        # Example if strategies were cached:
        # for strategy_instance in self.cached_strategy_instances.values():
        #     strategy_instance.update_config(self.config_manager)


    def _load_strategies_dynamically(self):
        # import importlib # Already at top
        import inspect # Moved here from global to be specific to this method
        strategies_package_name = "strategies"
        # Correct path relative to this file (core/strategy_engine.py) -> ../strategies/
        base_path = Path(__file__).resolve().parent.parent
        strategies_dir = base_path / strategies_package_name

        logger.info(f"Attempting to load strategies from: {strategies_dir}")

        if not strategies_dir.is_dir():
            logger.error(f"Strategies directory '{strategies_dir}' not found or is not a directory.")
            return

        for filepath in strategies_dir.glob("*.py"):
            filename = filepath.name
            if filename.startswith("__init__") or filename == "base_strategy.py": # Skip base_strategy
                continue
            
            module_name_fs = filename[:-3]
            # Assuming 'strategies' is directly importable (e.g., in PYTHONPATH or same level as 'core')
            full_module_path = f"{strategies_package_name}.{module_name_fs}"
            
            try:
                module = importlib.import_module(full_module_path)
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a subclass of BaseStrategy and not BaseStrategy itself
                    if issubclass(cls, BaseStrategy) and cls is not BaseStrategy:
                        strategy_name = getattr(cls, 'NAME', cls.__name__) # Use NAME attribute or class name
                        if strategy_name in self.strategies_available:
                            logger.warning(f"Duplicate strategy name '{strategy_name}' found in {full_module_path}. It will be overwritten by the version from this module.")
                        self.strategies_available[strategy_name] = cls
                        logger.info(f"Dynamically loaded strategy: '{strategy_name}' from {full_module_path}")
            except ImportError as e:
                logger.error(f"Failed to import strategy module {full_module_path}: {e}", exc_info=True)
            except Exception as e: # Catch other potential errors during module processing
                logger.error(f"Error processing or loading from strategy file {filepath}: {e}", exc_info=True)

    def get_strategy_instance(self, symbol: str) -> Optional[BaseStrategy]: # Return type BaseStrategy
        try:
            # This should provide a fully resolved config for the symbol, including merged defaults
            symbol_config = self.config_manager.get_symbol_config(symbol)
        except ValidationError:
            logger.error(f"Configuration for symbol '{symbol}' not found.", extra={'symbol': symbol})
            return None

        # Ensure CONFIG_STRATEGY_NAME and DEFAULT_STRATEGY_NAME are defined in constants.py
        strategy_name_from_config = symbol_config.get(
            getattr(C, 'CONFIG_STRATEGY_NAME', 'strategy_name'), # Fallback key if constant missing
            getattr(C, 'DEFAULT_STRATEGY_NAME', 'MACDStrategy') # Fallback default if constant missing
        )
        StrategyClass = self.strategies_available.get(strategy_name_from_config)
        
        if StrategyClass:
            try:
                strategy_params = symbol_config.get(C.CONFIG_STRATEGY_PARAMS, {})
                indicator_config = symbol_config.get(C.CONFIG_INDICATORS, {})
                
                # Instantiate IndicatorCalculator:
                # Assuming IndicatorCalculator's __init__(self, config: Dict[str, Any])
                # where 'config' is the specific indicator_config for the symbol.
                indicator_calculator = IndicatorCalculator(indicator_config)
                # If SRHandler is needed per strategy and not part of BaseStrategy init:
                # sr_config = symbol_config.get(C.CONFIG_SR, {})
                # sr_handler = SRHandler(sr_config)

                instance = StrategyClass(
                    symbol=symbol,
                    config_manager=self.config_manager, # Pass full config manager
                    indicator_calculator=indicator_calculator,
                    strategy_params=strategy_params,
                    indicator_config=indicator_config
                    # sr_handler=sr_handler # If passing SRHandler
                )
                logger.info(f"Created instance of '{strategy_name_from_config}' for symbol {symbol}.",
                            extra={'symbol': symbol, 'strategy_name': strategy_name_from_config})
                return instance
            except Exception as e:
                logger.error(f"Error instantiating strategy '{strategy_name_from_config}' for {symbol}: {e}",
                             extra={'symbol': symbol, 'strategy_name': strategy_name_from_config}, exc_info=True)
                return None
        else:
            logger.error(f"Strategy class for '{strategy_name_from_config}' not found for {symbol}. Available: {list(self.strategies_available.keys())}",
                         extra={'symbol': symbol, 'strategy_name': strategy_name_from_config})
            return None

    def analyze(
        self, 
        symbol: str, 
        data: Dict[str, pd.DataFrame],
        position_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        strategy_instance = self.get_strategy_instance(symbol) # Gets Optional[BaseStrategy]
        if not strategy_instance:
            logger.error(f"Could not get strategy instance for {symbol}. Analysis aborted.", extra={'symbol': symbol})
            # Return a dictionary matching the expected output of a strategy's analyze method
            return {
                'signal': SignalType.NONE, 'stop_loss': 0.0, 'take_profit': 0.0,
                'message': f"No strategy instance for {symbol}", 'indicators': {}, 'levels': {},
                'trend_direction': 0, 'signal_strength': 0.0, 'entry_price': 0.0
            }

        # Asegurarse de que los indicadores estén calculados en los DataFrames
        try:
            symbol_config = self.config_manager.get_symbol_config(symbol)
            indicator_config = symbol_config.get(C.CONFIG_INDICATORS, {})
            indicator_calculator = IndicatorCalculator(indicator_config)
            
            # Calcular indicadores para cada timeframe en data
            for timeframe, df in data.items():
                if not df.empty:
                    # Calcular todos los indicadores necesarios
                    indicator_calculator.calculate_all(df, indicator_config)
                    
                    # Asegurarse de que los indicadores esenciales estén presentes
                    if C.INDICATOR_RSI not in df.columns:
                        indicator_calculator.calculate_rsi(df, indicator_config)
                    if C.INDICATOR_MACD not in df.columns or C.INDICATOR_MACD_SIGNAL_LINE not in df.columns:
                        indicator_calculator.calculate_macd(df, indicator_config)
                    if C.INDICATOR_ATR not in df.columns:
                        indicator_calculator.calculate_atr(df, indicator_config)
                    if C.INDICATOR_ADX not in df.columns:
                        indicator_calculator.calculate_adx(df, indicator_config)
                    
                    # Reemplazar el DataFrame original con el que tiene los indicadores
                    data[timeframe] = df
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}", extra={'symbol': symbol}, exc_info=True)

        # The `analyze` method of BaseStrategy (and thus MACDStrategy) takes (self, data_dict, position_info)
        # and returns a dictionary.
        result_dict = strategy_instance.analyze(data, position_info)

        return result_dict
