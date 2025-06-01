from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import pandas as pd

# Use TYPE_CHECKING to avoid circular import issues at runtime
if TYPE_CHECKING:
    from core.config_manager import ConfigManager
    from core.indicator_calculator import IndicatorCalculator
    # from core.sr_handler import SRHandler # If SRHandler is to be passed or part of base

from core import constants as C # For SignalType or other constants if needed

class BaseStrategy(ABC):
    NAME: str = "BaseStrategy" # Default name, can be overridden by subclasses

    def __init__(self,
                 symbol: str, # The specific symbol this instance is for
                 config_manager: 'ConfigManager',
                 indicator_calculator: 'IndicatorCalculator',
                 # sr_handler: 'SRHandler', # Optional: if SRHandler is also managed by StrategyEngine
                 strategy_params: Dict[str, Any], # Strategy-specific logic parameters
                 indicator_config: Dict[str, Any]): # Configuration for indicators (periods, etc.)

        self.symbol = symbol
        self.config_manager = config_manager # Provides access to broader config if needed
        self.indicator_calc = indicator_calculator # Pre-configured and passed in
        # self.sr_handler = sr_handler # If passed
        # Alternatively, SRHandler can be initialized here if it's always needed:
        # from core.sr_handler import SRHandler # Local import to avoid circularity at module level
        # self.sr_handler = SRHandler(config_manager)


        self.strategy_params = strategy_params
        self.indicator_config = indicator_config

        # Common attributes for strategies to set in their analyze method
        self.stop_loss: float = 0.0
        self.take_profit: float = 0.0
        self.entry_price: float = 0.0 # Suggested by MACDStrategy's usage

    @abstractmethod
    def analyze(self, data_dict: Dict[str, pd.DataFrame], position_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyzes market data and position info to generate trading signals.

        Args:
            data_dict: Dictionary of pandas DataFrames, where keys are timeframe strings
                       (e.g., "M15", "H1") and values are the OHLCV data.
                       The strategy should expect data for its required timeframes here.
            position_info (Optional[Dict[str, Any]]): Information about an existing position
                                                     for the symbol, if any.
                                                     Example: {'type': SignalType.BUY, 'price_open': 1.12345, ...}

        Returns:
            Dict[str, Any]: A dictionary adhering to StrategyResult structure, e.g.:
                {
                    'signal': SignalType.NONE, # from core.strategy_engine.SignalType
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'message': "Reason for signal or no signal",
                    'entry_price': 0.0, # Suggested entry if applicable
                    'indicators': {} # Optional dict of key indicator values for logging/output
                }
        """
        pass

    def _get_strategy_param(self, param_name: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Helper to get a parameter value from self.strategy_params.
        These params are for the strategy's logic (e.g., thresholds, multipliers for SL/TP).
        """
        return self.strategy_params.get(param_name, default)

    def _get_indicator_param(self, param_name: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Helper to get an indicator parameter value from self.indicator_config.
        These params are for indicator calculations (e.g., periods).
        """
        return self.indicator_config.get(param_name, default)

```
