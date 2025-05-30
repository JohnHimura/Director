from backtesting import Strategy
import pandas as pd
import numpy as np
import sys
import os
# Ensure the path is correct for your project structure
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.strategy_engine import StrategyEngine, SignalType
from core.config_manager import ConfigManager
from core import constants as C # Import constants

class ConfluenciaBacktestStrategy(Strategy):
    '''Estrategia de backtesting basada en la lógica de confluencia del sistema (MACD, RSI, ATR, S/R, multi-timeframe).'''
    # It's better to pass config_path and symbol as parameters to init
    # config_path = 'config.json' # Hardcoded path
    # symbol = 'SYMBOL' # Hardcoded symbol
    
    def init(self, config_path='config.json', symbol='EURUSD'): # Provide defaults or make them required
        self.config_path = config_path
        self.symbol_name = symbol # Use a different name to avoid conflict with self.symbol from Backtesting.py

        self.config = ConfigManager(self.config_path)
        self.engine = StrategyEngine(self.config)

        timeframes = self.config.get_timeframes()
        # Using C.PRIMARY_TIMEFRAME_KEY if it's defined for how 'primary' is stored, else literal 'primary'
        self.primary_tf_key = C.PRIMARY_TIMEFRAME_KEY if hasattr(C, 'PRIMARY_TIMEFRAME_KEY') else 'primary'
        self.primary_tf = timeframes.get(self.primary_tf_key, 'M15') # Default to M15 or some constant

        # Assuming H4 and D1 are actual timeframe strings like "H4", "D1"
        self.h4_tf = "H4"
        self.d1_tf = "D1"

        self.data_primary = self.data # Assuming self.data is the primary timeframe data

        self.current_position_type = SignalType.NONE # Use SignalType enum
        self.current_entry_price = None
        self.current_stop_loss = None
        self.current_take_profit = None

    def next(self):
        # Build data_dict for StrategyEngine
        data_dict = {self.primary_tf: self.data_primary.df} # .df gives the pandas DataFrame

        # For backtesting with merged data (e.g., H4 data in M15 columns like 'h4_close')
        # This part needs careful handling based on how merged data is prepared.
        # For simplicity, let's assume StrategyEngine is designed for dicts of separate DataFrames.
        # If using merged columns, the Strategy inside StrategyEngine would need to know how to parse them.
        # The current StrategyEngine expects distinct DataFrames per timeframe.
        # So, for backtesting, one might need to pre-process data or adapt the strategy.
        # Here, we're just passing the primary timeframe data.
        # To use H4/D1 trend, the StrategyEngine's get_trend_direction would need these DataFrames.
        # If they are not available in self.data (e.g. if self.data is only M15), then trend will be neutral.

        position_info = None
        if self.position: # self.position is a Backtesting.py Position object
            position_info = {
                C.POSITION_TYPE: SignalType.BUY if self.position.is_long else SignalType.SELL,
                C.POSITION_OPEN_PRICE: self.position.entry_price,
                C.POSITION_SL: self.current_stop_loss, # SL/TP from our tracking
                C.POSITION_TP: self.current_take_profit,
                # 'current_price': self.data.Close[-1] # Strategy might need this
            }

        result = self.engine.analyze(
            symbol=self.symbol_name, # Use the configured symbol name
            data=data_dict,
            position_info=position_info
        )

        # result is a dict from StrategyResult.to_dict()
        signal = result.get('signal', SignalType.NONE)
        # stop_loss_price = result.get(C.POSITION_SL, 0.0) # Keys from StrategyResult.to_dict()
        # take_profit_price = result.get(C.POSITION_TP, 0.0)
        # entry_price_suggestion = result.get(C.POSITION_OPEN_PRICE, 0.0)

        # Accessing SL/TP from the strategy instance inside engine, as it's updated there
        # This assumes MACDStrategy (or any strategy) updates self.stop_loss and self.take_profit attributes.
        stop_loss_price = self.engine.strategy.stop_loss
        take_profit_price = self.engine.strategy.take_profit


        if not self.position: # If no open position
            if signal == SignalType.BUY:
                self.buy(sl=stop_loss_price, tp=take_profit_price)
                self.current_position_type = SignalType.BUY
                self.current_entry_price = self.data.Close[-1] # Or result.get('entry_price') if strategy suggests one
                self.current_stop_loss = stop_loss_price
                self.current_take_profit = take_profit_price
            elif signal == SignalType.SELL:
                self.sell(sl=stop_loss_price, tp=take_profit_price)
                self.current_position_type = SignalType.SELL
                self.current_entry_price = self.data.Close[-1]
                self.current_stop_loss = stop_loss_price
                self.current_take_profit = take_profit_price
        else: # If a position is open
            if (self.position.is_long and signal == SignalType.SELL) or \
               (self.position.is_short and signal == SignalType.BUY):
                self.position.close()
                self.current_position_type = SignalType.NONE
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
            # SL/TP management is handled by backtesting.py if sl, tp are passed to buy/sell
            # No need for manual SL/TP checking here if you pass them to self.buy()/self.sell()
            # However, if strategy dynamically changes SL/TP for an OPEN position (trailing),
            # then self.position.sl = new_sl or self.position.tp = new_tp might be needed.
            # The current StrategyEngine doesn't seem to support dynamic SL/TP updates mid-trade in this way.
            # The _check_exit_signal in MACDStrategy is more for deciding if a general "close" signal occurs.