from backtesting import Strategy
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../core')
from core.strategy_engine import StrategyEngine, SignalType
from core.config_manager import ConfigManager

class ConfluenciaBacktestStrategy(Strategy):
    '''Estrategia de backtesting basada en la lógica de confluencia del sistema (MACD, RSI, ATR, S/R, multi-timeframe).'''
    config_path = 'config.json'
    symbol = 'SYMBOL'
    
    def init(self):
        # Cargar configuración global
        self.config = ConfigManager(self.config_path)
        self.engine = StrategyEngine(self.config)
        # Guardar timeframe principal y otros
        self.primary_tf = self.config.get('timeframes', {}).get('primary', 'M30')
        self.h4_tf = self.config.get('timeframes', {}).get('h4', 'H4')
        self.d1_tf = self.config.get('timeframes', {}).get('d1', 'D1')
        # Asumimos que el DataFrame principal es M30 y que las columnas de H4 y D1 están fusionadas
        # Si no, solo se usará la lógica de un solo TF
        self.data_m30 = self.data
        # Estado de posición
        self.position_type = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        # Construir el diccionario de dataframes multi-TF (aquí solo M30, pero se puede enriquecer)
        data_dict = {self.primary_tf: self.data_m30}
        # Simular data de H4 y D1 si existen columnas fusionadas
        if any(col.startswith('h4_') for col in self.data_m30.columns):
            data_dict[self.h4_tf] = self.data_m30[[col for col in self.data_m30.columns if col.startswith('h4_')]].copy()
        if any(col.startswith('d1_') for col in self.data_m30.columns):
            data_dict[self.d1_tf] = self.data_m30[[col for col in self.data_m30.columns if col.startswith('d1_')]].copy()
        # Información de la posición actual
        position_info = None
        if self.position:
            position_info = {
                'position_type': 'BUY' if self.position.is_long else 'SELL',
                'entry_price': self.position.entry_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
        # Llamar a la lógica de la estrategia real
        result = self.engine.analyze(
            symbol=self.symbol,
            data=data_dict,
            position_info=position_info
        )
        signal = result.get('signal', SignalType.NONE)
        # Ejecutar señales
        if not self.position:
            if signal == SignalType.BUY:
                self.buy()
                self.position_type = 'BUY'
                self.entry_price = self.data.Close[-1]
                self.stop_loss = self.engine.stop_loss
                self.take_profit = self.engine.take_profit
            elif signal == SignalType.SELL:
                self.sell()
                self.position_type = 'SELL'
                self.entry_price = self.data.Close[-1]
                self.stop_loss = self.engine.stop_loss
                self.take_profit = self.engine.take_profit
        else:
            # Salida por señal contraria o por stop/take
            if (self.position.is_long and signal == SignalType.SELL) or (self.position.is_short and signal == SignalType.BUY):
                self.position.close()
                self.position_type = None
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
            # Salida por stop loss/take profit
            elif self.position.is_long:
                if self.stop_loss and self.data.Close[-1] <= self.stop_loss:
                    self.position.close()
                elif self.take_profit and self.data.Close[-1] >= self.take_profit:
                    self.position.close()
            elif self.position.is_short:
                if self.stop_loss and self.data.Close[-1] >= self.stop_loss:
                    self.position.close()
                elif self.take_profit and self.data.Close[-1] <= self.take_profit:
                    self.position.close() 