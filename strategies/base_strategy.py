from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import pandas as pd

"""
# BaseStrategy - Clase Base para Estrategias de Trading

## Descripción General
Esta clase abstracta define la interfaz y funcionalidad común para todas las estrategias de trading 
en el sistema. Proporciona una estructura estandarizada para crear nuevas estrategias, garantizando 
que todas implementen los métodos necesarios y mantengan un comportamiento consistente.

## Características Principales
1. **Interfaz Unificada**: Define un contrato común para todas las estrategias
2. **Abstracción**: Separa la interfaz de la implementación específica
3. **Reutilización**: Proporciona métodos utilitarios comunes para todas las estrategias
4. **Integración**: Se integra con ConfigManager, IndicatorCalculator y otros componentes

## Guía de Implementación para Desarrolladores de IA
Para crear una nueva estrategia de trading basada en esta clase base, siga estos pasos:

### 1. Crear una Nueva Clase de Estrategia
```python
from .base_strategy import BaseStrategy
from core import constants as C

class MiNuevaEstrategia(BaseStrategy):
    NAME: str = "MiNuevaEstrategia"  # Nombre único para la estrategia
    
    def __init__(self, symbol, config_manager, indicator_calculator, strategy_params, indicator_config):
        # Inicializar la clase base primero
        super().__init__(symbol, config_manager, indicator_calculator, strategy_params, indicator_config)
        
        # Inicializar componentes adicionales específicos de esta estrategia
        # Por ejemplo, manejadores especializados o variables de estado
        self.mi_variable_especial = self._get_strategy_param('mi_parametro_especial', valor_por_defecto)
```

### 2. Implementar el Método de Análisis
El método `analyze` es el corazón de cada estrategia y debe implementarse obligatoriamente:

```python
def analyze(self, data_dict: Dict[str, pd.DataFrame], position_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # 1. Preparar variables y obtener datos
    primary_tf = self._get_strategy_param('primary_timeframe', 'M15')  # Obtener timeframe principal
    df = data_dict.get(primary_tf)  # Obtener DataFrame del timeframe principal
    if df is None or df.empty:
        return {'signal': SignalType.NONE, 'message': 'Sin datos suficientes'}
    
    # 2. Obtener valores de indicadores relevantes
    current_price = df[C.INDICATOR_CLOSE_PRICE].iloc[-1]
    
    # 3. Analizar condiciones de mercado
    # Por ejemplo, detectar señales basadas en cruce de medias móviles
    ma_corta = df['ma_corta'].iloc[-1]
    ma_larga = df['ma_larga'].iloc[-1]
    cruce_alcista = ma_corta > ma_larga and df['ma_corta'].iloc[-2] <= df['ma_larga'].iloc[-2]
    
    # 4. Generar señal si se cumplen las condiciones
    if cruce_alcista:
        # 5. Calcular niveles de entrada, stop loss y take profit
        self.entry_price = current_price
        self.stop_loss = current_price - (df[C.INDICATOR_ATR].iloc[-1] * 2)  # Ejemplo: 2 ATR
        self.take_profit = current_price + (df[C.INDICATOR_ATR].iloc[-1] * 3)  # Ejemplo: 3 ATR
        
        return {
            'signal': SignalType.BUY,
            'signal_strength': 0.8,  # Fuerza de la señal entre 0 y 1
            'message': 'Cruce alcista de medias móviles',
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
    
    # 6. Devolver sin señal si no se cumplen condiciones
    return {'signal': SignalType.NONE, 'message': 'Sin señal', 'signal_strength': 0.0}
```

### 3. Métodos Auxiliares para Lógica de Trading
Implemente métodos privados para organizar la lógica de su estrategia:

```python
def _check_entry_signal(self, df, current_price, other_factors):
    # Lógica para detectar señales de entrada
    # Devuelve tipo de señal, fuerza, mensaje, etc.
    pass

def _check_exit_signal(self, df, position_info, current_price):
    # Lógica para detectar señales de salida para posiciones existentes
    pass

def _calculate_risk_levels(self, df, entry_price, signal_type):
    # Calcular stop loss y take profit apropiados
    pass
```

### 4. Integración con el Sistema
Asegúrese de que su estrategia:
- Utilice constantes desde `core.constants` para nombres de indicadores y configuraciones
- Devuelva resultados en el formato esperado por StrategyEngine
- Gestione correctamente los casos extremos (datos insuficientes, errores, etc.)
- Implemente logging adecuado para debugging y monitoreo

### 5. Parámetros de Estrategia Comunes
Los parámetros típicos a considerar para su estrategia:
- Umbrales para indicadores (RSI, ADX, etc.)
- Multiplicadores para cálculo de SL/TP basados en ATR
- Timeframes a utilizar para análisis
- Pesos para diferentes factores en la señal
- Configuración de filtros (tendencia, volatilidad, etc.)

### 6. Pruebas y Optimización
- Crear pruebas unitarias para verificar la lógica
- Usar backtesting para validar el rendimiento
- Optimizar parámetros mediante grid search o algoritmos genéticos
- Validar en diferentes condiciones de mercado y símbolos

## Elementos Clave a Implementar
1. **Señales Claras**: Buy, Sell, None (y sus variantes para salidas)
2. **Gestión de Riesgo**: Cálculo preciso de stops y targets
3. **Lógica Adaptativa**: Comportamiento diferente según condiciones de mercado
4. **Análisis Multi-timeframe**: Considerar diferentes escalas temporales
5. **Cálculo de Proximidad**: Informar qué tan cerca está de generar una señal

## Estructura de Respuesta
La respuesta del método `analyze` debe contener al menos:
- `signal`: Tipo de señal (NONE, BUY, SELL)
- `message`: Descripción de la señal o razón de no señal
- `entry_price`, `stop_loss`, `take_profit`: Niveles de precio para la operación
- `signal_strength`: Valor entre 0 y 1 indicando confianza en la señal
- Información adicional útil para logging y análisis
"""

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
