# Guía de Implementación para IA Desarrolladora: Sistema de Trading Algorítmico

**Fecha:** 28 de Mayo de 2025
**Ubicación:** Cali, Valle del Cauca, Colombia
**Objetivo:** Desarrollar un bot de trading algorítmico modular y configurable en Python 3.10, utilizando MetaTrader 5 para la ejecución y `backtesting.py` para las pruebas.

---

## I. Introducción y Objetivo General

El objetivo es construir un sistema de trading automatizado que implemente una estrategia basada en la confluencia de indicadores (MACD, RSI, ATR) y niveles de Soporte/Resistencia (S/R), con un enfoque multi-temporal. El sistema debe ser robusto, configurable por símbolo, gestionar el riesgo dinámicamente y contar con un sistema de logging completo. La ejecución principal se realizará en M30, con confirmaciones en múltiples temporalidades (M1, M5, M15, H1, H4, D1).

---

## II. Arquitectura General del Sistema

El sistema se diseñará siguiendo un enfoque **modular**, donde cada componente tendrá una responsabilidad específica. Esto facilitará el desarrollo, las pruebas, el mantenimiento y la escalabilidad.

**Estructura de Carpetas Sugerida:**

```
trading_bot/
├── config.json           # Archivo de configuración central
├── main_bot.py           # Orquestador Live Trading
├── backtester.py         # Script para Backtesting
├── requirements.txt      # Dependencias del proyecto
├── core/
│   ├── __init__.py
│   ├── config_manager.py
│   ├── logging_setup.py
│   ├── mt5_connector.py
│   ├── indicator_calculator.py
│   ├── sr_handler.py
│   ├── strategy_engine.py
│   └── risk_manager.py
├── strategies/
│   ├── __init__.py
│   └── confluencia_strategy.py # Clase para backtesting.py
├── data/                   # (Opcional) Para datos históricos CSV
└── logs/                   # Carpeta para archivos de log
```

**Flujo de Datos y Control (Live Trading):**

1.  `main_bot.py` inicia.
2.  `logging_setup.py` configura el logging usando `config.json`.
3.  `config_manager.py` carga y valida `config.json`.
4.  `mt5_connector.py` establece la conexión con MT5.
5.  El bucle principal en `main_bot.py` itera por los símbolos activos:
    * Verifica los límites globales de trades (Long/Short) vía `mt5_connector.py`.
    * Obtiene datos OHLC multi-temporales vía `mt5_connector.py`.
    * Calcula indicadores vía `indicator_calculator.py`.
    * Identifica S/R vía `sr_handler.py`.
    * Busca señales de entrada/salida vía `strategy_engine.py`.
    * Si hay señal y los límites lo permiten, envía órdenes vía `mt5_connector.py`.
    * Gestiona las operaciones abiertas (SL/TP dinámico) vía `risk_manager.py`.
6.  El bucle se repite tras una pausa.

---

## III. Requerimientos Técnicos

* **Lenguaje:** Python 3.10
* **Broker/Plataforma:** MetaTrader 5 (MT5)
* **Bibliotecas Clave:**
    * `MetaTrader5`: Interacción con MT5.
    * `pandas`: Manipulación de datos.
    * `numpy`: Cálculos numéricos.
    * `pandas_ta`: Cálculo de indicadores técnicos (MACD, RSI, ATR, ZigZag, Fractales).
    * `backtesting.py`: Framework de backtesting.
    * `pytz`: Manejo de zonas horarias (crucial para MT5).
    * `logging`: Módulo nativo para logs.

---

## IV. Archivo de Configuración (`config.json`)

Implementar la carga y el uso del archivo `config.json` como se definió previamente. Debe incluir secciones para `metatrader5`, `global_settings` (con `max_total_trades`), `logging`, `defaults` (para indicadores y S/R), y `symbols` (con configuraciones específicas y *overrides* por símbolo).

**Tarea:** Crear `core/config_manager.py` con una clase o función que cargue este JSON, lo valide mínimamente y provea una forma sencilla de acceder a la configuración de un símbolo específico (combinando `defaults` con `symbols[symbol_name]`).

---

## V. Implementación por Módulo

### 1. `core/logging_setup.py`

* **Propósito:** Configurar el sistema de logging.
* **Tareas:**
    * Crear la función `setup_logging(config_path)`.
    * Leer la sección `logging` del `config.json`.
    * Asegurar la creación del directorio `logs/`.
    * Usar `logging.config.dictConfig()` para aplicar la configuración.
    * Implementar los handlers (`StreamHandler`, `RotatingFileHandler`) y formatters como se definió.
    * Debe ser la *primera* cosa que se ejecute en `main_bot.py`.

### 2. `core/mt5_connector.py`

* **Propósito:** Aislar toda la comunicación con MT5.
* **Tareas:**
    * `initialize(config)`: Conectar a MT5 usando credenciales y path del `config.json`. Manejar errores de conexión.
    * `disconnect()`: Cerrar la conexión.
    * `get_data(symbol, timeframe_enum, count)`: Obtener datos OHLC. **¡Crucial: Manejar zonas horarias!** MT5 suele devolver UTC. Convertir a un formato consistente si es necesario. Usar `mt5.copy_rates_from_pos`. Convertir a DataFrame de Pandas.
    * `get_open_positions()`: Devolver una lista (o DataFrame) de posiciones abiertas. Debe incluir símbolo, ticket, tipo (BUY/SELL), volumen, precio de entrada, SL y TP. **Importante:** Calcular y devolver el conteo actual de `long_trades` y `short_trades`.
    * `send_order(symbol, order_type, volume, sl_price, tp_price, comment)`: Enviar órdenes de mercado. Manejar `trade_request` y `trade_result`. Registrar *detalladamente* el resultado (éxito o error).
    * `modify_position(ticket, sl_price, tp_price)`: Modificar SL/TP de una orden existente. Manejar errores.
    * `close_position(ticket, volume, comment)`: Cerrar una posición (total o parcialmente).

### 3. `core/indicator_calculator.py`

* **Propósito:** Calcular indicadores técnicos.
* **Tareas:**
    * Recibir DataFrames de Pandas y parámetros.
    * Usar `pandas_ta` para calcular:
        * `calculate_macd(df, fast, slow, signal)`
        * `calculate_rsi(df, period)`
        * `calculate_atr(df, period)`
    * Añadir los resultados como nuevas columnas al DataFrame.
    * Manejar `NaN` (valores no numéricos) iniciales.

### 4. `core/sr_handler.py`

* **Propósito:** Identificar niveles de Soporte y Resistencia.
* **Tareas:**
    * `get_sr_levels(df, config)`: Función principal que actúa como dispatcher.
    * Leer `config['sr_method']`.
    * Llamar a la función interna correspondiente:
        * `calculate_pivots(df_pivot_base, type)`: Calcular Puntos Pivote (Standard, Fibonacci, etc.). Requiere data de un período mayor (ej: D1 para pivots diarios). Devolver un diccionario `{'R3': ..., 'R2': ..., 'R1': ..., 'P': ..., 'S1': ..., 'S2': ..., 'S3': ...}`.
        * `calculate_fractals(df, window)`: Usar `pandas_ta.fractals` o implementarlo. Devolver listas de precios de soportes y resistencias fractales recientes.
        * `calculate_zigzag(df, depth, deviation, backstep)`: Usar `pandas_ta.zigzag`. Devolver los puntos de giro (soportes y resistencias) recientes.
    * Devolver los niveles en un formato *consistente*, independientemente del método.

### 5. `core/strategy_engine.py`

* **Propósito:** Implementar la lógica de trading y generar señales.
* **Tareas:**
    * `check_signal(symbol, data_dict, sr_levels, config)`: Función principal. `data_dict` contendrá los DataFrames de todas las temporalidades (`{'M1': df1, 'M30': df30, ...}`).
    * Implementar la lógica "Top-Down":
        1.  `_check_trend(data_h1, data_h4, data_d1)`: Analizar MACD/RSI en TFs altos para determinar la *dirección permitida* (Solo Compras, Solo Ventas, Ambas, Ninguna).
        2.  `_check_m30_signal(data_m30, sr_m30, direction)`: Buscar cruces de MACD y niveles de RSI en M30, verificando la proximidad a S/R y alineación con `direction`.
        3.  `_check_m15_confirmation(data_m15, signal_type)`: Confirmar la señal M30 con MACD/RSI en M15/M5.
    * Devolver `"BUY"`, `"SELL"` o `"HOLD"`.

### 6. `core/risk_manager.py`

* **Propósito:** Gestionar SL, TP y el tamaño de la operación.
* **Tareas:**
    * `calculate_initial_sl_tp(entry_price, atr_value, sr_levels, config, order_type)`: Calcular SL/TP inicial usando multiplicadores de ATR y/o niveles S/R.
    * `manage_open_trades(open_positions, data_dict)`: Función principal para el bucle.
        * Iterar sobre cada posición abierta.
        * Obtener el símbolo y los datos actuales.
        * `_move_to_be(position, current_price, atr_value, config)`: Mover SL a BE si se cumple la condición.
        * `_apply_trailing_sl(position, current_price, atr_value, config)`: Aplicar TSL según el método configurado (ATR, S/R, MA).
        * Si se requiere modificación, llamar a `mt5_connector.modify_position()`.

### 7. `main_bot.py`

* **Propósito:** Orquestar el bot en modo Live/Demo.
* **Tareas:**
    * Llamar a `setup_logging()`.
    * Cargar configuración.
    * Inicializar MT5.
    * Implementar el bucle infinito:
        * Manejar interrupciones (`KeyboardInterrupt`).
        * Implementar `try...except` robusto con logging.
        * Obtener posiciones abiertas (`get_open_positions()`).
        * Calcular `current_long`, `current_short`.
        * Calcular `max_long = max_short = max_total_trades / 2`.
        * Iterar sobre símbolos activos:
            * Si `current_long < max_long`, buscar señales de COMPRA.
            * Si `current_short < max_short`, buscar señales de VENTA.
            * Si hay señal, calcular SL/TP y enviar orden.
        * Llamar a `risk_manager.manage_open_trades()`.
        * Esperar (`time.sleep()`).

### 8. `backtester.py`

* **Propósito:** Ejecutar backtests individuales por símbolo.
* **Tareas:**
    * Cargar configuración.
    * Iterar sobre símbolos activos.
    * Para cada símbolo:
        * Cargar/Descargar datos históricos (M30 como base, y H1/H4/D1).
        * **Pre-procesar Datos:** ¡Paso crítico! Calcular indicadores y S/R para H1/H4/D1 y *fusionarlos* (usando `pd.merge_asof` o similar) con el DataFrame M30. Esto es necesario para simular el Multi-TF en `backtesting.py`.
        * Definir la clase `Strategy` (en `strategies/confluencia_strategy.py`) que implemente la lógica M30 usando los datos pre-procesados y los parámetros del config.
        * Hacer que la clase `Strategy` sea *configurable* (acepte parámetros en `init` o use variables de clase).
        * Instanciar `Backtest` con los datos M30 (ya enriquecidos) y la estrategia.
        * Ejecutar `bt.run()`.
        * Imprimir/Guardar resultados y gráficos (`bt.plot()`).

---

## VI. Lógica de la Estrategia (Detalle)

Implementar las reglas de entrada y salida como se describió en las discusiones previas, asegurándose de que el `strategy_engine.py` las verifique secuencialmente (Tendencia -> Señal M30 -> Confirmación M15/M5). Asegurarse de que las condiciones de RSI (no <30 para compra, no >70 para venta) y la proximidad a S/R se verifiquen.

---

## VII. Gestión de Riesgo (Detalle)

Implementar la gestión de riesgo dinámica en `risk_manager.py`. El TSL debe ser configurable (ej: mantener X * ATR detrás del máximo/mínimo reciente). El movimiento a BE debe ser claro (cuando el precio avanza Y * ATR, mover SL a entrada + spread/comisión).

---

## VIII. Backtesting (Detalle)

El principal desafío es la simulación Multi-TF. El enfoque de **pre-procesamiento** es el más viable para `backtesting.py`. Asegúrate de que los datos de TFs superiores estén correctamente alineados con las velas M30 (la vela H1 de las 10:00 aplica a las velas M30 de 10:00 y 10:30). La clase `Strategy` debe ser lo suficientemente inteligente para usar estos datos pre-calculados. El TSL también necesitará una implementación manual dentro del método `next()` de la estrategia.

---

## IX. Consideraciones Adicionales

* **Manejo de Errores:** Implementar `try...except` extensivamente, especialmente alrededor de las interacciones con MT5 y cálculos. Usar `logger.error()` y `logger.critical()`.
* **Timezones:** Ser *muy* cuidadoso con las zonas horarias. MT5, Python y el servidor del broker pueden tener configuraciones diferentes. Usar `pytz` y preferiblemente trabajar internamente en UTC.
* **Dependencias:** Crear un archivo `requirements.txt` con todas las bibliotecas necesarias.
* **Seguridad:** No guardar contraseñas directamente en `config.json` si el código va a un repositorio. Considerar usar variables de entorno o un sistema de gestión de secretos para producción.
* **Pruebas:** Además del backtesting, considerar pruebas unitarias para módulos individuales (especialmente cálculos y lógica pura).
* **Optimización:** La optimización (`bt.optimize()`) debe hacerse con cuidado para evitar el *overfitting*.

---

## X. Entregables Esperados

1.  La estructura completa de carpetas y archivos Python implementados.
2.  El archivo `config.json` de ejemplo.
3.  El archivo `requirements.txt`.
4.  Un archivo `README.md` que explique cómo instalar, configurar y ejecutar tanto el bot en vivo como el backtester.

---

Esta guía proporciona un marco detallado para el desarrollo. La IA desarrolladora debe proceder módulo por módulo, asegurando que cada componente funcione y se integre correctamente con los demás, prestando especial atención al logging y al manejo de errores en cada paso.