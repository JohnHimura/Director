import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

"""
# Estrategia MACD (Moving Average Convergence Divergence)

## Descripción General
Esta estrategia utiliza el indicador MACD junto con RSI, soportes/resistencias y análisis de tendencia 
en múltiples timeframes para generar señales de trading. Combina varios factores para determinar la 
fuerza de la señal antes de ejecutar operaciones.

## Componentes Principales
1. **MACD**: Identificación de cruces entre línea MACD y línea de señal
2. **RSI**: Confirmación de condiciones de sobrecompra/sobreventa
3. **Soportes/Resistencias**: Proximidad a niveles clave
4. **Análisis Multi-timeframe**: Evaluación de tendencia en timeframes superiores (H4, D1)
5. **Filtro ADX** (opcional): Confirmación de fuerza de tendencia

## Lógica de Entrada
### Señal de Compra (BUY)
- MACD cruza por encima de su línea de señal
- RSI < nivel de sobreventa O (RSI entre sobreventa/sobrecompra Y tendencia alcista/neutral)
- Precio cerca de un nivel de soporte
- Tendencia en timeframes superiores es alcista o neutral
- Si habilitado, ADX > valor umbral mínimo

### Señal de Venta (SELL)
- MACD cruza por debajo de su línea de señal
- RSI > nivel de sobrecompra O (RSI entre sobreventa/sobrecompra Y tendencia bajista/neutral)
- Precio cerca de un nivel de resistencia
- Tendencia en timeframes superiores es bajista o neutral
- Si habilitado, ADX > valor umbral mínimo

## Cálculo de Stops y Targets
### Stop Loss (SL)
1. Basado en ATR si la configuración 'use_atr_sl_tp' está activada
2. Basado en niveles S/R cercanos si están disponibles
3. Fallback a un múltiplo de ATR si los métodos anteriores fallan

### Take Profit (TP)
1. Basado en ATR si la configuración 'use_atr_sl_tp' está activada
2. Calculado como un ratio riesgo:recompensa sobre el stop loss

## Lógica de Salida
- MACD cruza en dirección opuesta a la posición
- RSI alcanza nivel de sobrecompra (para posiciones largas) o sobreventa (para posiciones cortas)
- Implementa trailing stops, breakeven stops y salidas basadas en tiempo si están configuradas

## Flujo de Análisis
1. Obtener datos de múltiples timeframes
2. Analizar tendencia en timeframes superiores
3. Identificar niveles de soporte/resistencia
4. Evaluar condiciones de entrada y calcular fuerza de la señal
5. Si la fuerza de la señal supera el umbral mínimo, generar señal de compra/venta
6. Calcular entrada, stop loss y take profit
7. Para posiciones abiertas, verificar condiciones de salida

## Métricas de Proximidad
La estrategia ahora calcula qué tan cerca está de generar una señal:
- buy_threshold_proximity_pct: Porcentaje del umbral mínimo alcanzado para señal de compra
- sell_threshold_proximity_pct: Porcentaje del umbral mínimo alcanzado para señal de venta
"""

from .base_strategy import BaseStrategy # Import from local package
from core.indicator_calculator import IndicatorCalculator
from core.sr_handler import SRHandler, SRLevel # Assuming SRLevel is used or defined
from core import constants as C
from core.strategy_engine import SignalType # This might need to be moved to constants or base_strategy
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    NAME: str = "MACDStrategy"

    def __init__(self,
                 symbol: str,
                 config_manager: ConfigManager,
                 indicator_calculator: IndicatorCalculator,
                 strategy_params: Dict[str, Any],
                 indicator_config: Dict[str, Any]):

        super().__init__(symbol, config_manager, indicator_calculator, strategy_params, indicator_config)
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME, 'method': '__init__'}
        logger.debug(f"Initializing MACDStrategy for {self.symbol}.", extra=log_extras)
        logger.debug(f"Effective strategy_params for {self.symbol}:", extra=log_extras)
        for key, value in self.strategy_params.items():
            logger.debug(f"  S_PARAM: {key} = {value}", extra=log_extras)
        logger.debug(f"Effective indicator_config for {self.symbol}:", extra=log_extras)
        for key, value in self.indicator_config.items():
            logger.debug(f"  I_PARAM: {key} = {value}", extra=log_extras)

        # Initialize SRHandler using the passed ConfigManager and symbol-specific SR config
        # SR parameters should be part of the broader symbol configuration, fetched by ConfigManager
        sr_config_for_symbol = self.config_manager.get_sr_params(self.symbol)
        # Note: SRHandler's __init__ might need adjustment if it expects just a dict vs. full ConfigManager
        # Current SRHandler __init__(self, config: Dict[str, Any]) -> self.config = config
        # So, passing sr_config_for_symbol directly is appropriate.
        self.sr_handler = SRHandler(sr_config_for_symbol)


    def analyze(self, data_dict: Dict[str, pd.DataFrame], position_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME, 'method': 'analyze'}
        # Determine primary timeframe
        primary_tf = self._get_strategy_param('primary_timeframe', list(data_dict.keys())[0] if data_dict else C.TF_M15)
        logger.debug(f"Starting analysis for {self.symbol} on primary_tf: {primary_tf}. Strategy Params Applied: {self.strategy_params}", extra=log_extras)

        if not data_dict or primary_tf not in data_dict or data_dict[primary_tf].empty:
            logger.warning(f"Primary timeframe {primary_tf} not found or empty in data_dict.", extra=log_extras)
            return {'signal': SignalType.NONE, 'message': f"No data for primary timeframe {primary_tf}"}

        df = data_dict[primary_tf] # Strategies should use the data as passed; calculation is done by engine/wrapper

        # Log last N candles from primary timeframe dataframe
        if not df.empty:
            num_candles_to_log = min(len(df), 3) # Log last 3 candles, or fewer if not available
            logger.debug(f"Last {num_candles_to_log} candles for {self.symbol} on {primary_tf} (Price & Key Indicators):", extra=log_extras)
            # Select only a few key columns for brevity in this specific log. Full data is in df.
            cols_to_log = [C.INDICATOR_OPEN_PRICE, C.INDICATOR_HIGH_PRICE, C.INDICATOR_LOW_PRICE, C.INDICATOR_CLOSE_PRICE, C.INDICATOR_VOLUME, 
                           C.INDICATOR_MACD, C.INDICATOR_MACD_SIGNAL_LINE, C.INDICATOR_RSI, C.INDICATOR_ATR]
            existing_cols_to_log = [col for col in cols_to_log if col in df.columns]
            if existing_cols_to_log:
                for i in range(num_candles_to_log, 0, -1):
                    candle_data_log = df[existing_cols_to_log].iloc[-i].to_dict()
                    # Format floats in the dictionary for consistent logging
                    formatted_candle_data = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in candle_data_log.items()}
                    logger.debug(f"  Candle[-{i}]: {formatted_candle_data}", extra=log_extras)
            else:
                logger.debug(f"  Could not log candle details as key indicator columns are missing from df.", extra=log_extras)
        else:
            logger.warning(f"DataFrame for {primary_tf} is empty, cannot log candle data.", extra=log_extras)

        # Ensure required columns are present (indicators should be pre-calculated on df)
        required_cols_for_analyze = [C.INDICATOR_CLOSE_PRICE] # Minimum
        # Add other indicators MACDStrategy directly uses from df to this list
        if not all(col in df.columns for col in required_cols_for_analyze):
            logger.warning(f"DataFrame missing required columns for analysis. Needs: {required_cols_for_analyze}", extra=log_extras)
            return {'signal': SignalType.NONE, 'message': "DataFrame missing essential columns"}


        sr_levels = self.sr_handler.get_sr_levels(df)
        current_price = df[C.INDICATOR_CLOSE_PRICE].iloc[-1]
        logger.debug(f"Current price (close of last bar on {primary_tf}): {current_price:.5f}", extra=log_extras)
        logger.debug(f"Identified S/R Levels: {[(lvl.type, lvl.price, lvl.strength) for lvl in sr_levels]}", extra=log_extras)

        trend_direction = self._get_higher_timeframe_trend(data_dict)
        logger.debug(f"Higher timeframe trend_direction: {trend_direction} (1=Up, -1=Down, 0=Neutral)", extra=log_extras)

        signal, signal_strength, message, buy_str, sell_str = self._check_entry_signal(
            df=df, current_price=current_price, trend_direction=trend_direction,
            sr_levels=sr_levels, position_info=position_info
        )

        # Calcular el porcentaje de proximidad al umbral de señal
        min_str = self._get_strategy_param(C.CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH, 0.7)
        buy_threshold_proximity_pct = (buy_str / min_str) * 100 if min_str > 0 else 0
        sell_threshold_proximity_pct = (sell_str / min_str) * 100 if min_str > 0 else 0
        
        # Nuevas métricas de proximidad al umbral
        proximity_metrics = {
            'buy_signal_strength': buy_str,
            'sell_signal_strength': sell_str,
            'min_signal_strength_required': min_str,
            'buy_threshold_proximity_pct': buy_threshold_proximity_pct,
            'sell_threshold_proximity_pct': sell_threshold_proximity_pct
        }
        
        # Log de las métricas de proximidad
        logger.info(f"Señal de COMPRA: {buy_threshold_proximity_pct:.1f}% del umbral necesario ({buy_str:.2f}/{min_str:.2f})", extra=log_extras)
        logger.info(f"Señal de VENTA: {sell_threshold_proximity_pct:.1f}% del umbral necesario ({sell_str:.2f}/{min_str:.2f})", extra=log_extras)

        if position_info and position_info.get(C.POSITION_TYPE, SignalType.NONE) != SignalType.NONE:
            logger.debug(f"Position currently open: Type={position_info.get(C.POSITION_TYPE)}, Entry={position_info.get(C.POSITION_OPEN_PRICE)}, SL={position_info.get(C.POSITION_SL)}, TP={position_info.get(C.POSITION_TP)}. Checking for exit.", extra=log_extras)
            exit_signal_type, exit_message = self._check_exit_signal(df, position_info, current_price)
            if exit_signal_type != SignalType.NONE:
                logger.info(f"Exit signal generated: Type={exit_signal_type}, Message='{exit_message}'. Overriding any entry signal.", extra=log_extras)
                signal = exit_signal_type
                message = exit_message
                signal_strength = 1.0

        indicators_for_result = {
            ind_name: df[ind_name].iloc[-1] if ind_name in df.columns and not df[ind_name].empty else None
            for ind_name in [C.INDICATOR_RSI, C.INDICATOR_MACD, C.INDICATOR_MACD_SIGNAL_LINE, C.INDICATOR_ADX, C.INDICATOR_ATR]
        }
        levels_for_result = {
            C.SR_SUPPORT: [lvl.price for lvl in sr_levels if lvl.type == C.SR_SUPPORT],
            C.SR_RESISTANCE: [lvl.price for lvl in sr_levels if lvl.type == C.SR_RESISTANCE],
        }

        logger.debug(f"Final analysis result for {self.symbol}: Signal={signal}, Strength={signal_strength:.2f}, SL={self.stop_loss:.5f}, TP={self.take_profit:.5f}, Msg='{message}'", extra=log_extras)
        logger.debug(f"Indicators at decision time: {indicators_for_result}", extra=log_extras)
        logger.debug(f"S/R levels at decision time: {levels_for_result}", extra=log_extras)

        return {
            'signal': signal, 'signal_strength': signal_strength, 'message': message,
            'entry_price': self.entry_price, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit,
            'indicators': indicators_for_result, 'levels': levels_for_result, 'trend_direction': trend_direction,
            'proximity_metrics': proximity_metrics  # Añadir métricas de proximidad al resultado
        }

    def _get_higher_timeframe_trend(self, data_dict: Dict[str, pd.DataFrame]) -> int:
        trend = 0
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME, 'method': '_get_higher_timeframe_trend'}
        logger.debug("Calculating higher timeframe trend...", extra=log_extras)
        for tf_key in [C.TF_H4, C.TF_D1]:
            if tf_key in data_dict and not data_dict[tf_key].empty and len(data_dict[tf_key]) >= 2:
                df_tf = data_dict[tf_key]
                if C.INDICATOR_EMA_50 in df_tf.columns and C.INDICATOR_EMA_200 in df_tf.columns:
                    ema50_last = df_tf[C.INDICATOR_EMA_50].iloc[-1]
                    ema200_last = df_tf[C.INDICATOR_EMA_200].iloc[-1]
                    ema50_prev = df_tf[C.INDICATOR_EMA_50].iloc[-2]
                    ema200_prev = df_tf[C.INDICATOR_EMA_200].iloc[-2]
                    logger.debug(f"Trend on {tf_key}: EMA50(last={ema50_last:.5f}, prev={ema50_prev:.5f}), EMA200(last={ema200_last:.5f}, prev={ema200_prev:.5f})", extra=log_extras)
                    if not pd.isna(ema50_last) and not pd.isna(ema200_last):
                        if ema50_last > ema200_last: trend += 1; logger.debug(f"{tf_key} contributes +1 to trend (EMA50 > EMA200).", extra=log_extras)
                        elif ema50_last < ema200_last: trend -= 1; logger.debug(f"{tf_key} contributes -1 to trend (EMA50 < EMA200).", extra=log_extras)
                        else: logger.debug(f"{tf_key} EMAs are equal, no trend contribution.", extra=log_extras)
                    else:
                        logger.debug(f"NaN values in EMAs for trend calc on {tf_key}.", extra=log_extras)
                else:
                    logger.debug(f"Trend EMAs not found or insufficient data on {tf_key} DataFrame.", extra=log_extras)
            else:
                logger.debug(f"Data for {tf_key} not found, empty, or less than 2 bars for trend calculation.", extra=log_extras)
        if trend > 0: return 1
        if trend < 0: return -1
        return 0

    def _check_entry_signal(self, df: pd.DataFrame, current_price: float, trend_direction: int,
                            sr_levels: List[SRLevel], position_info: Optional[Dict[str, Any]]) \
                            -> Tuple[int, float, str, float, float]:
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME, 'method': '_check_entry_signal'}
        logger.debug(f"Checking entry signal. Current Price: {current_price:.5f}, Trend: {trend_direction}", extra=log_extras)

        if position_info and position_info.get(C.POSITION_TYPE, SignalType.NONE) != SignalType.NONE:
            logger.debug("Already in a position, no new entry signal will be generated.", extra=log_extras)
            return SignalType.NONE, 0.0, "Already in a position", 0.0, 0.0

        enable_adx_filter = self._get_strategy_param(C.CONFIG_ENABLE_ADX_FILTER, C.DEFAULT_ENABLE_ADX_FILTER)
        adx_threshold = self._get_strategy_param(C.CONFIG_ADX_THRESHOLD, C.DEFAULT_ADX_THRESHOLD)
        logger.debug(f"ADX Filter: Enabled={enable_adx_filter}, Threshold={adx_threshold}", extra=log_extras)
        if enable_adx_filter:
            if C.INDICATOR_ADX not in df.columns:
                logger.warning(f"ADX filter enabled but ADX indicator not found.", extra=log_extras)
            else:
                latest_adx = df[C.INDICATOR_ADX].iloc[-1]
                log_msg_adx = f"ADX Value: {latest_adx:.2f}"
                if C.INDICATOR_PLUS_DI in df.columns and C.INDICATOR_MINUS_DI in df.columns:
                    plus_di = df[C.INDICATOR_PLUS_DI].iloc[-1]
                    minus_di = df[C.INDICATOR_MINUS_DI].iloc[-1]
                    log_msg_adx += f", +DI: {plus_di:.2f}, -DI: {minus_di:.2f}"
                logger.debug(log_msg_adx, extra=log_extras)

                if pd.isna(latest_adx): logger.warning(f"ADX value is NaN.", extra=log_extras)
                elif latest_adx < adx_threshold:
                    logger.info(f"ADX filter: Market not trending (ADX {latest_adx:.2f} < {adx_threshold:.2f}). No entry.", extra=log_extras)
                    return SignalType.NONE, 0.0, f"ADX non-trending (ADX {latest_adx:.2f})", 0.0, 0.0

        required_indicators = [C.INDICATOR_RSI, C.INDICATOR_MACD, C.INDICATOR_MACD_SIGNAL_LINE, C.INDICATOR_ATR]
        if not all(ind in df.columns and not df[ind].empty for ind in required_indicators):
            missing = [ind for ind in required_indicators if ind not in df.columns or df[ind].empty]
            logger.warning(f"Missing or empty indicators for entry: {missing}.", extra=log_extras)
            return SignalType.NONE, 0.0, f"Missing indicators: {missing}", 0.0, 0.0

        rsi = df[C.INDICATOR_RSI].iloc[-1]; macd = df[C.INDICATOR_MACD].iloc[-1]
        macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1]
        macd_prev = df[C.INDICATOR_MACD].iloc[-2] if len(df) > 1 else macd
        macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if len(df) > 1 else macd_s_line
        atr = df[C.INDICATOR_ATR].iloc[-1]
        logger.debug(f"Indicator values: RSI={rsi:.2f}, MACD={macd:.4f}, MACD_Signal={macd_s_line:.4f}, MACD_Prev={macd_prev:.4f}, MACD_Signal_Prev={macd_s_line_prev:.4f}, ATR={atr:.5f}", extra=log_extras)

        # Log last 2-3 values for key indicators
        num_vals_to_log = min(len(df), 3)
        if num_vals_to_log > 0:
            rsi_hist = df[C.INDICATOR_RSI].iloc[-num_vals_to_log:].values
            macd_hist = df[C.INDICATOR_MACD].iloc[-num_vals_to_log:].values
            macdsig_hist = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-num_vals_to_log:].values
            atr_hist = df[C.INDICATOR_ATR].iloc[-num_vals_to_log:].values
            logger.debug(f"  RSI History (last {num_vals_to_log}): {[f'{x:.2f}' for x in rsi_hist]}", extra=log_extras)
            logger.debug(f"  MACD History (last {num_vals_to_log}): {[f'{x:.4f}' for x in macd_hist]}", extra=log_extras)
            logger.debug(f"  MACD Signal History (last {num_vals_to_log}): {[f'{x:.4f}' for x in macdsig_hist]}", extra=log_extras)
            logger.debug(f"  ATR History (last {num_vals_to_log}): {[f'{x:.5f}' for x in atr_hist]}", extra=log_extras)

        if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_s_line) or pd.isna(atr) or atr == 0:
            logger.warning("NaN or zero ATR detected in critical indicators. No entry.", extra=log_extras)
            return SignalType.NONE, 0.0, "NaN in critical indicators or zero ATR", 0.0, 0.0

        rsi_oversold = self._get_strategy_param('rsi_oversold_threshold', 30)
        rsi_overbought = self._get_strategy_param('rsi_overbought_threshold', 70)
        sr_prox_perc = self._get_strategy_param('sr_proximity_percent', 0.5)
        logger.debug(f"Thresholds: RSI Oversold={rsi_oversold}, RSI Overbought={rsi_overbought}, SR Proximity%={sr_prox_perc}", extra=log_extras)

        macd_cross_up = macd_prev < macd_s_line_prev and macd > macd_s_line
        macd_cross_down = macd_prev > macd_s_line_prev and macd < macd_s_line
        logger.debug(f"MACD Cross Up: {macd_cross_up}, MACD Cross Down: {macd_cross_down}", extra=log_extras)

        rsi_buy_cond = rsi < rsi_oversold or (rsi_oversold <= rsi <= rsi_overbought and trend_direction >= 0)
        rsi_sell_cond = rsi > rsi_overbought or (rsi_oversold <= rsi <= rsi_overbought and trend_direction <= 0)
        logger.debug(f"RSI Buy Condition: {rsi_buy_cond}, RSI Sell Condition: {rsi_sell_cond}", extra=log_extras)

        sup = max((lvl.price for lvl in sr_levels if lvl.type == C.SR_SUPPORT and lvl.price < current_price), default=0.0)
        res = min((lvl.price for lvl in sr_levels if lvl.type == C.SR_RESISTANCE and lvl.price > current_price), default=float('inf'))
        logger.debug(f"Nearest Support (sup): {sup:.5f}, Nearest Resistance (res): {res:.5f}", extra=log_extras)

        sup_cond = (current_price - sup) / current_price * 100 < sr_prox_perc if sup > 0 else False
        res_cond = (res - current_price) / current_price * 100 < sr_prox_perc if res != float('inf') else False
        logger.debug(f"Support Proximity Condition (sup_cond): {sup_cond}, Resistance Proximity Condition (res_cond): {res_cond}", extra=log_extras)

        buy_conds = [macd_cross_up, rsi_buy_cond, sup_cond, trend_direction >= 0]
        sell_conds = [macd_cross_down, rsi_sell_cond, res_cond, trend_direction <= 0]
        logger.debug(f"Buy Conditions Eval: MACD_XU={buy_conds[0]}, RSI_Buy={buy_conds[1]}, Sup_Prox={buy_conds[2]}, Trend_UpOrNeut={buy_conds[3]}", extra=log_extras)
        logger.debug(f"Sell Conditions Eval: MACD_XD={sell_conds[0]}, RSI_Sell={sell_conds[1]}, Res_Prox={sell_conds[2]}, Trend_DownOrNeut={sell_conds[3]}", extra=log_extras)

        buy_str = sum(1 for c in buy_conds if c) / len(buy_conds)
        sell_str = sum(1 for c in sell_conds if c) / len(sell_conds)
        logger.debug(f"Buy Signal Strength: {buy_str:.2f}, Sell Signal Strength: {sell_str:.2f}", extra=log_extras)

        min_str = self._get_strategy_param(C.CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH, 0.7)
        logger.debug(f"Minimum Signal Strength Required: {min_str:.2f}", extra=log_extras)

        use_atr_sl_tp = self._get_strategy_param(C.CONFIG_USE_ATR_SL_TP, C.DEFAULT_USE_ATR_SL_TP)
        sl_mult = self._get_strategy_param(C.CONFIG_ATR_SL_MULTIPLIER, C.DEFAULT_ATR_SL_MULTIPLIER)
        tp_mult = self._get_strategy_param(C.CONFIG_ATR_TP_MULTIPLIER, C.DEFAULT_ATR_TP_MULTIPLIER)
        fb_sl_mult = self._get_strategy_param(C.OLD_SL_ATR_MULTIPLIER, getattr(C, 'DEFAULT_FALLBACK_SL_ATR_MULTIPLIER', 2.0))
        fb_tp_rr = self._get_strategy_param(C.CONFIG_RISK_REWARD_RATIO, getattr(C, 'DEFAULT_FALLBACK_TP_RR_RATIO', 1.5))
        logger.debug(f"SL/TP Params: Use_ATR_SL_TP={use_atr_sl_tp}, SL_Mult={sl_mult}, TP_Mult={tp_mult}, Fallback_SL_Mult={fb_sl_mult}, Fallback_TP_RR={fb_tp_rr}", extra=log_extras)

        self.entry_price = 0.0; self.stop_loss = 0.0; self.take_profit = 0.0

        if buy_str >= min_str and buy_str > sell_str:
            self.entry_price = current_price
            sl_established = False
            tp_established = False

            # 1. Intento principal de SL/TP
            if use_atr_sl_tp:
                logger.debug(f"Attempting ATR based SL for BUY. CurrentPrice={current_price:.5f}", extra=log_extras)
                potential_sl = current_price - (atr * sl_mult)
                potential_tp = current_price + (atr * tp_mult)
                if potential_sl > 0 and potential_sl < self.entry_price:
                    self.stop_loss = potential_sl
                    sl_established = True
                    if potential_tp > self.entry_price:
                        self.take_profit = potential_tp
                        tp_established = True
                if not sl_established:
                    logger.warning(f"ATR SL (primary) for BUY invalid: SL={potential_sl:.5f}, Entry={self.entry_price:.5f}", extra=log_extras)

            elif sup > 0: # No use_atr_sl_tp, intentar S/R
                logger.debug(f"Attempting S/R based SL for BUY. Support: {sup:.5f}", extra=log_extras)
                potential_sl = min(sup, current_price - (atr * fb_sl_mult)) # Usar fb_sl_mult como un safety net con S/R
                if potential_sl > 0 and potential_sl < self.entry_price:
                    self.stop_loss = potential_sl
                    sl_established = True
                    # TP para S/R se calcula siempre con R:R
                else:
                    logger.warning(f"S/R SL for BUY invalid: SL={potential_sl:.5f}, Support={sup:.5f}, Entry={self.entry_price:.5f}", extra=log_extras)
            
            # 2. Fallback de SL si no se estableció
            if not sl_established:
                logger.warning(f"Primary SL for BUY failed or not applicable. Using ATR fallback. CurrentPrice={current_price:.5f}, ATR={atr:.5f}, FB_Mult={fb_sl_mult}", extra=log_extras)
                self.stop_loss = current_price - (atr * fb_sl_mult)
                if not (self.stop_loss > 0 and self.stop_loss < self.entry_price):
                    logger.error(f"CRITICAL: ATR Fallback SL for BUY is invalid: SL={self.stop_loss:.5f}. Entry: {current_price:.5f}. Resetting.", extra=log_extras)
                    return SignalType.NONE, 0.0, "ATR Fallback SL invalid for BUY", buy_str, sell_str
                # Si el SL de fallback es válido, sl_established se considera verdadero para el cálculo de TP
                logger.debug(f"ATR Fallback SL for BUY set: {self.stop_loss:.5f}", extra=log_extras)
                sl_established = True 

            # 3. Cálculo de TP (si no se estableció con use_atr_sl_tp o si se usó S/R o fallback SL)
            if sl_established and not tp_established: # SL debe ser válido para calcular TP basado en R:R
                logger.debug(f"Calculating TP for BUY using R:R={fb_tp_rr}. Entry={self.entry_price:.5f}, SL={self.stop_loss:.5f}", extra=log_extras)
                self.take_profit = self.entry_price + ((self.entry_price - self.stop_loss) * fb_tp_rr)
                tp_established = True
                logger.debug(f"TP for BUY set by R:R: {self.take_profit:.5f}", extra=log_extras)

            # 4. Validación final de SL y TP
            if not sl_established or self.stop_loss <= 0 or self.stop_loss >= self.entry_price:
                logger.error(f"FINAL SL CHECK FAILED for BUY: SL={self.stop_loss:.5f}, Entry={self.entry_price:.5f}. No signal.", extra=log_extras)
                return SignalType.NONE, 0.0, "Final SL validation failed for BUY", buy_str, sell_str

            if tp_established and (self.take_profit <= self.entry_price or self.take_profit == 0):
                logger.warning(f"TP for BUY ({self.take_profit:.5f}) is not profitable or zero. Setting TP to 0.0 (no TP). Entry: {self.entry_price:.5f}, SL: {self.stop_loss:.5f}", extra=log_extras)
                self.take_profit = 0.0 # Permitir trades sin TP si el cálculo es problemático pero SL está bien
            elif not tp_established: # Si por alguna razón TP no se estableció
                 logger.warning(f"TP for BUY could not be established. Setting TP to 0.0 (no TP).", extra=log_extras)
                 self.take_profit = 0.0

            logger.info(f"BUY Signal generated: Strength={buy_str:.2f}, Entry={self.entry_price:.5f}, SL={self.stop_loss:.5f}, TP={self.take_profit:.5f}. Message: MACD Buy", extra=log_extras)
            return SignalType.BUY, buy_str, "MACD Buy", buy_str, sell_str

        if sell_str >= min_str:
            self.entry_price = current_price
            sl_established = False
            tp_established = False

            # 1. Intento principal de SL/TP
            if use_atr_sl_tp:
                logger.debug(f"Attempting ATR based SL for SELL. CurrentPrice={current_price:.5f}", extra=log_extras)
                potential_sl = current_price + (atr * sl_mult)
                potential_tp = current_price - (atr * tp_mult)
                if potential_sl > self.entry_price:
                    self.stop_loss = potential_sl
                    sl_established = True
                    if potential_tp < self.entry_price and potential_tp > 0:
                        self.take_profit = potential_tp
                        tp_established = True
                if not sl_established:
                    logger.warning(f"ATR SL (primary) for SELL invalid: SL={potential_sl:.5f}, Entry={self.entry_price:.5f}", extra=log_extras)
            
            elif res != float('inf'): # No use_atr_sl_tp, intentar S/R
                logger.debug(f"Attempting S/R based SL for SELL. Resistance: {res:.5f}", extra=log_extras)
                potential_sl = max(res, current_price + (atr * fb_sl_mult))
                if potential_sl > self.entry_price:
                    self.stop_loss = potential_sl
                    sl_established = True
                    # TP para S/R se calcula siempre con R:R
                else:
                    logger.warning(f"S/R SL for SELL invalid: SL={potential_sl:.5f}, Resistance={res:.5f}, Entry={self.entry_price:.5f}", extra=log_extras)

            # 2. Fallback de SL si no se estableció
            if not sl_established:
                logger.warning(f"Primary SL for SELL failed or not applicable. Using ATR fallback. CurrentPrice={current_price:.5f}, ATR={atr:.5f}, FB_Mult={fb_sl_mult}", extra=log_extras)
                self.stop_loss = current_price + (atr * fb_sl_mult)
                if not (self.stop_loss > self.entry_price):
                    logger.error(f"CRITICAL: ATR Fallback SL for SELL is invalid: SL={self.stop_loss:.5f}. Entry: {current_price:.5f}. Resetting.", extra=log_extras)
                    return SignalType.NONE, 0.0, "ATR Fallback SL invalid for SELL", buy_str, sell_str
                logger.debug(f"ATR Fallback SL for SELL set: {self.stop_loss:.5f}", extra=log_extras)
                sl_established = True # Si el SL de fallback es válido

            # 3. Cálculo de TP (si no se estableció con use_atr_sl_tp o si se usó S/R o fallback SL)
            if sl_established and not tp_established:
                logger.debug(f"Calculating TP for SELL using R:R={fb_tp_rr}. Entry={self.entry_price:.5f}, SL={self.stop_loss:.5f}", extra=log_extras)
                self.take_profit = self.entry_price - ((self.stop_loss - self.entry_price) * fb_tp_rr)
                tp_established = True
                logger.debug(f"TP for SELL set by R:R: {self.take_profit:.5f}", extra=log_extras)

            # 4. Validación final de SL y TP
            if not sl_established or self.stop_loss <= self.entry_price:
                logger.error(f"FINAL SL CHECK FAILED for SELL: SL={self.stop_loss:.5f}, Entry={self.entry_price:.5f}. No signal.", extra=log_extras)
                return SignalType.NONE, 0.0, "Final SL validation failed for SELL", buy_str, sell_str

            if tp_established and (self.take_profit >= self.entry_price or self.take_profit <= 0):
                logger.warning(f"TP for SELL ({self.take_profit:.5f}) is not profitable or zero/negative. Setting TP to 0.0 (no TP). Entry: {self.entry_price:.5f}, SL: {self.stop_loss:.5f}", extra=log_extras)
                self.take_profit = 0.0
            elif not tp_established:
                 logger.warning(f"TP for SELL could not be established. Setting TP to 0.0 (no TP).", extra=log_extras)
                 self.take_profit = 0.0

            logger.info(f"SELL Signal generated: Strength={sell_str:.2f}, Entry={self.entry_price:.5f}, SL={self.stop_loss:.5f}, TP={self.take_profit:.5f}. Message: MACD Sell", extra=log_extras)
            return SignalType.SELL, sell_str, "MACD Sell", buy_str, sell_str

        logger.debug("No entry signal generated based on current conditions.", extra=log_extras)
        return SignalType.NONE, 0.0, "No signal", buy_str, sell_str

    def _check_exit_signal(self, df: pd.DataFrame, position_info: Dict[str, Any], current_price: float) \
                           -> Tuple[int, str]:
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME, 'method': '_check_exit_signal', 'ticket': position_info.get(C.POSITION_TICKET)}
        pos_type = position_info[C.POSITION_TYPE]
        entry_price = position_info[C.POSITION_OPEN_PRICE]
        logger.debug(f"Checking exit for position Type={pos_type}, Entry={entry_price:.5f}, CurrentPrice={current_price:.5f}", extra=log_extras)

        rsi = df[C.INDICATOR_RSI].iloc[-1] if C.INDICATOR_RSI in df.columns else 50
        macd = df[C.INDICATOR_MACD].iloc[-1] if C.INDICATOR_MACD in df.columns else 0
        macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1] if C.INDICATOR_MACD_SIGNAL_LINE in df.columns else 0
        macd_prev = df[C.INDICATOR_MACD].iloc[-2] if C.INDICATOR_MACD in df.columns and len(df) > 1 else macd
        macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if C.INDICATOR_MACD_SIGNAL_LINE in df.columns and len(df) > 1 else macd_s_line

        rsi_oversold = self._get_strategy_param('rsi_oversold_threshold', 30)
        rsi_overbought = self._get_strategy_param('rsi_overbought_threshold', 70)
        logger.debug(f"Exit Check Indicators: RSI={rsi:.2f}, MACD={macd:.4f}, MACD_Signal={macd_s_line:.4f}. RSI Thresholds: Oversold={rsi_oversold}, Overbought={rsi_overbought}", extra=log_extras)

        if pos_type == SignalType.BUY:
            # MACD cross down for exiting a BUY
            if macd_prev > macd_s_line_prev and macd < macd_s_line:
                logger.info("Exit BUY condition: MACD cross down.", extra=log_extras)
                return SignalType.SELL, "Exit: MACD cross down"
            # RSI overbought for exiting a BUY
            if rsi > rsi_overbought:
                logger.info(f"Exit BUY condition: RSI overbought (RSI={rsi:.2f} > {rsi_overbought}).", extra=log_extras)
                return SignalType.SELL, "Exit: RSI overbought"
        elif pos_type == SignalType.SELL:
            # MACD cross up for exiting a SELL
            if macd_prev < macd_s_line_prev and macd > macd_s_line:
                logger.info("Exit SELL condition: MACD cross up.", extra=log_extras)
                return SignalType.BUY, "Exit: MACD cross up"
            # RSI oversold for exiting a SELL
            if rsi < rsi_oversold:
                logger.info(f"Exit SELL condition: RSI oversold (RSI={rsi:.2f} < {rsi_oversold}).", extra=log_extras)
                return SignalType.BUY, "Exit: RSI oversold"
        
        logger.debug("No exit signal based on strategy rules.", extra=log_extras)
        return SignalType.NONE, "No exit signal"
