import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

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

        # Initialize SRHandler using the passed ConfigManager and symbol-specific SR config
        # SR parameters should be part of the broader symbol configuration, fetched by ConfigManager
        sr_config_for_symbol = self.config_manager.get_sr_params(self.symbol)
        # Note: SRHandler's __init__ might need adjustment if it expects just a dict vs. full ConfigManager
        # Current SRHandler __init__(self, config: Dict[str, Any]) -> self.config = config
        # So, passing sr_config_for_symbol directly is appropriate.
        self.sr_handler = SRHandler(sr_config_for_symbol)


    def analyze(self, data_dict: Dict[str, pd.DataFrame], position_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME}
        # Determine primary timeframe
        primary_tf = self._get_strategy_param('primary_timeframe', list(data_dict.keys())[0] if data_dict else C.TF_M15)

        if not data_dict or primary_tf not in data_dict or data_dict[primary_tf].empty:
            logger.warning(f"Primary timeframe {primary_tf} not found or empty in data_dict.", extra=log_extras)
            return {'signal': SignalType.NONE, 'message': f"No data for primary timeframe {primary_tf}"}

        df = data_dict[primary_tf] # Strategies should use the data as passed; calculation is done by engine/wrapper

        # Ensure required columns are present (indicators should be pre-calculated on df)
        required_cols_for_analyze = [C.INDICATOR_CLOSE_PRICE] # Minimum
        # Add other indicators MACDStrategy directly uses from df to this list
        if not all(col in df.columns for col in required_cols_for_analyze):
            logger.warning(f"DataFrame missing required columns for analysis. Needs: {required_cols_for_analyze}", extra=log_extras)
            return {'signal': SignalType.NONE, 'message': "DataFrame missing essential columns"}


        sr_levels = self.sr_handler.get_sr_levels(df, symbol=self.symbol) # Pass symbol if sr_handler needs it
        current_price = df[C.INDICATOR_CLOSE_PRICE].iloc[-1]

        trend_direction = self._get_higher_timeframe_trend(data_dict)

        signal, signal_strength, message = self._check_entry_signal(
            df=df, current_price=current_price, trend_direction=trend_direction,
            sr_levels=sr_levels, position_info=position_info
        )

        if position_info and position_info.get(C.POSITION_TYPE, SignalType.NONE) != SignalType.NONE:
            exit_signal_type, exit_message = self._check_exit_signal(df, position_info, current_price)
            if exit_signal_type != SignalType.NONE:
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

        return {
            'signal': signal, 'signal_strength': signal_strength, 'message': message,
            'entry_price': self.entry_price, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit,
            'indicators': indicators_for_result, 'levels': levels_for_result, 'trend_direction': trend_direction
        }

    def _get_higher_timeframe_trend(self, data_dict: Dict[str, pd.DataFrame]) -> int:
        trend = 0
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME}
        for tf_key in [C.TF_H4, C.TF_D1]:
            if tf_key in data_dict and not data_dict[tf_key].empty:
                df_tf = data_dict[tf_key] # Assume indicators are already on this df
                if C.INDICATOR_EMA_50 in df_tf.columns and C.INDICATOR_EMA_200 in df_tf.columns:
                    ema50 = df_tf[C.INDICATOR_EMA_50].iloc[-1]
                    ema200 = df_tf[C.INDICATOR_EMA_200].iloc[-1]
                    if not pd.isna(ema50) and not pd.isna(ema200):
                        if ema50 > ema200: trend += 1
                        elif ema50 < ema200: trend -= 1
                    else:
                        logger.debug(f"NaN values in EMAs for trend calc on {tf_key}.", extra=log_extras)
                else:
                    logger.debug(f"Trend EMAs not found on {tf_key} DataFrame.", extra=log_extras)
        if trend > 0: return 1
        if trend < 0: return -1
        return 0

    def _check_entry_signal(self, df: pd.DataFrame, current_price: float, trend_direction: int,
                            sr_levels: List[SRLevel], position_info: Optional[Dict[str, Any]]) \
                            -> Tuple[int, float, str]:
        log_extras = {'symbol': self.symbol, 'strategy': self.NAME}

        if position_info and position_info.get(C.POSITION_TYPE, SignalType.NONE) != SignalType.NONE:
            return SignalType.NONE, 0.0, "Already in a position"

        enable_adx_filter = self._get_strategy_param(C.CONFIG_ENABLE_ADX_FILTER, C.DEFAULT_ENABLE_ADX_FILTER)
        if enable_adx_filter:
            adx_threshold = self._get_strategy_param(C.CONFIG_ADX_THRESHOLD, C.DEFAULT_ADX_THRESHOLD)
            if C.INDICATOR_ADX not in df.columns:
                logger.warning(f"ADX filter enabled but ADX indicator not found.", extra=log_extras)
            else:
                latest_adx = df[C.INDICATOR_ADX].iloc[-1]
                if pd.isna(latest_adx): logger.warning(f"ADX value is NaN.", extra=log_extras)
                elif latest_adx < adx_threshold:
                    logger.info(f"ADX filter: Market not trending (ADX {latest_adx:.2f} < {adx_threshold:.2f}).", extra=log_extras)
                    return SignalType.NONE, 0.0, f"ADX non-trending (ADX {latest_adx:.2f})"

        required_indicators = [C.INDICATOR_RSI, C.INDICATOR_MACD, C.INDICATOR_MACD_SIGNAL_LINE, C.INDICATOR_ATR]
        if not all(ind in df.columns and not df[ind].empty for ind in required_indicators):
            missing = [ind for ind in required_indicators if ind not in df.columns or df[ind].empty]
            logger.warning(f"Missing or empty indicators for entry: {missing}.", extra=log_extras)
            return SignalType.NONE, 0.0, f"Missing indicators: {missing}"

        rsi = df[C.INDICATOR_RSI].iloc[-1]; macd = df[C.INDICATOR_MACD].iloc[-1]
        macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1]
        macd_prev = df[C.INDICATOR_MACD].iloc[-2] if len(df) > 1 else macd
        macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if len(df) > 1 else macd_s_line
        atr = df[C.INDICATOR_ATR].iloc[-1]

        if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_s_line) or pd.isna(atr) or atr == 0:
            logger.warning("NaN or zero ATR detected in critical indicators.", extra=log_extras)
            return SignalType.NONE, 0.0, "NaN in critical indicators or zero ATR"

        rsi_oversold = self._get_strategy_param('rsi_oversold_threshold', 30)
        rsi_overbought = self._get_strategy_param('rsi_overbought_threshold', 70)
        sr_prox_perc = self._get_strategy_param('sr_proximity_percent', 0.5)

        macd_cross_up = macd_prev < macd_s_line_prev and macd > macd_s_line
        macd_cross_down = macd_prev > macd_s_line_prev and macd < macd_s_line

        rsi_buy_cond = rsi < rsi_oversold or (rsi_oversold <= rsi <= rsi_overbought and trend_direction >= 0)
        rsi_sell_cond = rsi > rsi_overbought or (rsi_oversold <= rsi <= rsi_overbought and trend_direction <= 0)

        sup = max((lvl.price for lvl in sr_levels if lvl.type == C.SR_SUPPORT and lvl.price < current_price), default=0.0)
        res = min((lvl.price for lvl in sr_levels if lvl.type == C.SR_RESISTANCE and lvl.price > current_price), default=float('inf'))

        sup_cond = (current_price - sup) / current_price * 100 < sr_prox_perc if sup > 0 else False
        res_cond = (res - current_price) / current_price * 100 < sr_prox_perc if res != float('inf') else False

        buy_conds = [macd_cross_up, rsi_buy_cond, sup_cond, trend_direction >= 0]
        sell_conds = [macd_cross_down, rsi_sell_cond, res_cond, trend_direction <= 0]

        buy_str = sum(1 for c in buy_conds if c) / len(buy_conds)
        sell_str = sum(1 for c in sell_conds if c) / len(sell_conds)

        min_str = self._get_strategy_param(C.CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH, 0.7)

        use_atr_sl_tp = self._get_strategy_param(C.CONFIG_USE_ATR_SL_TP, C.DEFAULT_USE_ATR_SL_TP)
        sl_mult = self._get_strategy_param(C.CONFIG_ATR_SL_MULTIPLIER, C.DEFAULT_ATR_SL_MULTIPLIER)
        tp_mult = self._get_strategy_param(C.CONFIG_ATR_TP_MULTIPLIER, C.DEFAULT_ATR_TP_MULTIPLIER)
        fb_sl_mult = self._get_strategy_param(C.OLD_SL_ATR_MULTIPLIER, getattr(C, 'DEFAULT_FALLBACK_SL_ATR_MULTIPLIER', 2.0))
        fb_tp_rr = self._get_strategy_param(C.CONFIG_RISK_REWARD_RATIO, getattr(C, 'DEFAULT_FALLBACK_TP_RR_RATIO', 1.5))

        self.entry_price = 0.0; self.stop_loss = 0.0; self.take_profit = 0.0

        if buy_str >= min_str and buy_str > sell_str:
            self.entry_price = current_price
            if use_atr_sl_tp:
                self.stop_loss = current_price - (atr * sl_mult)
                self.take_profit = current_price + (atr * tp_mult)
            elif sup > 0:
                self.stop_loss = min(sup, current_price - (atr * fb_sl_mult))
                if current_price > self.stop_loss > 0: self.take_profit = current_price + ((current_price - self.stop_loss) * fb_tp_rr)
                else: logger.warning("Invalid S/R SL for BUY.", extra=log_extras); return SignalType.NONE,0.0,"S/R SL issue"
            else: logger.warning("Cannot calc SL for BUY.", extra=log_extras); return SignalType.NONE,0.0,"SL calc issue"
            return SignalType.BUY, buy_str, "MACD Buy"

        if sell_str >= min_str:
            self.entry_price = current_price
            if use_atr_sl_tp:
                self.stop_loss = current_price + (atr * sl_mult)
                self.take_profit = current_price - (atr * tp_mult)
            elif res != float('inf'):
                self.stop_loss = max(res, current_price + (atr * fb_sl_mult))
                if current_price < self.stop_loss: self.take_profit = current_price - ((self.stop_loss - current_price) * fb_tp_rr)
                else: logger.warning("Invalid S/R SL for SELL.", extra=log_extras); return SignalType.NONE,0.0,"S/R SL issue"
            else: logger.warning("Cannot calc SL for SELL.", extra=log_extras); return SignalType.NONE,0.0,"SL calc issue"
            return SignalType.SELL, sell_str, "MACD Sell"

        return SignalType.NONE, 0.0, "No signal"

    def _check_exit_signal(self, df: pd.DataFrame, position_info: Dict[str, Any], current_price: float) \
                           -> Tuple[int, str]:
        log_extras = {'symbol': self.symbol, 'ticket': position_info.get(C.POSITION_TICKET)}
        pos_type = position_info[C.POSITION_TYPE]

        rsi = df[C.INDICATOR_RSI].iloc[-1] if C.INDICATOR_RSI in df.columns else 50
        macd = df[C.INDICATOR_MACD].iloc[-1] if C.INDICATOR_MACD in df.columns else 0
        macd_s_line = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-1] if C.INDICATOR_MACD_SIGNAL_LINE in df.columns else 0
        macd_prev = df[C.INDICATOR_MACD].iloc[-2] if C.INDICATOR_MACD in df.columns and len(df) > 1 else macd
        macd_s_line_prev = df[C.INDICATOR_MACD_SIGNAL_LINE].iloc[-2] if C.INDICATOR_MACD_SIGNAL_LINE in df.columns and len(df) > 1 else macd_s_line

        rsi_oversold = self._get_strategy_param('rsi_oversold_threshold', 30)
        rsi_overbought = self._get_strategy_param('rsi_overbought_threshold', 70)

        if pos_type == SignalType.BUY:
            if macd_prev > macd_s_line_prev and macd < macd_s_line:
                logger.info("Exit BUY: MACD cross down.", extra=log_extras)
                return SignalType.SELL, "Exit: MACD cross down"
            if rsi > rsi_overbought:
                logger.info(f"Exit BUY: RSI overbought ({rsi:.2f}).", extra=log_extras)
                return SignalType.SELL, "Exit: RSI overbought"
        elif pos_type == SignalType.SELL:
            if macd_prev < macd_s_line_prev and macd > macd_s_line:
                logger.info("Exit SELL: MACD cross up.", extra=log_extras)
                return SignalType.BUY, "Exit: MACD cross up"
            if rsi < rsi_oversold:
                logger.info(f"Exit SELL: RSI oversold ({rsi:.2f}).", extra=log_extras)
                return SignalType.BUY, "Exit: RSI oversold"

        return SignalType.NONE, "No exit signal"

```
