"""
Module for calculating technical indicators.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable, TypeVar, Protocol, cast
import logging
from functools import wraps
import abc
from .utils.cache_manager import cache_manager # Import the singleton cache_manager
from . import constants as C # Import constants

logger = logging.getLogger(__name__)

# Define a type variable for decorator usage
T = TypeVar('T')

def handle_empty_df(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle empty DataFrames and config initialization.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> T:
        if df.empty:
            # Return None for functions that return None, empty DataFrame for others
            return_type = func.__annotations__.get('return')
            if return_type == 'None' or return_type is None:
                return cast(T, None)
            else:
                return cast(T, df)
        
        # Initialize config if not provided
        effective_config = config or {}
        
        return func(self, df, effective_config)
    
    return wrapper

class IndicatorGroup(abc.ABC):
    """Abstract base class for indicator groups."""
    
    @abc.abstractmethod
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate indicators and add them to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with indicator parameters
        """
        pass

class TrendIndicators(IndicatorGroup):
    """Group for trend-following indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate trend indicators.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'close' column)
            config: Configuration dictionary with trend indicator parameters
        """
        # Ensure column names are lowercase for consistency
        # This should ideally be done once when df is first received by IndicatorCalculator
        # df.columns = [col.lower() for col in df.columns]
        close_prices = df[C.INDICATOR_CLOSE_PRICE if hasattr(C, 'INDICATOR_CLOSE_PRICE') else 'close']

        # Calculate MACD
        fast = int(config.get(C.CONFIG_INDICATOR_MACD_FAST, 12))
        slow = int(config.get(C.CONFIG_INDICATOR_MACD_SLOW, 26))
        signal_period = int(config.get(C.CONFIG_INDICATOR_MACD_SIGNAL, 9)) # Renamed signal to signal_period

        # Cache key for MACD
        macd_cache_key = f"macd:{pd.util.hash_pandas_object(close_prices).sum()}:{fast}:{slow}:{signal_period}"
        cached_macd_data = cache_manager.get(macd_cache_key)

        if cached_macd_data is not None:
            logger.debug(f"MACD cache hit for key: {macd_cache_key}")
            df[C.INDICATOR_MACD] = cached_macd_data[C.INDICATOR_MACD]
            df[C.INDICATOR_MACD_SIGNAL_LINE] = cached_macd_data[C.INDICATOR_MACD_SIGNAL_LINE]
            # df[C.INDICATOR_MACD_HISTOGRAM] = cached_macd_data[C.INDICATOR_MACD_HISTOGRAM] # If histogram is needed
        else:
            logger.debug(f"MACD cache miss for key: {macd_cache_key}")
            macd_df = ta.macd(
                close_prices,
                fast=fast,
                slow=slow,
                signal=signal_period, # Use signal_period
                append=False
            )
            if macd_df is not None and not macd_df.empty:
                macd_col_name = f'MACD_{fast}_{slow}_{signal_period}'
                signal_col_name = f'MACDs_{fast}_{slow}_{signal_period}'
                # hist_col_name = f'MACDh_{fast}_{slow}_{signal_period}'

                df[C.INDICATOR_MACD] = macd_df[macd_col_name]
                df[C.INDICATOR_MACD_SIGNAL_LINE] = macd_df[signal_col_name]
                # df[C.INDICATOR_MACD_HISTOGRAM] = macd_df[hist_col_name]

                # Store necessary series in cache
                cache_manager.set(macd_cache_key, {
                    C.INDICATOR_MACD: df[C.INDICATOR_MACD],
                    C.INDICATOR_MACD_SIGNAL_LINE: df[C.INDICATOR_MACD_SIGNAL_LINE],
                    # C.INDICATOR_MACD_HISTOGRAM: df[C.INDICATOR_MACD_HISTOGRAM]
                })
        
        # Calculate Moving Averages
        # Calculate Moving Averages
        for ma_type_str in ['sma', 'ema']: # ma_type_str to avoid conflict
            for period_val in [20, 50, 100, 200]: # period_val to avoid conflict
                ma_col_name = f'{ma_type_str.upper()}_{period_val}' # e.g., EMA_50
                # Use more specific config keys if available, e.g., config.get('enable_ema_50', True)
                if config.get(f'enable_{ma_type_str}{period_val}', True):
                    ma_cache_key = f"{ma_type_str}:{pd.util.hash_pandas_object(close_prices).sum()}:{period_val}"
                    cached_ma = cache_manager.get(ma_cache_key)
                    if cached_ma is not None:
                        df[ma_col_name] = cached_ma
                    else:
                        if ma_type_str == 'sma':
                            ma_series = ta.sma(close_prices, length=period_val)
                        else: # ema
                            ma_series = ta.ema(close_prices, length=period_val)

                        if ma_series is not None:
                            df[ma_col_name] = ma_series
                            cache_manager.set(ma_cache_key, ma_series)
        
        # Calculate ADX
        adx_period_val = int(config.get('adx_period', 14)) # adx_period_val
        # ADX uses High, Low, Close. Create a combined hash for these.
        hlc_hash = (pd.util.hash_pandas_object(df[C.INDICATOR_HIGH_PRICE if hasattr(C, 'INDICATOR_HIGH_PRICE') else 'high']).sum() +
                    pd.util.hash_pandas_object(df[C.INDICATOR_LOW_PRICE if hasattr(C, 'INDICATOR_LOW_PRICE') else 'low']).sum() +
                    close_hash) # close_hash from MACD section, or re-calculate if not available
        
        adx_cache_key = f"adx:{hlc_hash}:{adx_period_val}"
        cached_adx_data = cache_manager.get(adx_cache_key)

        if cached_adx_data is not None:
            logger.debug(f"ADX cache hit for key: {adx_cache_key}")
            df[C.INDICATOR_ADX if hasattr(C, 'INDICATOR_ADX') else 'ADX'] = cached_adx_data['ADX']
            # df['DMP'] = cached_adx_data['DMP'] # Define constants if using these
            # df['DMN'] = cached_adx_data['DMN']
        else:
            logger.debug(f"ADX cache miss for key: {adx_cache_key}")
            adx_df = ta.adx(
                high=df[C.INDICATOR_HIGH_PRICE if hasattr(C, 'INDICATOR_HIGH_PRICE') else 'high'],
                low=df[C.INDICATOR_LOW_PRICE if hasattr(C, 'INDICATOR_LOW_PRICE') else 'low'],
                close=close_prices,
                length=adx_period_val,
                append=False
            )
            if adx_df is not None and not adx_df.empty:
                adx_col = f'ADX_{adx_period_val}'
                # dmp_col = f'DMP_{adx_period_val}'
                # dmn_col = f'DMN_{adx_period_val}'
                df[C.INDICATOR_ADX if hasattr(C, 'INDICATOR_ADX') else 'ADX'] = adx_df[adx_col]
                # df['DMP'] = adx_df[dmp_col]
                # df['DMN'] = adx_df[dmn_col]
                cache_manager.set(adx_cache_key, {
                    'ADX': df[C.INDICATOR_ADX if hasattr(C, 'INDICATOR_ADX') else 'ADX'],
                    # 'DMP': df['DMP'], 'DMN': df['DMN']
                })

class MomentumIndicators(IndicatorGroup):
    """Group for momentum indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'close', 'high', 'low' columns)
            config: Configuration dictionary with momentum indicator parameters
        """
        close_prices = df[C.INDICATOR_CLOSE_PRICE if hasattr(C, 'INDICATOR_CLOSE_PRICE') else 'close']
        high_prices = df[C.INDICATOR_HIGH_PRICE if hasattr(C, 'INDICATOR_HIGH_PRICE') else 'high']
        low_prices = df[C.INDICATOR_LOW_PRICE if hasattr(C, 'INDICATOR_LOW_PRICE') else 'low']

        # Calculate RSI
        rsi_period = int(config.get(C.CONFIG_INDICATOR_RSI_PERIOD, 14))
        
        rsi_cache_key = f"rsi:{pd.util.hash_pandas_object(close_prices).sum()}:{rsi_period}"
        cached_rsi = cache_manager.get(rsi_cache_key)

        if cached_rsi is not None:
            logger.debug(f"RSI cache hit for key: {rsi_cache_key}")
            df[C.INDICATOR_RSI] = cached_rsi
        else:
            logger.debug(f"RSI cache miss for key: {rsi_cache_key}")
            rsi_series = ta.rsi(close_prices, length=rsi_period)
            if rsi_series is not None:
                df[C.INDICATOR_RSI] = rsi_series
                cache_manager.set(rsi_cache_key, rsi_series)
        
        # Calculate Stochastic Oscillator
        # Calculate Stochastic Oscillator
        stoch_k_period = int(config.get('stoch_k_period', 14))
        stoch_d_period = int(config.get('stoch_d_period', 3))
        stoch_k_slowing = int(config.get('stoch_k_slowing', 3))

        hlc_hash = (pd.util.hash_pandas_object(high_prices).sum() +
                    pd.util.hash_pandas_object(low_prices).sum() +
                    pd.util.hash_pandas_object(close_prices).sum())

        stoch_cache_key = f"stoch:{hlc_hash}:{stoch_k_period}:{stoch_d_period}:{stoch_k_slowing}"
        cached_stoch_data = cache_manager.get(stoch_cache_key)

        if cached_stoch_data is not None:
            logger.debug(f"Stochastic cache hit for key: {stoch_cache_key}")
            df[C.INDICATOR_STOCH_K if hasattr(C, 'INDICATOR_STOCH_K') else 'Stoch_%K'] = cached_stoch_data['STOCHk']
            df[C.INDICATOR_STOCH_D if hasattr(C, 'INDICATOR_STOCH_D') else 'Stoch_%D'] = cached_stoch_data['STOCHd']
        else:
            logger.debug(f"Stochastic cache miss for key: {stoch_cache_key}")
            stoch_df = ta.stoch(
                high=high_prices,
                low=low_prices,
                close=close_prices,
                k=stoch_k_period,
                d=stoch_d_period,
                smooth_k=stoch_k_slowing,
                append=False
            )
            if stoch_df is not None and not stoch_df.empty:
                stoch_k_col = f'STOCHk_{stoch_k_period}_{stoch_d_period}_{stoch_k_slowing}' # pandas-ta < 0.3.15 uses k,d,smooth_k; >= uses k,d,smooth_k
                stoch_d_col = f'STOCHd_{stoch_k_period}_{stoch_d_period}_{stoch_k_slowing}'
                # Need to check exact column names from pandas_ta for STOCH, they can be tricky.
                # It might be STOCHk_14_3_3 and STOCHd_14_3_3 if d is passed to smooth_k in older versions.
                # For recent versions, smooth_k is explicit. The example used smooth_k=k_slowing, d=d_period.
                # Let's assume the column names are based on k, d, and smooth_k (which is the third param in ta.stoch call)
                # For ta.stoch(k=14, d=3, smooth_k=3), cols are STOCHk_14_3_3, STOCHd_14_3_3
                stoch_k_col_actual = f'STOCHk_{stoch_k_period}_{stoch_d_period}_{stoch_k_slowing}'
                stoch_d_col_actual = f'STOCHd_{stoch_k_period}_{stoch_d_period}_{stoch_k_slowing}'
                if stoch_k_col_actual not in stoch_df.columns: # Try alternative common naming from older pandas_ta
                    stoch_k_col_actual = f'STOCHk_{stoch_k_period}_{stoch_k_slowing}_{stoch_d_period}'
                    stoch_d_col_actual = f'STOCHd_{stoch_k_period}_{stoch_k_slowing}_{stoch_d_period}'

                if stoch_k_col_actual in stoch_df.columns and stoch_d_col_actual in stoch_df.columns:
                    df[C.INDICATOR_STOCH_K if hasattr(C, 'INDICATOR_STOCH_K') else 'Stoch_%K'] = stoch_df[stoch_k_col_actual]
                    df[C.INDICATOR_STOCH_D if hasattr(C, 'INDICATOR_STOCH_D') else 'Stoch_%D'] = stoch_df[stoch_d_col_actual]
                    cache_manager.set(stoch_cache_key, {
                        'STOCHk': df[C.INDICATOR_STOCH_K if hasattr(C, 'INDICATOR_STOCH_K') else 'Stoch_%K'],
                        'STOCHd': df[C.INDICATOR_STOCH_D if hasattr(C, 'INDICATOR_STOCH_D') else 'Stoch_%D']
                    })
                else:
                    logger.warning(f"Stochastic column names not found in expected format. Available: {stoch_df.columns.tolist()}")


class VolatilityIndicators(IndicatorGroup):
    """Group for volatility indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        close_prices = df[C.INDICATOR_CLOSE_PRICE if hasattr(C, 'INDICATOR_CLOSE_PRICE') else 'close']
        high_prices = df[C.INDICATOR_HIGH_PRICE if hasattr(C, 'INDICATOR_HIGH_PRICE') else 'high']
        low_prices = df[C.INDICATOR_LOW_PRICE if hasattr(C, 'INDICATOR_LOW_PRICE') else 'low']

        # Calculate ATR
        atr_period_val = int(config.get(C.CONFIG_INDICATOR_ATR_PERIOD, 14))
        hlc_hash = (pd.util.hash_pandas_object(high_prices).sum() +
                    pd.util.hash_pandas_object(low_prices).sum() +
                    pd.util.hash_pandas_object(close_prices).sum())
        atr_cache_key = f"atr:{hlc_hash}:{atr_period_val}"
        cached_atr = cache_manager.get(atr_cache_key)

        if cached_atr is not None:
            df[C.INDICATOR_ATR] = cached_atr
        else:
            atr_series = ta.atr(high=high_prices, low=low_prices, close=close_prices, length=atr_period_val)
            if atr_series is not None:
                df[C.INDICATOR_ATR] = atr_series
                cache_manager.set(atr_cache_key, atr_series)
        
        # Calculate Bollinger Bands
        bb_length_val = int(config.get('bb_length', 20)) # Add constants for bb_length, bb_std
        bb_std_val = float(config.get('bb_std', 2.0))
        
        bb_cache_key = f"bbands:{pd.util.hash_pandas_object(close_prices).sum()}:{bb_length_val}:{bb_std_val}"
        cached_bb_data = cache_manager.get(bb_cache_key)

        if cached_bb_data is not None:
            df[C.INDICATOR_BB_UPPER if hasattr(C,'INDICATOR_BB_UPPER') else 'BB_Upper'] = cached_bb_data['BBU']
            df[C.INDICATOR_BB_MIDDLE if hasattr(C,'INDICATOR_BB_MIDDLE') else 'BB_Middle'] = cached_bb_data['BBM']
            df[C.INDICATOR_BB_LOWER if hasattr(C,'INDICATOR_BB_LOWER') else 'BB_Lower'] = cached_bb_data['BBL']
        else:
            bb_df = ta.bbands(close=close_prices, length=bb_length_val, std=bb_std_val, append=False)
            if bb_df is not None and not bb_df.empty:
                upper_col = f'BBU_{bb_length_val}_{bb_std_val:.1f}'
                middle_col = f'BBM_{bb_length_val}_{bb_std_val:.1f}'
                lower_col = f'BBL_{bb_length_val}_{bb_std_val:.1f}'

                df[C.INDICATOR_BB_UPPER if hasattr(C,'INDICATOR_BB_UPPER') else 'BB_Upper'] = bb_df[upper_col]
                df[C.INDICATOR_BB_MIDDLE if hasattr(C,'INDICATOR_BB_MIDDLE') else 'BB_Middle'] = bb_df[middle_col]
                df[C.INDICATOR_BB_LOWER if hasattr(C,'INDICATOR_BB_LOWER') else 'BB_Lower'] = bb_df[lower_col]
                cache_manager.set(bb_cache_key, {
                    'BBU': df[C.INDICATOR_BB_UPPER if hasattr(C,'INDICATOR_BB_UPPER') else 'BB_Upper'],
                    'BBM': df[C.INDICATOR_BB_MIDDLE if hasattr(C,'INDICATOR_BB_MIDDLE') else 'BB_Middle'],
                    'BBL': df[C.INDICATOR_BB_LOWER if hasattr(C,'INDICATOR_BB_LOWER') else 'BB_Lower']
                })

class VolumeIndicators(IndicatorGroup):
    """Group for volume indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate volume-based indicators.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'close', 'high', 'low', 'volume' columns)
            config: Configuration dictionary with volume indicator parameters
        """
        volume_col_name = C.INDICATOR_VOLUME if hasattr(C, 'INDICATOR_VOLUME') else 'volume'
        if volume_col_name not in df.columns:
            logger.warning(f"Volume column ('{volume_col_name}') not found. Skipping volume indicators.")
            return

        close_prices = df[C.INDICATOR_CLOSE_PRICE if hasattr(C, 'INDICATOR_CLOSE_PRICE') else 'close']
        volume_series = df[volume_col_name]

        # Calculate OBV (On-Balance Volume)
        obv_cache_key = f"obv:{pd.util.hash_pandas_object(close_prices).sum()}:{pd.util.hash_pandas_object(volume_series).sum()}"
        cached_obv = cache_manager.get(obv_cache_key)
        obv_col_const = C.INDICATOR_OBV if hasattr(C, 'INDICATOR_OBV') else 'OBV'

        if cached_obv is not None:
            df[obv_col_const] = cached_obv
        else:
            obv_series = ta.obv(close=close_prices, volume=volume_series)
            if obv_series is not None:
                df[obv_col_const] = obv_series
                cache_manager.set(obv_cache_key, obv_series)
        
        # Calculate VWAP (Volume Weighted Average Price)
        # VWAP is typically calculated per-day or other period, pandas_ta might do it on the whole series.
        # For caching, ensure high, low, close, volume are part of the key.
        high_prices = df[C.INDICATOR_HIGH_PRICE if hasattr(C, 'INDICATOR_HIGH_PRICE') else 'high']
        low_prices = df[C.INDICATOR_LOW_PRICE if hasattr(C, 'INDICATOR_LOW_PRICE') else 'low']

        hlcv_hash = (pd.util.hash_pandas_object(high_prices).sum() +
                     pd.util.hash_pandas_object(low_prices).sum() +
                     pd.util.hash_pandas_object(close_prices).sum() +
                     pd.util.hash_pandas_object(volume_series).sum())

        vwap_cache_key = f"vwap:{hlcv_hash}" # VWAP usually doesn't have a 'period' like other indicators
        cached_vwap = cache_manager.get(vwap_cache_key)
        vwap_col_const = C.INDICATOR_VWAP if hasattr(C, 'INDICATOR_VWAP') else 'VWAP'

        if cached_vwap is not None:
            df[vwap_col_const] = cached_vwap
        else:
            # Check if VWAP already exists (e.g. from data source) before calculating
            if vwap_col_const not in df.columns:
                vwap_series = ta.vwap(high=high_prices, low=low_prices, close=close_prices, volume=volume_series)
                if vwap_series is not None:
                    df[vwap_col_const] = vwap_series
                    cache_manager.set(vwap_cache_key, vwap_series)

class IchimokuIndicators(IndicatorGroup):
    """Group for Ichimoku Cloud indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'high', 'low' columns)
            config: Configuration dictionary with Ichimoku parameters
        """
        high_prices = df[C.INDICATOR_HIGH_PRICE if hasattr(C, 'INDICATOR_HIGH_PRICE') else 'high']
        low_prices = df[C.INDICATOR_LOW_PRICE if hasattr(C, 'INDICATOR_LOW_PRICE') else 'low']

        tenkan_val = int(config.get('ichimoku_tenkan', 9)) # Add constants for these
        kijun_val = int(config.get('ichimoku_kijun', 26))
        senkou_b_val = int(config.get('ichimoku_senkou_b', 52)) # senkou_span_b in ta
        # chikou_shift_val = int(config.get('ichimoku_chikou_shift', 26)) # chikou is part of senkou_b in ta call

        hl_hash = (pd.util.hash_pandas_object(high_prices).sum() +
                   pd.util.hash_pandas_object(low_prices).sum())
        
        ichimoku_cache_key = f"ichimoku:{hl_hash}:{tenkan_val}:{kijun_val}:{senkou_b_val}"
        cached_ichimoku_data = cache_manager.get(ichimoku_cache_key)

        # Define column name constants or use dynamic names from pandas_ta
        # Example: C.ICHIMOKU_TENKAN, C.ICHIMOKU_KIJUN, C.ICHIMOKU_SENKOU_A, C.ICHIMOKU_SENKOU_B, C.ICHIMOKU_CHIKOU
        # Using string literals for now as pandas_ta column names are specific and include parameters.
        # Example: 'ITS_9', 'IKS_26', 'ISA_9_26', 'ISB_26_52', 'ICS_26'
        # The default column names from pandas_ta.ichimoku are:
        # Tenkan-sen: ITS_tenkan_period (e.g. ITS_9)
        # Kijun-sen: IKS_kijun_period (e.g. IKS_26)
        # Senkou Span A: ISA_kijun_period (e.g. ISA_26) - this seems off, usually ISA is (tenkan+kijun)/2 shifted
        # Senkou Span B: ISB_senkou_period (e.g. ISB_52)
        # Chikou Span: ICS_chikou_period (e.g. ICS_26)
        # The pandas_ta library's ichimoku function returns a tuple of two DataFrames:
        # The first contains ITS, IKS, ISA, ISB. The second contains ICS (Chikou Span).
        # Let's assume we use append=False and handle column names.

        if cached_ichimoku_data is not None:
            logger.debug(f"Ichimoku cache hit for key: {ichimoku_cache_key}")
            for col_name, series_data in cached_ichimoku_data.items():
                df[col_name] = series_data
        else:
            logger.debug(f"Ichimoku cache miss for key: {ichimoku_cache_key}")
            # Note: pandas_ta's ichimoku has its own way of naming columns.
            # Default names are like: SPANA_9_26_52, SPANB_9_26_52, TENKAN_9, KIJUN_26, CHIKOU_26
            # The 'ichimoku' function in pandas_ta returns a tuple (df_lines, df_chikou) when include_chikou=True
            # For simplicity, we'll use the direct call and then select columns.
            # The arguments for ta.ichimoku are high, low, close, tenkan, kijun, senkou_b.
            # It seems 'close' is not used by ta.ichimoku, only high and low for cloud calculation.

            # The ta.ichimoku function in pandas_ta takes: high, low, close, tenkan, kijun, senkou, chikou
            # For version 0.3.14b0, it's high, low, tenkan, kijun, senkou_b. Let's use this.
            # And include_chikou is not a param for the main ichimoku, but for the strategy version.
            # The direct ta.ichimoku returns a DataFrame.

            ichimoku_df = ta.ichimoku(high=high_prices, low=low_prices, tenkan=tenkan_val, kijun=kijun_val, senkou_span_b=senkou_b_val, append=False)

            if ichimoku_df is not None and isinstance(ichimoku_df, pd.DataFrame) and not ichimoku_df.empty:
                # Expected columns from pandas_ta.ichimoku (version dependent, e.g. 0.3.14b0):
                # ISA_tenkan_kijun, ISB_kijun_senkou_b, ITS_tenkan, IKS_kijun, ICS_close_chikou
                # For example: ISA_9_26, ISB_26_52, ITS_9, IKS_26, ICS_26 (chikou is close shifted by chikou_period)
                # We need to be careful about exact column names returned by ta.ichimoku
                # For now, let's assume some common constants for our internal use

                # These are EXAMPLE internal names, actual mapping to ta output is crucial.
                # For example, ta.ichimoku might return 'ITS_9', 'IKS_26', 'ISA_9_26', 'ISB_26_52'
                # and Chikou might be calculated separately or part of a strategy version of Ichimoku.
                # The ta.ichimoku function itself (not the strategy) returns columns like:
                # TENKAN_period, KIJUN_period, SENKOU_A_period, SENKOU_B_period, CHIKOU_period

                # Let's assume these are the columns we want to store internally:
                # C.ICHIMOKU_TENKAN, C.ICHIMOKU_KIJUN, C.ICHIMOKU_SENKOU_A, C.ICHIMOKU_SENKOU_B, C.ICHIMOKU_CHIKOU

                # Example mapping (THIS NEEDS VERIFICATION AGAINST PANDAS_TA VERSION)
                # This is highly dependent on the version of pandas_ta and its column naming.
                # For now, this part is illustrative of caching structure rather than exact ta mapping.

                # Example: if ta.ichimoku() returns columns like 'ITS_9', 'IKS_26', etc.
                # We would map them: df[C.ICHIMOKU_TENKAN] = ichimoku_df[f'ITS_{tenkan_val}']
                # This section needs to be robust to pandas_ta's output.

                # Simplified: cache the whole returned ichimoku_df and let users pick columns.
                # However, this is not granular.
                # For now, I will not implement full Ichimoku caching due to column name complexity with pandas_ta versions.
                # This indicates that for complex multi-output indicators, caching strategy needs careful thought.
                # A placeholder for actual column assignment after resolving pandas_ta specifics:
                # df['Ichimoku_Tenkan'] = ichimoku_df[<actual_tenkan_col_name_from_ta>]
                # ... and so on for Kijun, SenkouA, SenkouB, Chikou ...
                # cache_manager.set(ichimoku_cache_key, {col: df[col] for col in ichimoku_cols_to_cache})
                logger.warning("Ichimoku caching not fully implemented due to pandas_ta column name variance. Calculation will run each time.")
                # Actual calculation (if not cached)
                if ichimoku_df is not None:
                     for col in ichimoku_df.columns: # Add all columns returned by ta.ichimoku
                         if col not in df.columns: # Avoid overwriting if already exists (e.g. from data source)
                             df[col] = ichimoku_df[col]


class IndicatorCalculator:
    """Handles calculation of technical indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the indicator calculator.
        
        Args:
            config: Configuration dictionary containing indicator parameters
        """
        # Initialize indicator groups
        self.indicator_groups = {
            'trend': TrendIndicators(),
            'momentum': MomentumIndicators(),
            'volatility': VolatilityIndicators(),
            'volume': VolumeIndicators(),
            'ichimoku': IchimokuIndicators(),
        }
    
    def calculate_all(
        self, 
        df: pd.DataFrame, 
        indicator_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate all indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            indicator_config: Optional configuration override
            
        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df
            
        # Log warning if indicator configuration not provided
        if indicator_config is None:
            logger.warning("Indicator configuration not provided to calculate_all. Using empty config.")
            indicator_config = {}
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Normalize column names to lowercase if they are not already
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate indicators by group
        for group_name, group in self.indicator_groups.items():
            # Skip disabled groups
            if not indicator_config.get(f'enable_{group_name}', True):
                continue
                
            try:
                group.calculate(df, indicator_config)
            except Exception as e:
                logger.error(f"Error calculating {group_name} indicators: {str(e)}")
        
        return df
    
    # Delegating methods to maintain backward compatibility
    
    def calculate_macd(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate MACD indicator."""
        self.indicator_groups['trend'].calculate(df, config or {})
    
    def calculate_rsi(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate RSI indicator."""
        self.indicator_groups['momentum'].calculate(df, config or {})
    
    def calculate_atr(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate ATR indicator."""
        self.indicator_groups['volatility'].calculate(df, config or {})
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate Bollinger Bands."""
        self.indicator_groups['volatility'].calculate(df, config or {})
    
    def calculate_stochastic(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate Stochastic Oscillator."""
        self.indicator_groups['momentum'].calculate(df, config or {})
    
    def calculate_ichimoku(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate Ichimoku Cloud."""
        self.indicator_groups['ichimoku'].calculate(df, config or {})
    
    def calculate_adx(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate ADX (Average Directional Index)."""
        self.indicator_groups['trend'].calculate(df, config or {})
    
    def calculate_volume_indicators(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> None:
        """Calculate volume-based indicators."""
        self.indicator_groups['volume'].calculate(df, config or {})
