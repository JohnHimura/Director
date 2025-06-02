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
        close_prices = df['close'] if 'close' in df.columns else df['Close']

        # Calculate MACD
        fast = int(config.get(C.CONFIG_INDICATOR_MACD_FAST, 12))
        slow = int(config.get(C.CONFIG_INDICATOR_MACD_SLOW, 26))
        signal_period = int(config.get(C.CONFIG_INDICATOR_MACD_SIGNAL, 9)) # Renamed signal to signal_period

        # Cache key for MACD - usar función hash para evitar overflow
        close_hash = hash(pd.util.hash_pandas_object(close_prices).sum())
        macd_cache_key = f"macd:{close_hash}:{fast}:{slow}:{signal_period}"
        cached_macd_data = cache_manager.get(macd_cache_key)

        if cached_macd_data is not None:
            logger.debug(f"MACD cache hit for key: {macd_cache_key}")
            df['macd'] = cached_macd_data['macd']
            df['macd_signal'] = cached_macd_data['macd_signal']
            # df['macd_histogram'] = cached_macd_data['macd_histogram'] # If histogram is needed
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

                df['macd'] = macd_df[macd_col_name]
                df['macd_signal'] = macd_df[signal_col_name]
                # df['macd_histogram'] = macd_df[hist_col_name]

                # Store necessary series in cache
                cache_manager.set(macd_cache_key, {
                    'macd': df['macd'],
                    'macd_signal': df['macd_signal'],
                    # 'macd_histogram': df['macd_histogram']
                })
        
        # Calculate Moving Averages
        for ma_type_str in ['sma', 'ema']: # ma_type_str to avoid conflict
            for period_val in [20, 50, 100, 200]: # period_val to avoid conflict
                ma_col_name = f'{ma_type_str.upper()}_{period_val}' # e.g., EMA_50
                # Use more specific config keys if available, e.g., config.get('enable_ema_50', True)
                if config.get(f'enable_{ma_type_str}{period_val}', True):
                    ma_cache_key = f"{ma_type_str}:{close_hash}:{period_val}"
                    cached_ma = cache_manager.get(ma_cache_key)
                    if cached_ma is not None:
                        df[ma_col_name.lower()] = cached_ma
                    else:
                        if ma_type_str == 'sma':
                            ma_series = ta.sma(close_prices, length=period_val)
                        else: # ema
                            ma_series = ta.ema(close_prices, length=period_val)

                        if ma_series is not None:
                            df[ma_col_name.lower()] = ma_series
                            cache_manager.set(ma_cache_key, ma_series)
        
        # Calculate ADX
        adx_period_val = int(config.get('adx_period', 14)) # adx_period_val
        # ADX uses High, Low, Close. Create a combined hash for these.
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        hlc_hash = hash(pd.util.hash_pandas_object(high_prices).sum()) ^ hash(pd.util.hash_pandas_object(low_prices).sum()) ^ close_hash
        
        adx_cache_key = f"adx:{hlc_hash}:{adx_period_val}"
        cached_adx_data = cache_manager.get(adx_cache_key)

        if cached_adx_data is not None:
            logger.debug(f"ADX cache hit for key: {adx_cache_key}")
            df['adx'] = cached_adx_data['ADX']
            # df['DMP'] = cached_adx_data['DMP'] # Define constants if using these
            # df['DMN'] = cached_adx_data['DMN']
        else:
            logger.debug(f"ADX cache miss for key: {adx_cache_key}")
            adx_df = ta.adx(
                high=high_prices,
                low=low_prices,
                close=close_prices,
                length=adx_period_val,
                append=False
            )
            if adx_df is not None and not adx_df.empty:
                adx_col = f'ADX_{adx_period_val}'
                # dmp_col = f'DMP_{adx_period_val}'
                # dmn_col = f'DMN_{adx_period_val}'
                df['adx'] = adx_df[adx_col]
                # df['DMP'] = adx_df[dmp_col]
                # df['DMN'] = adx_df[dmn_col]
                cache_manager.set(adx_cache_key, {
                    'ADX': df['adx'],
                    # 'DMP': df['DMP'], 'DMN': df['DMN']
                })

class MomentumIndicators(IndicatorGroup):
    """Group for momentum indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'close' column)
            config: Configuration dictionary with momentum indicator parameters
        """
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Calculate RSI
        rsi_period = int(config.get(C.CONFIG_INDICATOR_RSI_PERIOD, 14))
        
        # Hash for caching - usar hash() para evitar overflow
        close_hash = hash(pd.util.hash_pandas_object(close_prices).sum())
        rsi_cache_key = f"rsi:{close_hash}:{rsi_period}"
        cached_rsi = cache_manager.get(rsi_cache_key)
        
        if cached_rsi is not None:
            logger.debug(f"RSI cache hit for key: {rsi_cache_key}")
            df['rsi'] = cached_rsi
        else:
            logger.debug(f"RSI cache miss for key: {rsi_cache_key}")
            rsi_series = ta.rsi(close_prices, length=rsi_period)
            if rsi_series is not None:
                df['rsi'] = rsi_series
                cache_manager.set(rsi_cache_key, rsi_series)
        
        # Calculate Stochastic
        stoch_k_period = int(config.get(C.CONFIG_INDICATOR_STOCH_K_PERIOD, 14))
        stoch_d_period = int(config.get(C.CONFIG_INDICATOR_STOCH_D_PERIOD, 3))
        stoch_smooth_k = int(config.get(C.CONFIG_INDICATOR_STOCH_SMOOTH_K, 3))
        
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        
        # Combine all relevant series into a single hash for stochastic - usar XOR para evitar overflow
        hlc_hash = hash(pd.util.hash_pandas_object(high_prices).sum()) ^ hash(pd.util.hash_pandas_object(low_prices).sum()) ^ close_hash
        
        stoch_cache_key = f"stoch:{hlc_hash}:{stoch_k_period}:{stoch_d_period}:{stoch_smooth_k}"
        cached_stoch_data = cache_manager.get(stoch_cache_key)
        
        if cached_stoch_data is not None:
            logger.debug(f"Stochastic cache hit for key: {stoch_cache_key}")
            df['stoch_%k'] = cached_stoch_data['stoch_%k']
            df['stoch_%d'] = cached_stoch_data['stoch_%d']
        else:
            logger.debug(f"Stochastic cache miss for key: {stoch_cache_key}")
            stoch_df = ta.stoch(
                high=high_prices,
                low=low_prices,
                close=close_prices,
                k=stoch_k_period,
                d=stoch_d_period,
                smooth_k=stoch_smooth_k,
                append=False
            )
            if stoch_df is not None and not stoch_df.empty:
                # Get the correct column names from pandas_ta output
                k_col = f"STOCHk_{stoch_k_period}_{stoch_d_period}_{stoch_smooth_k}"
                d_col = f"STOCHd_{stoch_k_period}_{stoch_d_period}_{stoch_smooth_k}"
                
                df['stoch_%k'] = stoch_df[k_col]
                df['stoch_%d'] = stoch_df[d_col]
                
                # Store in cache
                cache_manager.set(stoch_cache_key, {
                    'stoch_%k': stoch_df[k_col],
                    'stoch_%d': stoch_df[d_col]
                })


class VolatilityIndicators(IndicatorGroup):
    """Group for volatility indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate volatility indicators.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'high', 'low', 'close' columns)
            config: Configuration dictionary with volatility indicator parameters
        """
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        
        # Calculate ATR
        atr_period = int(config.get(C.CONFIG_INDICATOR_ATR_PERIOD, 14))
        
        # Combine high, low, close for ATR hashing - usar XOR para evitar overflow
        hlc_hash = hash(pd.util.hash_pandas_object(high_prices).sum()) ^ hash(pd.util.hash_pandas_object(low_prices).sum()) ^ hash(pd.util.hash_pandas_object(close_prices).sum())
        
        atr_cache_key = f"atr:{hlc_hash}:{atr_period}"
        cached_atr = cache_manager.get(atr_cache_key)
        
        if cached_atr is not None:
            logger.debug(f"ATR cache hit for key: {atr_cache_key}")
            df['atr'] = cached_atr
        else:
            logger.debug(f"ATR cache miss for key: {atr_cache_key}")
            atr_series = ta.atr(high=high_prices, low=low_prices, close=close_prices, length=atr_period)
            if atr_series is not None:
                df['atr'] = atr_series
                cache_manager.set(atr_cache_key, atr_series)
        
        # Calculate Bollinger Bands
        bb_period = int(config.get(C.CONFIG_INDICATOR_BB_PERIOD, 20))
        bb_std = float(config.get(C.CONFIG_INDICATOR_BB_STD_DEV, 2.0))
        
        # Hash for Bollinger Bands - usar hash() para evitar overflow
        close_hash = hash(pd.util.hash_pandas_object(close_prices).sum())
        bb_cache_key = f"bbands:{close_hash}:{bb_period}:{bb_std}"
        cached_bb_data = cache_manager.get(bb_cache_key)
        
        if cached_bb_data is not None:
            logger.debug(f"Bollinger Bands cache hit for key: {bb_cache_key}")
            df['bb_upper'] = cached_bb_data['bb_upper']
            df['bb_middle'] = cached_bb_data['bb_middle']
            df['bb_lower'] = cached_bb_data['bb_lower']
        else:
            logger.debug(f"Bollinger Bands cache miss for key: {bb_cache_key}")
            bbands_df = ta.bbands(close=close_prices, length=bb_period, std=bb_std, append=False)
            if bbands_df is not None and not bbands_df.empty:
                bb_upper_col = f"BBU_{bb_period}_{bb_std}"
                bb_middle_col = f"BBM_{bb_period}_{bb_std}"
                bb_lower_col = f"BBL_{bb_period}_{bb_std}"
                
                df['bb_upper'] = bbands_df[bb_upper_col]
                df['bb_middle'] = bbands_df[bb_middle_col]
                df['bb_lower'] = bbands_df[bb_lower_col]
                
                # Store in cache
                cache_manager.set(bb_cache_key, {
                    'bb_upper': bbands_df[bb_upper_col],
                    'bb_middle': bbands_df[bb_middle_col],
                    'bb_lower': bbands_df[bb_lower_col]
                })

class VolumeIndicators(IndicatorGroup):
    """Group for volume indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate volume indicators.
        
        Args:
            df: DataFrame with OHLCV data (expected to have 'close', 'volume' columns)
            config: Configuration dictionary with volume indicator parameters
        """
        volume_series = df['volume'] if 'volume' in df.columns else df['Volume']
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Calculate OBV
        vol_close_hash = hash(pd.util.hash_pandas_object(volume_series).sum()) ^ hash(pd.util.hash_pandas_object(close_prices).sum())
        obv_cache_key = f"obv:{vol_close_hash}"
        cached_obv = cache_manager.get(obv_cache_key)
        
        if cached_obv is not None:
            logger.debug(f"OBV cache hit for key: {obv_cache_key}")
            df['obv'] = cached_obv
        else:
            logger.debug(f"OBV cache miss for key: {obv_cache_key}")
            obv_series = ta.obv(close=close_prices, volume=volume_series)
            if obv_series is not None:
                df['obv'] = obv_series
                cache_manager.set(obv_cache_key, obv_series)
        
        # Calculate VWAP if needed
        # Note: VWAP typically requires datetime index and is calculated from market open
        # This is a simplified version that works regardless of index type
        if not config.get('calculate_vwap', True):
            return
            
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']

        # Usar XOR para evitar overflow
        hlcv_hash = hash(pd.util.hash_pandas_object(high_prices).sum()) ^ hash(pd.util.hash_pandas_object(low_prices).sum()) ^ hash(pd.util.hash_pandas_object(close_prices).sum()) ^ hash(pd.util.hash_pandas_object(volume_series).sum())

        vwap_cache_key = f"vwap:{hlcv_hash}" # VWAP usually doesn't have a 'period' like other indicators
        cached_vwap = cache_manager.get(vwap_cache_key)

        if cached_vwap is not None:
            df['vwap'] = cached_vwap
        else:
            # Check if VWAP already exists (e.g. from data source) before calculating
            if 'vwap' not in df.columns:
                vwap_series = ta.vwap(high=high_prices, low=low_prices, close=close_prices, volume=volume_series)
                if vwap_series is not None:
                    df['vwap'] = vwap_series
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
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        close_prices = df['close'] if 'close' in df.columns else df['Close']

        tenkan_val = int(config.get('ichimoku_tenkan', 9)) # Add constants for these
        kijun_val = int(config.get('ichimoku_kijun', 26))
        senkou_b_val = int(config.get('ichimoku_senkou_b', 52)) # senkou_span_b in ta
        # chikou_shift_val = int(config.get('ichimoku_chikou_shift', 26)) # chikou is part of senkou_b in ta call

        # Usar XOR para evitar overflow
        hl_hash = hash(pd.util.hash_pandas_object(high_prices).sum()) ^ hash(pd.util.hash_pandas_object(low_prices).sum())
        
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

            ichimoku_df = ta.ichimoku(high=high_prices, low=low_prices, close=close_prices, tenkan=tenkan_val, kijun=kijun_val, senkou_span_b=senkou_b_val, append=False)

            if ichimoku_df is not None and isinstance(ichimoku_df, pd.DataFrame) and not ichimoku_df.empty:
                # Expected columns from pandas_ta.ichimoku (version dependent, e.g. 0.3.14b0):
                # ISA_tenkan_kijun, ISB_kijun_senkou_b, ITS_tenkan, IKS_kijun, ICS_close_chikou
                # For example: ISA_9_26, ISB_26_52, ITS_9, IKS_26, ICS_26 (chikou is close shifted by chikou_period)
                # We need to be careful about exact column names returned by ta.ichimoku
                # For now, let's assume some common constants for our internal use

                # These are EXAMPLE internal names, actual mapping to ta output is crucial.
                # For example, ta.ichimoku might return 'ITS_9', 'IKS_26', etc.
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
