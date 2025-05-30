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
            df: DataFrame with OHLCV data
            config: Configuration dictionary with trend indicator parameters
        """
        # Calculate MACD
        fast = int(config.get('macd_fast', 12))
        slow = int(config.get('macd_slow', 26))
        signal = int(config.get('macd_signal', 9))
        
        macd = ta.macd(
            df['Close'], 
            fast=fast, 
            slow=slow, 
            signal=signal,
            append=False
        )
        
        if macd is not None:
            df['MACD'] = macd[f'MACD_{fast}_{slow}_{signal}']
            df['MACD_Signal'] = macd[f'MACDs_{fast}_{slow}_{signal}']
            df['MACD_Hist'] = macd[f'MACDh_{fast}_{slow}_{signal}']
        
        # Calculate Moving Averages
        for ma_type in ['sma', 'ema']:
            for period in [20, 50, 100, 200]:
                param_name = f'{ma_type}_{period}'
                if config.get(param_name, True):  # Default to True if not specified
                    if ma_type == 'sma':
                        ma = ta.sma(df['Close'], length=period)
                    else:  # ema
                        ma = ta.ema(df['Close'], length=period)
                    
                    if ma is not None:
                        df[param_name] = ma
        
        # Calculate ADX
        adx_period = int(config.get('adx_period', 14))
        adx = ta.adx(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            length=adx_period,
            append=False
        )
        
        if adx is not None:
            df['ADX'] = adx[f'ADX_{adx_period}']
            df['DMP'] = adx[f'DMP_{adx_period}']
            df['DMN'] = adx[f'DMN_{adx_period}']

class MomentumIndicators(IndicatorGroup):
    """Group for momentum indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with momentum indicator parameters
        """
        # Calculate RSI
        rsi_period = int(config.get('rsi_period', 14))
        rsi = ta.rsi(df['Close'], length=rsi_period)
        
        if rsi is not None:
            df['RSI'] = rsi
        
        # Calculate Stochastic Oscillator
        k_period = int(config.get('stoch_k_period', 14))
        k_slowing = int(config.get('stoch_k_slowing', 3))
        d_period = int(config.get('stoch_d_period', 3))
        
        stoch = ta.stoch(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            k=k_period,
            d=d_period,
            smooth_k=k_slowing,
            append=False
        )
        
        if stoch is not None:
            df['Stoch_%K'] = stoch[f'STOCHk_{k_period}_{k_slowing}_{d_period}']
            df['Stoch_%D'] = stoch[f'STOCHd_{k_period}_{k_slowing}_{d_period}']

class VolatilityIndicators(IndicatorGroup):
    """Group for volatility indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate volatility indicators.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with volatility indicator parameters
        """
        # Calculate ATR
        atr_period = int(config.get('atr_period', 14))
        atr = ta.atr(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            length=atr_period
        )
        
        if atr is not None:
            df['ATR'] = atr
        
        # Calculate Bollinger Bands
        bb_length = int(config.get('bb_length', 20))
        bb_std = float(config.get('bb_std', 2.0))
        
        bb = ta.bbands(
            close=df['Close'],
            length=bb_length,
            std=bb_std,
            append=False
        )
        
        if bb is not None:
            df['BB_Upper'] = bb[f'BBU_{bb_length}_{bb_std}']
            df['BB_Middle'] = bb[f'BBM_{bb_length}_{bb_std}']
            df['BB_Lower'] = bb[f'BBL_{bb_length}_{bb_std}']

class VolumeIndicators(IndicatorGroup):
    """Group for volume indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate volume-based indicators.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with volume indicator parameters
        """
        # Skip if volume data is not available
        if 'Volume' not in df.columns:
            return
        
        # Calculate OBV (On-Balance Volume)
        df['OBV'] = ta.obv(close=df['Close'], volume=df['Volume'])
        
        # Calculate VWAP (Volume Weighted Average Price)
        if 'VWAP' not in df.columns:
            df['VWAP'] = ta.vwap(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume']
            )

class IchimokuIndicators(IndicatorGroup):
    """Group for Ichimoku Cloud indicators."""
    
    @handle_empty_df
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with Ichimoku parameters
        """
        # Get parameters with defaults
        tenkan = int(config.get('ichimoku_tenkan', 9))
        kijun = int(config.get('ichimoku_kijun', 26))
        senkou_span_b = int(config.get('ichimoku_senkou_span_b', 52))
        chikou_shift = int(config.get('ichimoku_chikou_shift', 26))
        
        # Calculate Ichimoku using pandas_ta
        ichimoku = ta.ichimoku(
            high=df['High'],
            low=df['Low'],
            tenkan=tenkan,
            kijun=kijun,
            senkou_span_b=senkou_span_b,
            chikou_shift=chikou_shift,
            include_chikou=True,
            append=False
        )
        
        # Add to DataFrame
        if ichimoku is not None:
            # Ichimoku returns a tuple of DataFrames
            # First DataFrame has the main lines
            # Second DataFrame has the cloud (Senkou Span A/B)
            if len(ichimoku) >= 1 and ichimoku[0] is not None:
                df['Ichimoku_Conversion'] = ichimoku[0]['ITS_9']
                df['Ichimoku_Base'] = ichimoku[0]['IKS_26']
                df['Ichimoku_Leading_Span_A'] = ichimoku[0]['ISA_9']
                df['Ichimoku_Leading_Span_B'] = ichimoku[0]['ISB_26']
                df['Ichimoku_Lagging_Span'] = ichimoku[0]['ICS_26']

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
