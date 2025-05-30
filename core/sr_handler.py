"""
Module for identifying and managing Support/Resistance levels.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
import logging
from dataclasses import dataclass
import abc
from enum import Enum, auto

logger = logging.getLogger(__name__)

@dataclass
class SRLevel:
    """Class to represent a Support/Resistance level."""
    price: float
    strength: float  # 0-1 scale indicating strength/confidence
    type: str  # 'support' or 'resistance'
    method: str  # Method used to identify the level (pivot, fractal, etc.)
    timestamp: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict:
        """Convert SRLevel to dictionary."""
        return {
            'price': self.price,
            'strength': self.strength,
            'type': self.type,
            'method': self.method,
            'timestamp': self.timestamp
        }

class SRMethodType(Enum):
    """Enum for SR calculation methods."""
    PIVOTS = auto()
    FRACTALS = auto()
    ZIGZAG = auto()
    AUTO = auto()

class SRCalculator(abc.ABC):
    """Abstract base class for SR calculation strategies."""
    
    @abc.abstractmethod
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[SRLevel]:
        """
        Calculate support/resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary
            
        Returns:
            List of SRLevel objects
        """
        pass

class PivotCalculator(SRCalculator):
    """Calculate SR levels using pivot points."""
    
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[SRLevel]:
        """
        Calculate pivot point levels.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary
            
        Returns:
            List of SRLevel objects
        """
        if df.empty:
            return []
            
        pivot_config = config.get('pivots', {})
        pivot_type = pivot_config.get('type', 'standard')
        
        # Get the most recent completed period's OHLC
        last_bar = df.iloc[-1]
        h = last_bar['High']
        l = last_bar['Low']
        c = last_bar['Close']
        
        # Calculate pivot point
        if pivot_type == 'standard':
            p = (h + l + c) / 3
            r1 = 2 * p - l
            s1 = 2 * p - h
            r2 = p + (h - l)
            s2 = p - (h - l)
            r3 = h + 2 * (p - l)
            s3 = l - 2 * (h - p)
        elif pivot_type == 'fibonacci':
            p = (h + l + c) / 3
            r1 = p + 0.382 * (h - l)
            r2 = p + 0.618 * (h - l)
            r3 = p + 1.0 * (h - l)
            s1 = p - 0.382 * (h - l)
            s2 = p - 0.618 * (h - l)
            s3 = p - 1.0 * (h - l)
        else:  # woodie, camarilla, etc.
            # Default to standard if unknown type
            p = (h + l + c) / 3
            r1 = 2 * p - l
            s1 = 2 * p - h
            r2 = p + (h - l)
            s2 = p - (h - l)
            r3 = h + 2 * (p - l)
            s3 = l - 2 * (h - p)
        
        # Create SRLevel objects
        levels = [
            SRLevel(price=p, strength=1.0, type='pivot', method='pivot'),
            SRLevel(price=r1, strength=0.8, type='resistance', method='pivot'),
            SRLevel(price=s1, strength=0.8, type='support', method='pivot'),
            SRLevel(price=r2, strength=0.6, type='resistance', method='pivot'),
            SRLevel(price=s2, strength=0.6, type='support', method='pivot'),
            SRLevel(price=r3, strength=0.4, type='resistance', method='pivot'),
            SRLevel(price=s3, strength=0.4, type='support', method='pivot'),
        ]
        
        return levels

class FractalCalculator(SRCalculator):
    """Calculate SR levels using fractals."""
    
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[SRLevel]:
        """
        Calculate support/resistance levels using fractals.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary
            
        Returns:
            List of SRLevel objects
        """
        if df.empty:
            return []
            
        fractal_config = config.get('fractals', {})
        window = fractal_config.get('window', 2)
        
        # Calculate fractals
        df = df.copy()
        
        # Get bearish and bullish fractals
        bearish_fractals = ta.fractals(df['High'], df['Low'], asbool=False)
        bullish_fractals = ta.fractals(df['High'], df['Low'], asbool=False)
        
        # Get the most recent fractals
        recent_bearish = bearish_fractals[bearish_fractals == 1].tail(5)
        recent_bullish = bullish_fractals[bullish_fractals == 1].tail(5)
        
        levels = []
        
        # Add resistance levels from bearish fractals (highs)
        for idx, val in recent_bearish.items():
            price = df.loc[idx, 'High']
            levels.append(SRLevel(
                price=price,
                strength=0.9,
                type='resistance',
                method='fractal',
                timestamp=idx
            ))
        
        # Add support levels from bullish fractals (lows)
        for idx, val in recent_bullish.items():
            price = df.loc[idx, 'Low']
            levels.append(SRLevel(
                price=price,
                strength=0.9,
                type='support',
                method='fractal',
                timestamp=idx
            ))
        
        # Merge nearby levels (delegamos al SRHandler)
        return levels

class ZigZagCalculator(SRCalculator):
    """Calculate SR levels using ZigZag indicator."""
    
    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[SRLevel]:
        """
        Calculate support/resistance levels using ZigZag indicator.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary
            
        Returns:
            List of SRLevel objects
        """
        if df.empty:
            return []
            
        zigzag_config = config.get('zigzag', {})
        depth = zigzag_config.get('depth', 12)
        deviation = zigzag_config.get('deviation', 5)
        backstep = zigzag_config.get('backstep', 3)
        
        # Calculate ZigZag
        zigzag = ta.zigzag(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            depth=depth,
            deviation=deviation,
            backstep=backstep,
            append=False
        )
        
        if zigzag is None or 'ZIGZAG' not in zigzag.columns:
            return []
        
        # Get swing points
        swing_points = df[zigzag['ZIGZAG'].notna()]
        
        levels = []
        
        for idx, row in swing_points.iterrows():
            if not np.isnan(row['ZIGZAG']):
                # Determine if it's a high or low
                if row['High'] == row['ZIGZAG']:  # Swing high (resistance)
                    levels.append(SRLevel(
                        price=row['High'],
                        strength=0.85,
                        type='resistance',
                        method='zigzag',
                        timestamp=idx
                    ))
                else:  # Swing low (support)
                    levels.append(SRLevel(
                        price=row['Low'],
                        strength=0.85,
                        type='support',
                        method='zigzag',
                        timestamp=idx
                    ))
        
        # Merge nearby levels (delegamos al SRHandler)
        return levels

class SRHandler:
    """Handles identification and management of Support/Resistance levels."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SR handler.
        
        Args:
            config: Configuration dictionary containing SR parameters
        """
        self.config = config
        self.levels: List[SRLevel] = []
        
        # Initialize calculators using Strategy pattern
        self._calculators = {
            SRMethodType.PIVOTS: PivotCalculator(),
            SRMethodType.FRACTALS: FractalCalculator(),
            SRMethodType.ZIGZAG: ZigZagCalculator(),
        }
    
    def get_sr_levels(
        self, 
        df: pd.DataFrame, 
        method: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> List[SRLevel]:
        """
        Get Support/Resistance levels using the specified method.
        
        Args:
            df: DataFrame with OHLCV data
            method: Method to use ('pivots', 'fractals', 'zigzag', or 'auto')
            config: Optional configuration override
            
        Returns:
            List of SRLevel objects
        """
        if df.empty:
            return []
            
        config = config or self.config
        method_str = method or config.get('method', 'auto')
        
        # Convert string method to enum
        try:
            method_type = SRMethodType[method_str.upper()]
        except KeyError:
            method_type = SRMethodType.AUTO
        
        # Handle AUTO method
        if method_type == SRMethodType.AUTO:
            # Try to determine the best method based on data
            if len(df) >= 100:  # Need sufficient data for fractals/zigzag
                method_type = SRMethodType.FRACTALS
            else:
                method_type = SRMethodType.PIVOTS
        
        # Get calculator and calculate levels
        calculator = self._calculators.get(method_type, self._calculators[SRMethodType.PIVOTS])
        levels = calculator.calculate(df, config)
        
        # Process levels (merge, filter, etc.)
        threshold = df['Close'].iloc[-1] * 0.005  # 0.5% threshold
        levels = self._merge_nearby_levels(levels, threshold)
        
        return levels
    
    def _merge_nearby_levels(
        self, 
        levels: List[SRLevel], 
        threshold: float
    ) -> List[SRLevel]:
        """
        Merge nearby support/resistance levels.
        
        Args:
            levels: List of SRLevel objects
            threshold: Price difference threshold for merging
            
        Returns:
            List of merged SRLevel objects
        """
        if not levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        
        merged = []
        current = sorted_levels[0]
        
        for level in sorted_levels[1:]:
            if abs(level.price - current.price) <= threshold:
                # Merge levels (weighted average by strength)
                total_strength = current.strength + level.strength
                new_price = (current.price * current.strength + level.price * level.strength) / total_strength
                
                # If types match, keep it; otherwise, use the stronger one
                if current.type == level.type:
                    new_type = current.type
                else:
                    new_type = current.type if current.strength >= level.strength else level.type
                
                current = SRLevel(
                    price=new_price,
                    strength=total_strength / 2,  # Average strength
                    type=new_type,
                    method=f"{current.method}+{level.method}"
                )
            else:
                merged.append(current)
                current = level
        
        # Add the last level
        merged.append(current)
        
        return merged
    
    def filter_relevant_levels(
        self, 
        levels: List[SRLevel], 
        current_price: float, 
        distance_pct: float = 5.0
    ) -> Tuple[List[SRLevel], List[SRLevel]]:
        """
        Filter support/resistance levels to only those near the current price.
        
        Args:
            levels: List of SRLevel objects
            current_price: Current price
            distance_pct: Maximum distance percentage from current price
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if not levels:
            return [], []
        
        distance = current_price * (distance_pct / 100.0)
        
        support = []
        resistance = []
        
        for level in levels:
            if abs(level.price - current_price) <= distance:
                if level.type == 'support':
                    support.append(level)
                else:  # resistance
                    resistance.append(level)
        
        # Sort support in descending order (closest to price first)
        support.sort(key=lambda x: x.price, reverse=True)
        
        # Sort resistance in ascending order (closest to price first)
        resistance.sort(key=lambda x: x.price)
        
        return support, resistance
    
    def validate_sr_level(
        self, 
        level: float, 
        price_series: pd.Series, 
        tolerance_pct: float = 0.5
    ) -> float:
        """
        Validate a support/resistance level by checking price reactions.
        
        Args:
            level: The price level to validate
            price_series: Series of prices (e.g., High for resistance, Low for support)
            tolerance_pct: Percentage tolerance for level validation
            
        Returns:
            Strength score (0-1) indicating how strong the level is
        """
        if price_series.empty:
            return 0.0
        
        tolerance = level * (tolerance_pct / 100.0)
        
        # Count touches (price comes within tolerance of the level)
        touches = ((price_series >= level - tolerance) & 
                  (price_series <= level + tolerance)).sum()
        
        # Calculate strength based on number of touches (capped at 10)
        strength = min(touches / 10.0, 1.0)
        
        return strength
