"""
Tests for the StrategyEngine class.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from core.strategy_engine import StrategyEngine, SignalType
from core.config_manager import ConfigManager
from core.sr_handler import SRLevel

# Sample market data for testing
def create_sample_market_data():
    """Create sample market data for testing."""
    # Create a date range
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
    
    # Create sample data for M15 timeframe
    # Ensure all column names are lowercase to match expected format
    m15_data = {
        'open': 100.0 + np.random.randn(100).cumsum() * 0.1,
        'high': 101.0 + np.random.randn(100).cumsum() * 0.1,
        'low': 99.0 + np.random.randn(100).cumsum() * 0.1,
        'close': 100.0 + np.random.randn(100).cumsum() * 0.1,
        'volume': np.random.randint(100, 1000, 100)
    }
    m15_df = pd.DataFrame(m15_data, index=dates)
    
    # Resample for other timeframes
    h1_df = m15_df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    h4_df = m15_df.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    d1_df = m15_df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return {
        'M15': m15_df,
        'H1': h1_df,
        'H4': h4_df,
        'D1': d1_df
    }

@pytest.fixture
def mock_indicator_calculator():
    """Create a mock IndicatorCalculator."""
    mock = MagicMock()
    
    # Configure the mock to return the same dataframe it receives
    def mock_calculate_all(df, *args, **kwargs):
        # Add mock indicator columns to the dataframe
        df['rsi'] = 30.0  # Oversold
        df['macd'] = 1.0
        df['macd_signal'] = 0.9
        df['macd_hist'] = 0.8
        df['atr'] = 0.5
        return df
    
    # Set up the mock to use our function
    mock.calculate_all.side_effect = mock_calculate_all
    
    return mock

@pytest.fixture
def mock_sr_handler():
    """Create a mock SRHandler."""
    mock = MagicMock()
    
    # Create mock SRLevel objects using positional arguments
    support = MagicMock()
    support.price = 95.0
    support.strength = 0.8
    support.level_type = 'support'
    
    resistance = MagicMock()
    resistance.price = 105.0
    resistance.strength = 0.7
    resistance.level_type = 'resistance'
    
    mock.get_sr_levels.return_value = [support, resistance]
    return mock

def test_strategy_engine_initialization():
    """Test StrategyEngine initialization."""
    # Create a mock config
    config = MagicMock(spec=ConfigManager)
    config.get_indicator_params.return_value = {}
    
    # Initialize the strategy engine
    strategy = StrategyEngine(config)
    
    assert strategy is not None
    assert strategy.config == config
    assert strategy.last_signal == SignalType.NONE

@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_buy_signal(mock_indicator_calc_class, mock_indicator_calculator, mock_sr_handler):
    """Test analyze method with buy signal."""
    # Setup mocks
    mock_indicator_calc_class.return_value = mock_indicator_calculator
    
    # Create a mock config with all required methods
    config = MagicMock()
    config.get.return_value = {'primary': 'M15', 'secondary': 'H1'}
    config.get_indicator_params.return_value = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2.0
    }
    
    # Initialize the strategy engine with mocks
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler):
        strategy = StrategyEngine(config)
    
    # Create sample data
    data = create_sample_market_data()
    
    # Test with no open position
    result = strategy.analyze('EURUSD', data)
    
    # Check results
    assert isinstance(result, dict)
    assert 'signal' in result
    
    # If we got a signal, check it's valid
    if result['signal'] != 0:  # 0=No signal, 1=Buy, -1=Sell
        assert result['signal'] in [1, -1]
        assert 'entry_price' in result
        assert 'stop_loss' in result
        assert 'take_profit' in result
    
    # Verify indicator calculator was called
    mock_indicator_calculator.calculate_all.assert_called()
    
    # Verify SR handler was called
    mock_sr_handler.get_sr_levels.assert_called()

@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_with_open_position(mock_indicator_calc_class, mock_indicator_calculator, mock_sr_handler):
    """Test analyze method when there's an open position."""
    # Setup mocks
    mock_indicator_calc_class.return_value = mock_indicator_calculator
    
    # Create a mock config with all required methods
    config = MagicMock()
    config.get.return_value = {'primary': 'M15', 'secondary': 'H1'}
    config.get_indicator_params.return_value = {}
    
    # Initialize the strategy engine with mocks
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler):
        strategy = StrategyEngine(config)
    
    # Create sample data with all required timeframes
    data = create_sample_market_data()
    
    # Test with open position
    position_info = {
        'size': 1.0,
        'entry_price': 100.0,
        'stop_loss': 99.0,
        'take_profit': 102.0,
        'symbol': 'EURUSD'
    }
    
    result = strategy.analyze('EURUSD', data, position_info=position_info)
    
    # Should return a result with a signal
    assert isinstance(result, dict)
    assert 'signal' in result
    
    # The signal should be valid (0, 1, or -1)
    assert result['signal'] in [0, 1, -1]
    
    # If we have an open position, we should get back the position info
    if result['signal'] != 0:
        assert 'entry_price' in result
        assert 'stop_loss' in result
        assert 'take_profit' in result
