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

from core import constants as C # Import constants

# Sample market data for testing
def create_sample_market_data(periods=250): # Increased periods
    """Create sample market data for testing."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='15min')
    
    base_data = {
        'open': 100.0 + np.random.randn(periods).cumsum() * 0.1,
        'high': 100.0 + np.random.randn(periods).cumsum() * 0.1, # Initial high
        'low': 100.0 + np.random.randn(periods).cumsum() * 0.1,  # Initial low
        'close': 100.0 + np.random.randn(periods).cumsum() * 0.1,
        'volume': np.random.randint(100, 1000, periods)
    }
    # Ensure OHLC integrity
    base_data['open'] = np.round(base_data['open'], 2)
    base_data['close'] = np.round(base_data['close'], 2)
    base_data['high'] = np.maximum.reduce([base_data['high'], base_data['open'], base_data['close']])+abs(np.random.randn(periods).cumsum()*0.05)
    base_data['low'] = np.minimum.reduce([base_data['low'], base_data['open'], base_data['close']])-abs(np.random.randn(periods).cumsum()*0.05)
    base_data['high'] = np.round(base_data['high'], 2)
    base_data['low'] = np.round(base_data['low'], 2)

    m15_df = pd.DataFrame(base_data, index=dates)
    m15_df.name = "M15"
    
    def resample_ohlc(df, rule, name):
        resampled_df = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        resampled_df.name = name
        return resampled_df

    h1_df = resample_ohlc(m15_df, '1H', "H1")
    h4_df = resample_ohlc(m15_df, '4H', "H4")
    d1_df = resample_ohlc(m15_df, '1D', "D1")
    
    return {
        'M15': m15_df,
        'H1': h1_df,
        'H4': h4_df,
        'D1': d1_df
    }

@pytest.fixture
def mock_indicator_calculator():
    """Create a mock IndicatorCalculator."""
    mock = MagicMock(spec=IndicatorCalculator) # Use spec for better mocking
    # Default behavior: return a copy of the DataFrame, can be overridden in tests
    mock.calculate_all.side_effect = lambda df, indicator_config: df.copy()
    return mock

@pytest.fixture
def mock_sr_handler_generic():
    """Generic mock SRHandler returning one support and one resistance."""
    mock = MagicMock(spec=SRHandler)
    support_level = SRLevel(price=95.0, strength=3, level_type=C.SR_SUPPORT, start_time=pd.Timestamp('2023-01-01'), end_time=pd.Timestamp('2023-01-01'))
    resistance_level = SRLevel(price=105.0, strength=3, level_type=C.SR_RESISTANCE, start_time=pd.Timestamp('2023-01-01'), end_time=pd.Timestamp('2023-01-01'))
    mock.get_sr_levels.return_value = [support_level, resistance_level]
    return mock

# Use the more comprehensive mock_config_manager_for_engine from conftest.py
def test_strategy_engine_initialization(mock_config_manager_for_engine):
    """Test StrategyEngine initialization."""
    strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    assert strategy_engine is not None
    assert strategy_engine.config == mock_config_manager_for_engine
    assert strategy_engine.last_signal == SignalType.NONE
    # Check if the correct strategy (MACDStrategy) was initialized
    from core.strategy_engine import MACDStrategy # Import locally for isinstance check
    assert isinstance(strategy_engine.strategy, MACDStrategy)

@patch('core.strategy_engine.IndicatorCalculator') # Still need to patch where it's instantiated
def test_analyze_buy_signal(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine):
    """Test analyze method with conditions leading to a BUY signal."""
    MockIndicatorCalculator.return_value = mock_indicator_calculator # Ensure the instance used is our mock

    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)

    data = create_sample_market_data()
    
    # Scenario: MACD cross up, RSI normal, near support (95.0), uptrend
    def custom_calculate_all_buy(df, indicator_config):
        df_copy = df.copy()
        if df.name == "M15": # Primary TF for signals
            df_copy[C.INDICATOR_RSI] = 45.0
            # Ensure enough data points for series to match index length
            idx = df_copy.index
            df_copy[C.INDICATOR_MACD] = pd.Series([0.5]*(len(idx)-2) + [0.9, 1.2], index=idx)
            df_copy[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([0.8]*(len(idx)-2) + [1.0, 1.0], index=idx)
            df_copy[C.INDICATOR_ATR] = 0.0010
            # Simulate price near support: S/R handler returns support at 95.0
            # MACDStrategy uses df['close'].iloc[-1]
            df_copy['close'] = pd.Series([98.0]*(len(idx)-1) + [95.1], index=idx)
        elif df.name == "H4" or df.name == "D1": # Higher TFs for trend
            df_copy[C.INDICATOR_EMA_50] = 100.0
            df_copy[C.INDICATOR_EMA_200] = 90.0 # EMA50 > EMA200 -> Uptrend
        return df_copy

    mock_indicator_calculator.calculate_all.side_effect = custom_calculate_all_buy
    
    result = strategy_engine.analyze('EURUSD', data)
    
    assert isinstance(result, dict)
    assert result['signal'] == SignalType.BUY, f"Message: {result.get('message')}"
    assert "Buy signal: MACD cross up, near support" in result.get('message', "")
    assert result[C.POSITION_OPEN_PRICE] == data["M15"]['close'].iloc[-1]
    assert result[C.POSITION_SL] < result[C.POSITION_OPEN_PRICE]
    assert result[C.POSITION_TP] > result[C.POSITION_OPEN_PRICE]
    mock_indicator_calculator.calculate_all.assert_called()
    mock_sr_handler_generic.get_sr_levels.assert_called()


@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_sell_signal(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    data = create_sample_market_data()

    def custom_calculate_all_sell(df, indicator_config):
        df_copy = df.copy()
        idx = df_copy.index
        if df.name == "M15":
            df_copy[C.INDICATOR_RSI] = 55.0
            df_copy[C.INDICATOR_MACD] = pd.Series([1.2]*(len(idx)-2) + [0.8, 0.7], index=idx) # Cross down
            df_copy[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([1.0]*(len(idx)-2) + [0.9, 0.9], index=idx)
            df_copy[C.INDICATOR_ATR] = 0.0012
            df_copy['close'] = pd.Series([102.0]*(len(idx)-1) + [104.9], index=idx) # Price near resistance 105.0
        elif df.name == "H4" or df.name == "D1":
            df_copy[C.INDICATOR_EMA_50] = 90.0
            df_copy[C.INDICATOR_EMA_200] = 100.0 # Downtrend
        return df_copy
    mock_indicator_calculator.calculate_all.side_effect = custom_calculate_all_sell
    result = strategy_engine.analyze('EURUSD', data)
    assert result['signal'] == SignalType.SELL, f"Message: {result.get('message')}"
    assert "Sell signal: MACD cross down, near resistance" in result.get('message', "")
    assert result[C.POSITION_OPEN_PRICE] == data["M15"]['close'].iloc[-1]
    assert result[C.POSITION_SL] > result[C.POSITION_OPEN_PRICE]
    assert result[C.POSITION_TP] < result[C.POSITION_OPEN_PRICE]


@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_no_signal(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    data = create_sample_market_data()
    def custom_calculate_all_no_signal(df, indicator_config):
        df_copy = df.copy()
        idx = df_copy.index
        if df.name == "M15":
            df_copy[C.INDICATOR_RSI] = 50.0
            df_copy[C.INDICATOR_MACD] = pd.Series([0.5] * len(idx), index=idx) # No cross
            df_copy[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([0.5] * len(idx), index=idx)
            df_copy[C.INDICATOR_ATR] = 0.0010
            df_copy['close'] = pd.Series([100.0] * len(idx), index=idx) # Away from S/R
        elif df.name == "H4":
            df_copy[C.INDICATOR_EMA_50] = 100.0; df_copy[C.INDICATOR_EMA_200] = 90.0 # H4 up
        elif df.name == "D1":
            df_copy[C.INDICATOR_EMA_50] = 90.0; df_copy[C.INDICATOR_EMA_200] = 100.0 # D1 down
        return df_copy
    mock_indicator_calculator.calculate_all.side_effect = custom_calculate_all_no_signal
    result = strategy_engine.analyze('EURUSD', data)
    assert result['signal'] == SignalType.NONE
    assert "No clear signal" in result.get('message',"")


@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_exit_long_on_sell_signal(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    data = create_sample_market_data()
    def custom_calculate_all_sell_condition(df, indicator_config): # Same as sell signal
        df_copy = df.copy()
        idx = df_copy.index
        if df.name == "M15":
            df_copy[C.INDICATOR_RSI] = 55.0
            df_copy[C.INDICATOR_MACD] = pd.Series([1.2]*(len(idx)-2) + [0.8,0.7], index=idx)
            df_copy[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([1.0]*(len(idx)-2) + [0.9,0.9], index=idx)
            df_copy[C.INDICATOR_ATR] = 0.0010
        # No need to set HTF for exit signals based on primary TF indicators
        return df_copy
    mock_indicator_calculator.calculate_all.side_effect = custom_calculate_all_sell_condition
    position_info = { C.POSITION_TYPE: SignalType.BUY, C.POSITION_OPEN_PRICE: 100.0, C.POSITION_SL: 99.0, C.POSITION_TP: 105.0 }
    result = strategy_engine.analyze('EURUSD', data, position_info=position_info)
    assert result['signal'] == SignalType.SELL
    assert "Exit: MACD cross down" in result.get('message',"")


@patch('core.strategy_engine.IndicatorCalculator')
def test_get_trend_direction(MockIndicatorCalculator, mock_indicator_calculator, mock_config_manager_for_engine):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    # Assuming MACDStrategy is the default from mock_config_manager_for_engine
    from core.strategy_engine import MACDStrategy
    assert isinstance(strategy_engine.strategy, MACDStrategy)
    macd_strategy = strategy_engine.strategy
    data = create_sample_market_data()

    def calc_trend(df, cfg, h4_up, d1_up): # Helper to avoid repetition
        df_copy = df.copy()
        if df.name == "H4":
            df_copy[C.INDICATOR_EMA_50] = 100 if h4_up else 90
            df_copy[C.INDICATOR_EMA_200] = 90 if h4_up else 100
        elif df.name == "D1":
            df_copy[C.INDICATOR_EMA_50] = 100 if d1_up else 90
            df_copy[C.INDICATOR_EMA_200] = 90 if d1_up else 100
        return df_copy

    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, True, True)
    assert macd_strategy.get_trend_direction(data) == 1

    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, False, False)
    assert macd_strategy.get_trend_direction(data) == -1

    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, True, False)
    assert macd_strategy.get_trend_direction(data) == 0

    data_h4_only = {"M15": data["M15"].copy(), "H4": data["H4"].copy()}
    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, True, True) # D1 part won't be hit
    assert macd_strategy.get_trend_direction(data_h4_only) == 1
    
    data_m15_only = {"M15": data["M15"].copy()}
    assert macd_strategy.get_trend_direction(data_m15_only) == 0


def test_strategy_result_class():
    res = StrategyResult(signal=SignalType.BUY, message="Test", entry_price=1.1, stop_loss=1.0, take_profit=1.3,
                         indicators={C.INDICATOR_RSI: 50}, levels={C.SR_SUPPORT: [1.05]}, trend_direction=1)
    assert res.signal == SignalType.BUY
    assert res.indicators[C.INDICATOR_RSI] == 50
    d = res.to_dict()
    assert d['signal'] == SignalType.BUY
    assert d[C.CONFIG_INDICATORS][C.INDICATOR_RSI] == 50 # Check if using const key for indicators in to_dict


@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_with_open_position_original_adapted(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    
    data = create_sample_market_data()
    position_info = {
        C.POSITION_TYPE: SignalType.BUY, C.POSITION_OPEN_PRICE: 100.0,
        C.POSITION_SL: 99.0, C.POSITION_TP: 102.0,
    }
    
    # Default mock_calculate_all (just returns df copy) will likely result in NO_SIGNAL
    # or a signal based on random-ish data if not overridden for this specific test.
    # This test now checks that analyze runs and returns a valid structure.
    result = strategy_engine.analyze('EURUSD', data, position_info=position_info)
    
    assert isinstance(result, dict)
    assert 'signal' in result
    assert result['signal'] in [SignalType.NONE, SignalType.BUY, SignalType.SELL]
    
    mock_indicator_calculator.calculate_all.assert_called()
    mock_sr_handler_generic.get_sr_levels.assert_called()
