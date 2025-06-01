"""
Tests for the StrategyEngine and MACDStrategy.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from core.strategy_engine import StrategyEngine, SignalType, MACDStrategy, StrategyResult
from core.config_manager import ConfigManager # For spec
from core.indicator_calculator import IndicatorCalculator # For spec
from core.sr_handler import SRHandler, SRLevel # For spec
from core import constants as C

# --- Fixtures ---

@pytest.fixture
def mock_indicator_calculator():
    """Create a default mock IndicatorCalculator."""
    mock = MagicMock(spec=IndicatorCalculator)
    # Default: return a copy of the DataFrame, can be overridden in tests
    mock.calculate_all.side_effect = lambda df, indicator_config=None: df.copy()
    return mock

@pytest.fixture
def mock_sr_handler_generic():
    """Generic mock SRHandler returning one support and one resistance."""
    mock = MagicMock(spec=SRHandler)
    support_level = SRLevel(price=95.0, strength=3, level_type=C.SR_SUPPORT, start_time=pd.Timestamp('2023-01-01'), end_time=pd.Timestamp('2023-01-01'))
    resistance_level = SRLevel(price=105.0, strength=3, level_type=C.SR_RESISTANCE, start_time=pd.Timestamp('2023-01-01'), end_time=pd.Timestamp('2023-01-01'))
    mock.get_sr_levels.return_value = [support_level, resistance_level]
    return mock

def create_sample_market_data(periods=250, symbol_name="M15"):
    """Creates sample market data with OHLCV and assigned name."""
    dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=periods, freq='15min')
    base_data = {
        'open': 100.0 + np.random.randn(periods).cumsum() * 0.1,
        'high': 100.0, # Will be adjusted
        'low': 100.0,  # Will be adjusted
        'close': 100.0 + np.random.randn(periods).cumsum() * 0.1,
        'volume': np.random.randint(100, 1000, periods)
    }
    df = pd.DataFrame(base_data, index=dates)
    df['open'] = df['open'].round(4)
    df['close'] = df['close'].round(4)
    df['high'] = np.maximum.reduce([df['high'], df['open'], df['close']]) + np.abs(np.random.randn(periods) * 0.05)
    df['low'] = np.minimum.reduce([df['low'], df['open'], df['close']]) - np.abs(np.random.randn(periods) * 0.05)
    df['high'] = df['high'].round(4)
    df['low'] = df['low'].round(4)
    df.name = symbol_name
    return df

@pytest.fixture
def sample_market_data_dict():
    """Returns a dictionary of DataFrames for M15, H1, H4, D1."""
    m15 = create_sample_market_data(periods=250, symbol_name="M15")

    def resample_ohlc(df, rule, name):
        resampled = df.resample(rule).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        resampled.name = name
        return resampled

    return {
        "M15": m15,
        "H1": resample_ohlc(m15, "1H", "H1"),
        "H4": resample_ohlc(m15, "4H", "H4"),
        "D1": resample_ohlc(m15, "1D", "D1"),
    }

# --- StrategyEngine Tests ---
def test_strategy_engine_initialization(mock_config_manager_for_engine):
    engine = StrategyEngine(mock_config_manager_for_engine)
    assert isinstance(engine.strategy, MACDStrategy)

# --- MACDStrategy Tests ---
@patch('core.strategy_engine.IndicatorCalculator')
def test_macd_strategy_analyze_buy_atr_sltp(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine, sample_market_data_dict):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    # Configure for ATR SL/TP
    mock_config_manager_for_engine.get_symbol_config.return_value[C.CONFIG_STRATEGY_PARAMS] = {
        C.CONFIG_USE_ATR_SL_TP: True,
        C.CONFIG_ATR_SL_MULTIPLIER: 2.0,
        C.CONFIG_ATR_TP_MULTIPLIER: 3.0
    }
    
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        engine = StrategyEngine(mock_config_manager_for_engine)

    entry_price_sim = 100.0
    atr_value_sim = 1.0 # Make ATR simple for assertion

    def custom_calc_all(df, indicator_config):
        df_c = df.copy()
        idx = df_c.index
        if df.name == "M15":
            df_c[C.INDICATOR_RSI] = 45.0
            df_c[C.INDICATOR_MACD] = pd.Series([0.5]*(len(idx)-2) + [0.9, 1.2], index=idx)
            df_c[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([0.8]*(len(idx)-2) + [1.0, 1.0], index=idx)
            df_c[C.INDICATOR_ATR] = pd.Series([atr_value_sim]*len(idx), index=idx)
            df_c[C.INDICATOR_CLOSE_PRICE] = pd.Series([entry_price_sim]*len(idx), index=idx) # Current price = entry
        elif df.name in ["H4", "D1"]: # Uptrend
            df_c[C.INDICATOR_EMA_50] = 100.0; df_c[C.INDICATOR_EMA_200] = 90.0
        return df_c
    mock_indicator_calculator.calculate_all.side_effect = custom_calc_all

    result = engine.analyze("EURUSD_ATR_ON", sample_market_data_dict) # Use special symbol for ATR config

    assert result['signal'] == SignalType.BUY
    assert result[C.POSITION_OPEN_PRICE] == entry_price_sim
    assert result[C.POSITION_SL] == pytest.approx(entry_price_sim - (atr_value_sim * 2.0))
    assert result[C.POSITION_TP] == pytest.approx(entry_price_sim + (atr_value_sim * 3.0))

@patch('core.strategy_engine.IndicatorCalculator')
def test_macd_strategy_analyze_sell_fallback_sltp(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine, sample_market_data_dict):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    # Configure for fallback SL/TP (use_atr_sl_tp = False)
    mock_config_manager_for_engine.get_symbol_config.return_value[C.CONFIG_STRATEGY_PARAMS] = {
        C.CONFIG_USE_ATR_SL_TP: False,
        # Fallback multipliers might come from C.DEFAULT_ATR_SL_MULTIPLIER etc. if used by strategy
    }
    # Mock S/R handler to return specific resistance for fallback logic
    resistance_price = 105.0
    mock_sr_handler_generic.get_sr_levels.return_value = [
        SRLevel(price=95.0, strength=3, level_type=C.SR_SUPPORT, start_time=pd.Timestamp('2023-01-01'), end_time=pd.Timestamp('2023-01-01')),
        SRLevel(price=resistance_price, strength=3, level_type=C.SR_RESISTANCE, start_time=pd.Timestamp('2023-01-01'), end_time=pd.Timestamp('2023-01-01'))
    ]

    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        engine = StrategyEngine(mock_config_manager_for_engine)

    entry_price_sim = 104.8 # Near resistance
    atr_value_sim = 0.2 # Example ATR

    def custom_calc_all(df, indicator_config):
        df_c = df.copy()
        idx = df_c.index
        if df.name == "M15":
            df_c[C.INDICATOR_RSI] = 55.0
            df_c[C.INDICATOR_MACD] = pd.Series([1.2]*(len(idx)-2) + [0.8, 0.7], index=idx)
            df_c[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([1.0]*(len(idx)-2) + [0.9, 0.9], index=idx)
            df_c[C.INDICATOR_ATR] = pd.Series([atr_value_sim]*len(idx), index=idx)
            df_c[C.INDICATOR_CLOSE_PRICE] = pd.Series([entry_price_sim]*len(idx), index=idx)
        elif df.name in ["H4", "D1"]: # Downtrend
            df_c[C.INDICATOR_EMA_50] = 90.0; df_c[C.INDICATOR_EMA_200] = 100.0
        return df_c
    mock_indicator_calculator.calculate_all.side_effect = custom_calc_all

    result = engine.analyze("EURUSD", sample_market_data_dict)

    assert result['signal'] == SignalType.SELL
    assert result[C.POSITION_OPEN_PRICE] == entry_price_sim
    # Fallback SL for SELL: max(nearest_resistance.price, current_price + atr * DEFAULT_ATR_SL_MULTIPLIER)
    expected_sl = max(resistance_price, entry_price_sim + atr_value_sim * C.DEFAULT_ATR_SL_MULTIPLIER)
    assert result[C.POSITION_SL] == pytest.approx(expected_sl)
    # Fallback TP for SELL: entry_price - (sl - entry_price) * DEFAULT_ATR_TP_MULTIPLIER (R:R)
    expected_tp = entry_price_sim - (expected_sl - entry_price_sim) * C.DEFAULT_ATR_TP_MULTIPLIER
    assert result[C.POSITION_TP] == pytest.approx(expected_tp)

# Add other tests from previous refactoring (no_signal, exit_signal, get_trend_direction, StrategyResult)
# Ensure they use the updated fixtures and structure.

@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_no_signal(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine, sample_market_data_dict):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)

    def custom_calculate_all_no_signal(df, indicator_config):
        df_copy = df.copy()
        idx = df_copy.index
        if df.name == "M15":
            df_copy[C.INDICATOR_RSI] = 50.0
            df_copy[C.INDICATOR_MACD] = pd.Series([0.5] * len(idx), index=idx)
            df_copy[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([0.5] * len(idx), index=idx)
            df_copy[C.INDICATOR_ATR] = 0.0010
            df_copy[C.INDICATOR_CLOSE_PRICE] = pd.Series([100.0] * len(idx), index=idx)
        elif df.name == "H4":
            df_copy[C.INDICATOR_EMA_50] = 100.0; df_copy[C.INDICATOR_EMA_200] = 90.0
        elif df.name == "D1":
            df_copy[C.INDICATOR_EMA_50] = 90.0; df_copy[C.INDICATOR_EMA_200] = 100.0
        return df_copy
    mock_indicator_calculator.calculate_all.side_effect = custom_calculate_all_no_signal
    result = strategy_engine.analyze('EURUSD', sample_market_data_dict)
    assert result['signal'] == SignalType.NONE
    assert "No clear signal" in result.get('message',"")


@patch('core.strategy_engine.IndicatorCalculator')
def test_analyze_exit_long_on_sell_signal(MockIndicatorCalculator, mock_indicator_calculator, mock_sr_handler_generic, mock_config_manager_for_engine, sample_market_data_dict):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    with patch('core.strategy_engine.SRHandler', return_value=mock_sr_handler_generic):
        strategy_engine = StrategyEngine(mock_config_manager_for_engine)

    def custom_calculate_all_sell_condition(df, indicator_config):
        df_copy = df.copy()
        idx = df_copy.index
        if df.name == "M15":
            df_copy[C.INDICATOR_RSI] = 55.0
            df_copy[C.INDICATOR_MACD] = pd.Series([1.2]*(len(idx)-2) + [0.8,0.7], index=idx)
            df_copy[C.INDICATOR_MACD_SIGNAL_LINE] = pd.Series([1.0]*(len(idx)-2) + [0.9,0.9], index=idx)
            df_copy[C.INDICATOR_ATR] = 0.0010
        return df_copy
    mock_indicator_calculator.calculate_all.side_effect = custom_calculate_all_sell_condition
    position_info = { C.POSITION_TYPE: SignalType.BUY, C.POSITION_OPEN_PRICE: 100.0, C.POSITION_SL: 99.0, C.POSITION_TP: 105.0 }
    result = strategy_engine.analyze('EURUSD', sample_market_data_dict, position_info=position_info)
    assert result['signal'] == SignalType.SELL
    assert "Exit: MACD cross down" in result.get('message',"")


@patch('core.strategy_engine.IndicatorCalculator')
def test_get_trend_direction(MockIndicatorCalculator, mock_indicator_calculator, mock_config_manager_for_engine, sample_market_data_dict):
    MockIndicatorCalculator.return_value = mock_indicator_calculator
    strategy_engine = StrategyEngine(mock_config_manager_for_engine)
    from core.strategy_engine import MACDStrategy
    assert isinstance(strategy_engine.strategy, MACDStrategy)
    macd_strategy = strategy_engine.strategy

    data_for_trend = sample_market_data_dict # Use the full dict

    def calc_trend(df, cfg, h4_up, d1_up):
        df_copy = df.copy()
        if df.name == "H4":
            df_copy[C.INDICATOR_EMA_50] = 100 if h4_up else 90
            df_copy[C.INDICATOR_EMA_200] = 90 if h4_up else 100
        elif df.name == "D1":
            df_copy[C.INDICATOR_EMA_50] = 100 if d1_up else 90
            df_copy[C.INDICATOR_EMA_200] = 90 if d1_up else 100
        return df_copy

    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, True, True)
    assert macd_strategy.get_trend_direction(data_for_trend) == 1

    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, False, False)
    assert macd_strategy.get_trend_direction(data_for_trend) == -1

    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, True, False)
    assert macd_strategy.get_trend_direction(data_for_trend) == 0

    data_h4_only = {"M15": data_for_trend["M15"].copy(), "H4": data_for_trend["H4"].copy()}
    mock_indicator_calculator.calculate_all.side_effect = lambda df, cfg: calc_trend(df, cfg, True, True)
    assert macd_strategy.get_trend_direction(data_h4_only) == 1
    
    data_m15_only = {"M15": data_for_trend["M15"].copy()}
    assert macd_strategy.get_trend_direction(data_m15_only) == 0


def test_strategy_result_class():
    res = StrategyResult(signal=SignalType.BUY, message="Test", entry_price=1.1, stop_loss=1.0, take_profit=1.3,
                         indicators={C.INDICATOR_RSI: 50}, levels={C.SR_SUPPORT: [1.05]}, trend_direction=1)
    assert res.signal == SignalType.BUY
    assert res.indicators[C.INDICATOR_RSI] == 50
    d = res.to_dict()
    assert d['signal'] == SignalType.BUY
    assert d[C.CONFIG_INDICATORS][C.INDICATOR_RSI] == 50
