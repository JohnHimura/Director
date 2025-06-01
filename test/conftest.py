"""
Configuration file for pytest.
Defines fixtures and other test configurations.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from core.config_manager import ConfigManager


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "metatrader5": {
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            "server": "Demo",
            "login": 12345678,
            "password": "password123"
        },
        "global_settings": {
            "max_open_positions": 5,
            "default_lot_size": 0.1,
            "max_daily_drawdown_pct": 5.0
        },
        "logging": {
            "level": "INFO",
            "file": "trading_bot.log",
            "max_size_mb": 10,
            "backup_count": 5
        },
        "defaults": {
            "indicators": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2.0
            },
            "sr": {
                "pivot_lookback": 5,
                "fractal_lookback": 2,
                "merge_threshold_pct": 0.5
            },
            "risk": {
                "risk_per_trade_pct": 1.0,
                "max_risk_pct": 5.0,
                "min_risk_reward_ratio": 1.5
            }
        },
        "symbols": {
            "EURUSD": {
                "enabled": True,
                "risk": {
                    "risk_per_trade_pct": 0.5,
                    "max_risk_pct": 3.0
                }
            },
            "GBPUSD": {
                "enabled": False
            }
        }
    }


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.3)
    low = close - np.abs(np.random.randn(100) * 0.3)
    volume = np.random.randint(100, 1000, size=100)
    
    return pd.DataFrame({
        'open': close - 0.1,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def mock_config_manager_for_engine():
    """
    Create a more comprehensive mock ConfigManager specifically for StrategyEngine tests,
    or for general use if applicable.
    """
    mock_cm = MagicMock(spec=ConfigManager)

    # Global settings, including strategy type and parameters
    mock_cm.get_global_settings.return_value = {
        C.CONFIG_STRATEGY: {
            C.CONFIG_STRATEGY_TYPE: "MACD",  # Default or common strategy for testing
            C.CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH: 0.7
        },
        # Add other global settings if StrategyEngine or strategies use them directly
        C.CONFIG_PAPER_TRADING: False, # Example
    }

    # Timeframes configuration
    mock_cm.get_timeframes.return_value = {
        "M15": "mt5_timeframe_m15_dummy", # Value isn't used by strategy, just keys
        "H1": "mt5_timeframe_h1_dummy",
        "H4": "mt5_timeframe_h4_dummy",
        "D1": "mt5_timeframe_d1_dummy",
    }

    # Default indicator parameters (can be overridden by symbol-specific if needed)
    mock_cm.get_indicator_params.return_value = {
        C.CONFIG_INDICATOR_MACD_FAST: 12,
        C.CONFIG_INDICATOR_MACD_SLOW: 26,
        C.CONFIG_INDICATOR_MACD_SIGNAL: 9,
        C.CONFIG_INDICATOR_RSI_PERIOD: 14,
        C.CONFIG_INDICATOR_ATR_PERIOD: 14,
        # Default for ADX, Stochastic, BB, etc.
        'adx_period': 14,
        'stoch_k_period': 14,
        'stoch_d_period': 3,
        'stoch_k_slowing': 3,
        'bb_length': 20,
        'bb_std': 2.0,
        # Default for Ichimoku
        'ichimoku_tenkan': 9,
        'ichimoku_kijun': 26,
        'ichimoku_senkou_span_b': 52,
    }

    # Mock get_symbol_config to return a merged view if needed by a strategy
    # For now, assuming strategies mostly use get_indicator_params, get_sr_params, etc.
    # If a strategy calls get_symbol_config directly, this might need more detail.
    def mock_get_symbol_config(symbol):
        # Default strategy_params, can be overridden in specific tests if needed
        strategy_params = {
            C.CONFIG_USE_ATR_SL_TP: C.DEFAULT_USE_ATR_SL_TP, # False by default
            C.CONFIG_ATR_SL_TP_ATR_PERIOD: C.DEFAULT_ATR_SL_TP_ATR_PERIOD, # e.g. 14
            C.CONFIG_ATR_SL_MULTIPLIER: C.DEFAULT_ATR_SL_MULTIPLIER, # e.g. 1.5
            C.CONFIG_ATR_TP_MULTIPLIER: C.DEFAULT_ATR_TP_MULTIPLIER, # e.g. 3.0
            C.CONFIG_DEFAULT_SL_PIPS: 100, # Example fallback
            C.CONFIG_DEFAULT_TP_PIPS: 200  # Example fallback
        }
        if symbol == "EURUSD_ATR_ON": # Special symbol name for testing ATR SL/TP ON
            strategy_params[C.CONFIG_USE_ATR_SL_TP] = True
            strategy_params[C.CONFIG_ATR_SL_MULTIPLIER] = 2.0
            strategy_params[C.CONFIG_ATR_TP_MULTIPLIER] = 4.0

        base_config = {
            C.CONFIG_INDICATORS: mock_cm.get_indicator_params(symbol), # Uses the above general mock
            C.CONFIG_SR: mock_cm.get_sr_params(symbol),             # Uses the general mock
            C.CONFIG_RISK: mock_cm.get_risk_params(symbol),           # Uses the general mock
            C.CONFIG_STRATEGY_PARAMS: strategy_params,
            C.CONFIG_ENABLED: True # Assume enabled for tests using this
            # Add other symbol-level configs if needed, like lot_size, spread_limit_pips
        }
        return base_config
    mock_cm.get_symbol_config.side_effect = mock_get_symbol_config

    mock_cm.get_sr_params.return_value = {"method":"pivots"} # Default SR params
    mock_cm.get_risk_params.return_value = {C.CONFIG_RISK_PER_TRADE: 0.01} # Default Risk params

    return mock_cm

# Keep the old mock_config if other tests rely on its specific simple structure,
# or update them to use the more comprehensive one.
# For clarity, I'll rename the old one or ensure new tests use the new one.
@pytest.fixture
def simple_mock_config():
    """A simpler mock ConfigManager, similar to the original mock_config."""
    config = MagicMock(spec=ConfigManager)
    config.get_indicator_params.return_value = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2.0
    }
    # Add get_global_settings and get_timeframes if any old tests implicitly need them
    config.get_global_settings.return_value = { C.CONFIG_STRATEGY: { C.CONFIG_STRATEGY_TYPE: "MACD"}}
    config.get_timeframes.return_value = {"M15": "dummy"}
    return config

# --- MockMT5 Integration Fixtures ---
# Import MockMT5 class from its location
# Assuming mock_mt5.py is in test/mocks/
try:
    from .mocks.mock_mt5 import MockMT5
except ImportError: # If conftest is run from a different path context, adjust import
    from test.mocks.mock_mt5 import MockMT5


@pytest.fixture(scope="function") # New instance for each test function
def mock_mt5_instance(monkeypatch):
    """
    Provides a fresh MockMT5 instance for each test and patches the 'mt5' import
    in specified core modules.
    """
    print("Setting up mock_mt5_instance fixture")
    mock = MockMT5()

    # Modules where 'import MetaTrader5 as mt5' is used and needs patching.
    # This ensures that these modules use our MockMT5 instance instead of the real one.
    modules_to_patch = [
        'core.utils.mt5_utils',    # MT5ConnectionManager in mt5_utils uses mt5 directly
        'core.mt5_connector',      # MT5Connector uses mt5 directly
        'core.trading_operations', # MT5TradingOperations uses mt5 directly
        'main_bot'                 # main_bot.py might use mt5 directly (e.g. for constants like TIMEFRAME_*)
    ]

    for module_name in modules_to_patch:
        # Patch the 'mt5' object within each specified module
        # This makes 'module.mt5' refer to our 'mock' instance.
        monkeypatch.setattr(f"{module_name}.mt5", mock)
        print(f"Patched '{module_name}.mt5' with MockMT5 instance.")

    # Also patch MT5_AVAILABLE flags to ensure mocked logic runs
    monkeypatch.setattr("core.utils.mt5_utils.MT5_AVAILABLE", True)
    monkeypatch.setattr("core.mt5_connector.MT5_AVAILABLE", True)
    monkeypatch.setattr("core.trading_operations.MT5_AVAILABLE", True)
    try: # main_bot might not have this flag if it only imports mt5 for constants
        monkeypatch.setattr("main_bot.MT5_AVAILABLE", True)
    except AttributeError:
        pass # It's fine if main_bot doesn't have this specific flag

    yield mock # Provide the configured mock instance to the test

    # Teardown (monkeypatch handles teardown of its changes automatically)
    print("Tearing down mock_mt5_instance fixture")
