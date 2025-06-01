import pandas as pd
from backtesting import Backtest, Strategy
# from backtesting.lib import crossover # Removed as it's not directly used here
# from core.indicator_calculator import IndicatorCalculator # Not directly used here
from core.config_manager import ConfigManager
from core.strategy_engine import MACDStrategy, SignalType # Example strategy, direct import
# from core.strategy_engine import Strategy as BaseBotStrategy # For type hinting if needed for strategy_class_to_run
from core import constants as C

import logging
from typing import Optional, Callable, Type # Added for type hints
import numpy as np # Added for WFA summary
# import matplotlib.pyplot as plt # For combined equity plot if uncommented

logger = logging.getLogger(__name__)

def load_historical_data(filepath: str, symbol: str) -> pd.DataFrame:
    """
    Loads historical data from a CSV file, filters by symbol,
    and prepares it for backtesting.py.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading data file {filepath}: {e}")
        raise

    # Assuming a 'Symbol' column exists for filtering
    if 'Symbol' in df.columns:
        df = df[df['Symbol'] == symbol]
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} in {filepath}")
    else:
        logger.warning(f"'Symbol' column not found in {filepath}, assuming all data is for the target symbol.")

    # Ensure standard column names (Open, High, Low, Close, Volume)
    # backtesting.py is case-sensitive for these.
    rename_map = {
        # Potential variations from different data sources
        'timestamp': 'Timestamp', 'date': 'Timestamp', 'datetime': 'Timestamp', 'Time': 'Timestamp',
        'open': 'Open', 'OPEN': 'Open',
        'high': 'High', 'HIGH': 'High',
        'low': 'Low', 'LOW': 'Low',
        'close': 'Close', 'CLOSE': 'Close',
        'volume': 'Volume', 'VOLUME': 'Volume', 'vol': 'Volume'
    }
    # Only rename columns that exist in the DataFrame
    df_columns_lower = {col.lower():col for col in df.columns}
    effective_rename_map = {}
    for k_lower, k_expected in rename_map.items():
        if k_lower in df_columns_lower:
            effective_rename_map[df_columns_lower[k_lower]] = k_expected

    df.rename(columns=effective_rename_map, inplace=True)

    if 'Timestamp' not in df.columns:
        raise ValueError("Timestamp column ('Timestamp', 'Date', 'Time', or 'datetime') not found in data.")

    required_ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_ohlcv_cols:
        if col not in df.columns:
            raise ValueError(f"Required OHLCV column '{col}' not found in data after renaming.")

    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        logger.error(f"Error converting Timestamp column to datetime: {e}")
        raise

    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True) # Ensure data is sorted by time

    # Ensure OHLC are numeric, handle potential errors
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=required_ohlcv_cols, inplace=True) # Drop rows where essential OHLCV data is missing

    logger.info(f"Loaded {len(df)} data points for {symbol} from {filepath}")
    return df


class BacktestingStrategyWrapper(Strategy):
    # --- Parameters to be dynamically set by backtesting.py optimize ---
    # These names MUST match the keys in optimization_params in run_walk_forward_analysis
    rsi_period = C.DEFAULT_RSI_PERIOD
    macd_fast = C.DEFAULT_MACD_FAST
    macd_slow = C.DEFAULT_MACD_SLOW
    macd_signal = C.DEFAULT_MACD_SIGNAL
    atr_period = C.DEFAULT_ATR_PERIOD

    use_atr_sl_tp = C.DEFAULT_USE_ATR_SL_TP
    atr_sl_tp_atr_period = C.DEFAULT_ATR_SL_TP_ATR_PERIOD
    atr_sl_multiplier = C.DEFAULT_ATR_SL_MULTIPLIER
    atr_tp_multiplier = C.DEFAULT_ATR_TP_MULTIPLIER

    # ADX Filter parameters
    enable_adx_filter = C.DEFAULT_ENABLE_ADX_FILTER
    adx_threshold = C.DEFAULT_ADX_THRESHOLD

    # --- Class attributes to be set by run_walk_forward_analysis or run_backtest ---
    strategy_class_to_run: Optional[Type[MACDStrategy]] = None # Type hint for clarity
    symbol_to_trade: str = "DEFAULTSYMBOL"
    config_manager_instance: Optional[ConfigManager] = None

    _opt_keys = [
        "rsi_period", "macd_fast", "macd_slow", "macd_signal", "atr_period",
        "use_atr_sl_tp", "atr_sl_tp_atr_period", "atr_sl_multiplier", "atr_tp_multiplier",
        "enable_adx_filter", "adx_threshold"
    ]

    def init(self):
        if self.strategy_class_to_run is None or self.config_manager_instance is None:
            raise ValueError("Strategy class or ConfigManager not provided to BacktestingStrategyWrapper.")

        current_opt_params = {key: getattr(self, key) for key in self._opt_keys if hasattr(self, key)}

        self.bot_strategy = self.strategy_class_to_run(
            config_manager=self.config_manager_instance,
            symbol_for_params=self.symbol_to_trade,
            **current_opt_params
        )

        current_data_df = pd.DataFrame({
            C.INDICATOR_OPEN_PRICE: self.data.Open, C.INDICATOR_HIGH_PRICE: self.data.High,
            C.INDICATOR_LOW_PRICE: self.data.Low, C.INDICATOR_CLOSE_PRICE: self.data.Close,
            C.INDICATOR_VOLUME: self.data.Volume
        })
        # Ensure primary_tf is correctly determined
        timeframes = self.config_manager_instance.get_timeframes()
        primary_tf_key = C.PRIMARY_TIMEFRAME_KEY if hasattr(C, 'PRIMARY_TIMEFRAME_KEY') else 'primary'
        primary_tf = timeframes.get(primary_tf_key, list(timeframes.keys())[0] if timeframes else C.TF_M15)

        current_data_df.name = primary_tf # Set name for potential use within strategy
        self.data_dict_for_strategy = {primary_tf: current_data_df}

        # Calculate all indicators needed by the strategy for the entire dataset
        # The bot_strategy's analyze method will use these pre-calculated indicators
        indicator_params_for_calc = self.bot_strategy.config.get_indicator_params(self.symbol_to_trade)
        # Override with optimized params for calculation if necessary
        # This ensures that indicators are calculated with the parameters being tested.
        for key in self._opt_keys: # Iterate through all optimizable keys
            # Check if the key is an indicator parameter expected by IndicatorCalculator
            # This check can be made more specific if IndicatorCalculator params are distinctly named
            if key in indicator_params_for_calc and hasattr(self, key):
                 indicator_params_for_calc[key] = getattr(self, key)
            elif key == 'adx_period' and hasattr(self, key): # Explicitly handle adx_period if it's separate
                indicator_params_for_calc['adx_period'] = getattr(self, key)


        self.data_dict_for_strategy[primary_tf] = self.bot_strategy.indicator_calc.calculate_all(
            self.data_dict_for_strategy[primary_tf].copy(), # Pass a copy to avoid issues
            indicator_config=indicator_params_for_calc
        )

        for col in self.data_dict_for_strategy[primary_tf].columns:
            if col.lower() not in [C.INDICATOR_OPEN_PRICE, C.INDICATOR_HIGH_PRICE, C.INDICATOR_LOW_PRICE, C.INDICATOR_CLOSE_PRICE, C.INDICATOR_VOLUME]:
                indicator_series = self.data_dict_for_strategy[primary_tf][col]
                # Sanitize column name for self.I (e.g. STOCHk_14_3_3 -> STOCHk_14_3_3)
                safe_col_name = ''.join(c if c.isalnum() else '_' for c in col)
                setattr(self, safe_col_name, self.I(lambda x, s=indicator_series: s, name=safe_col_name))

    def next(self):
        position_info = None
        if self.position:
            position_info = {
                C.POSITION_TYPE: SignalType.BUY if self.position.is_long else SignalType.SELL,
                C.POSITION_OPEN_PRICE: self.position.entry_price,
                C.POSITION_SL: self.position.sl or 0.0,
                C.POSITION_TP: self.position.tp or 0.0,
            }

        analysis_result_dict = self.bot_strategy.analyze(
            self.symbol_to_trade,
            self.data_dict_for_strategy,
            position_info
        )

        signal = analysis_result_dict.get('signal', SignalType.NONE)
        sl_price = self.bot_strategy.stop_loss
        tp_price = self.bot_strategy.take_profit

        if not self.position:
            if signal == SignalType.BUY:
                self.buy(sl=sl_price if sl_price > 0 else None, tp=tp_price if tp_price > 0 else None)
            elif signal == SignalType.SELL:
                self.sell(sl=sl_price if sl_price > 0 else None, tp=tp_price if tp_price > 0 else None)
        elif self.position:
            if self.position.is_long and signal == SignalType.SELL:
                self.position.close()
            elif self.position.is_short and signal == SignalType.BUY:
                self.position.close()

def run_backtest(
    symbol: str,
    strategy_class_to_run: Type[MACDStrategy],
    config_manager_instance: ConfigManager,
    data_filepath: str,
    cash: float = 10000,
    commission_bps: float = 2.0,
    margin: float = 1.0,
    show_plot: bool = True,
    **strategy_fixed_params
):
    """
    Runs a single backtest for a given symbol and strategy.
    """
    logger.info(f"Starting backtest for {symbol} with strategy {strategy_class_to_run.__name__}")
    logger.info(f"Initial cash: {cash}, Commission BPS: {commission_bps}, Margin: {margin}")
    if strategy_fixed_params:
        logger.info(f"Using fixed strategy parameters: {strategy_fixed_params}")

    try:
        data_df = load_historical_data(data_filepath, symbol)
        if data_df.empty:
            logger.error(f"No data loaded for {symbol}. Aborting backtest.")
            return None
    except Exception as e:
        logger.error(f"Failed to load data for backtest: {e}")
        return None

    BacktestingStrategyWrapper.strategy_class_to_run = strategy_class_to_run
    BacktestingStrategyWrapper.symbol_to_trade = symbol
    BacktestingStrategyWrapper.config_manager_instance = config_manager_instance

    for key, value in strategy_fixed_params.items():
        if hasattr(BacktestingStrategyWrapper, key):
            setattr(BacktestingStrategyWrapper, key, value)
            logger.info(f"Set fixed param on BacktestingStrategyWrapper: {key} = {value}")
        else:
            logger.warning(f"Parameter '{key}' not found as an attribute in BacktestingStrategyWrapper.")

    commission_decimal = commission_bps / 10000.0

    logger.warning("Slippage is not explicitly simulated in this backtest configuration. ")

    bt = Backtest(
        data_df,
        BacktestingStrategyWrapper,
        cash=cash,
        commission=commission_decimal,
        margin=margin
    )

    try:
        stats = bt.run()
        logger.info("Backtest finished. Statistics:")
        print(stats)

        trades_df = stats['_trades']
        if not trades_df.empty:
            logger.info("Trades:")
            print(trades_df[['Size', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'EntryTime', 'ExitTime', 'Duration']])
        else:
            logger.info("No trades were executed.")

        if show_plot:
            plot_filename = f"backtest_{symbol}_{strategy_class_to_run.__name__}.html"
            try:
                bt.plot(filename=plot_filename, open_browser=False)
                logger.info(f"Backtest plot saved to {plot_filename}")
            except Exception as e:
                logger.error(f"Failed to generate plot: {e}. Bokeh might be missing or there's an issue with data.")
        return stats

    except Exception as e:
        logger.error(f"Error during backtest execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    default_config_path = str(project_root / "config.json")
    default_data_path = str(project_root / "data" / "historical_data.csv")

    logger.info(f"Using config file: {default_config_path}")
    logger.info(f"Using data file: {default_data_path}")

    if not Path(default_config_path).exists():
        logger.error(f"Config file not found at {default_config_path}.")
    elif not Path(default_data_path).exists():
         logger.error(f"Data file not found at {default_data_path}.")
    else:
        logger.info("--- Running Single Backtest Example ---")
        config_mngr_single = ConfigManager(config_path=default_config_path)
        fixed_params_example = {
            'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'use_atr_sl_tp': True, 'atr_sl_multiplier': 2.0, 'atr_tp_multiplier': 3.0,
            'enable_adx_filter': True, 'adx_threshold': 20.0 # Example for ADX filter
        }
        run_backtest(
            symbol="EURUSD",
            strategy_class_to_run=MACDStrategy,
            config_manager_instance=config_mngr_single,
            data_filepath=default_data_path,
            cash=100000, commission_bps=0.2, margin=1.0/30.0,
            show_plot=True, **fixed_params_example
        )
        logger.info("--- Finished Single Backtest Example ---")

        logger.info("--- Running Walk-Forward Analysis Example ---")
        optimization_params_example = {
            'rsi_period': range(10, 16, 2),
            'macd_fast': range(10, 16, 2),
            'macd_slow': range(20, 31, 5),
            'macd_signal': range(7, 10, 1),
            'adx_threshold': range(20, 31, 5), # Optimizing ADX threshold
            'enable_adx_filter': [True]      # Keep ADX filter enabled for this optimization
        }
        def macd_constraint_wfa(params): # Renamed to avoid conflict if another constraint is defined
            return params.get('macd_fast', 12) < params.get('macd_slow', 26)

        run_walk_forward_analysis(
            symbol="EURUSD",
            strategy_class_name="MACDStrategy",
            config_path=default_config_path,
            data_filepath=default_data_path,
            train_duration_years=1,
            test_duration_years=1,
            optimization_params=optimization_params_example,
            optimization_constraint=macd_constraint_wfa,
            maximize_metric='SQN',
            initial_cash=100000, commission_bps=0.2, margin=1.0/30.0,
            show_plot_for_each_test_period=False
        )
        logger.info("--- Finished Walk-Forward Analysis Example ---")

def run_walk_forward_analysis(
    symbol: str,
    strategy_class_name: str,
    config_path: str,
    data_filepath: str,
    train_duration_years: int,
    test_duration_years: int,
    optimization_params: dict,
    optimization_constraint: Optional[Callable[[dict], bool]] = None,
    maximize_metric: str = 'SQN',
    initial_cash: float = 10000,
    commission_bps: float = 0.0,
    margin: float = 1.0,
    show_plot_for_each_test_period: bool = False
):
    logger.info(f"Starting Walk-Forward Analysis for {symbol}, Strategy: {strategy_class_name}")
    logger.info(f"Train: {train_duration_years} years, Test: {test_duration_years} years")
    logger.info(f"Optimization Params: {optimization_params}")
    logger.info(f"Maximize Metric: {maximize_metric}")

    full_data_df = load_historical_data(data_filepath, symbol)
    if full_data_df.empty:
        logger.error("No data loaded, aborting walk-forward analysis.")
        return

    all_trades_df = pd.DataFrame()
    all_stats_list = []

    min_year = full_data_df.index.min().year
    max_year = full_data_df.index.max().year
    total_data_years = max_year - min_year + 1

    if total_data_years < train_duration_years + test_duration_years:
        logger.error(f"Not enough data ({total_data_years} years) for train ({train_duration_years}y) and test ({test_duration_years}y) durations.")
        return None, None # Return tuple

    config_manager = ConfigManager(config_path=config_path)

    if strategy_class_name == "MACDStrategy":
        strategy_class_to_run = MACDStrategy
    else:
        raise ValueError(f"Unsupported strategy_class_name: {strategy_class_name}")

    BacktestingStrategyWrapper.strategy_class_to_run = strategy_class_to_run
    BacktestingStrategyWrapper.symbol_to_trade = symbol
    BacktestingStrategyWrapper.config_manager_instance = config_manager

    commission_decimal = commission_bps / 10000.0

    current_start_year = min_year
    period_num = 0
    while True:
        period_num += 1
        train_start_date = pd.Timestamp(f"{current_start_year}-01-01")
        train_end_date = pd.Timestamp(f"{current_start_year + train_duration_years -1}-12-31")
        test_start_date = pd.Timestamp(f"{current_start_year + train_duration_years}-01-01")
        test_end_date = pd.Timestamp(f"{current_start_year + train_duration_years + test_duration_years -1}-12-31")

        if test_end_date.year > max_year :
             logger.info("Reached end of data for walk-forward periods.")
             break

        logger.info(f"\n--- Period {period_num}: Train {train_start_date.year}-{train_end_date.year}, Test {test_start_date.year}-{test_end_date.year} ---")

        train_data = full_data_df[(full_data_df.index >= train_start_date) & (full_data_df.index <= train_end_date)]
        test_data = full_data_df[(full_data_df.index >= test_start_date) & (full_data_df.index <= test_end_date)]

        if train_data.empty or len(train_data) < 60:
            logger.warning(f"Skipping Period {period_num} due to insufficient training data: {len(train_data)} bars.")
            current_start_year += test_duration_years
            if current_start_year + train_duration_years + test_duration_years -1 > max_year +1:
                 logger.info("Not enough data for subsequent periods.")
                 break
            continue
        if test_data.empty or len(test_data) < 60:
            logger.warning(f"Skipping Period {period_num} due to insufficient testing data: {len(test_data)} bars.")
            current_start_year += test_duration_years
            if current_start_year + train_duration_years + test_duration_years -1 > max_year+1:
                 logger.info("Not enough data for subsequent periods.")
                 break
            continue

        logger.info(f"Optimizing on training data ({len(train_data)} bars)...")
        bt_optimize = Backtest(train_data, BacktestingStrategyWrapper, cash=initial_cash, commission=commission_decimal, margin=margin)

        try:
            opt_params_copy = {k: list(v) if isinstance(v, range) else v for k, v in optimization_params.items()}

            opt_stats = bt_optimize.optimize(
                **opt_params_copy,
                maximize=maximize_metric,
                constraint=optimization_constraint,
                return_heatmap=False
            )
            best_params = opt_stats['_strategy']
            logger.info(f"Period {period_num} - Best params: {best_params}")
        except Exception as e:
            logger.error(f"Error during optimization for period {period_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            current_start_year += test_duration_years
            continue

        logger.info(f"Validating on testing data ({len(test_data)} bars) with best params...")
        for key, value in best_params.items():
            if hasattr(BacktestingStrategyWrapper, key):
                setattr(BacktestingStrategyWrapper, key, value)
            else:
                logger.warning(f"Parameter {key} from optimization not found as class attribute in BacktestingStrategyWrapper.")

        bt_validate = Backtest(test_data, BacktestingStrategyWrapper, cash=initial_cash, commission=commission_decimal, margin=margin)
        try:
            stats_test = bt_validate.run()
            logger.info(f"Period {period_num} - Test Results:\n{stats_test}")
            all_stats_list.append(stats_test)
            if not stats_test['_trades'].empty:
                all_trades_df = pd.concat([all_trades_df, stats_test['_trades']], ignore_index=True)

            if show_plot_for_each_test_period:
                plot_filename = f"wfa_period_{period_num}_{symbol}_{strategy_class_name}_test.html"
                bt_validate.plot(filename=plot_filename, open_browser=False)
                logger.info(f"Test period plot saved to {plot_filename}")

        except Exception as e:
            logger.error(f"Error during validation for period {period_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        current_start_year += test_duration_years

    logger.info("\n--- Walk-Forward Analysis Summary ---")
    if not all_stats_list:
        logger.info("No test periods were successfully completed.")
        return None, None

    avg_sqn = np.mean([s['SQN'] for s in all_stats_list if 'SQN' in s and pd.notna(s['SQN'])])
    avg_sharpe = np.mean([s['Sharpe Ratio'] for s in all_stats_list if 'Sharpe Ratio' in s and pd.notna(s['Sharpe Ratio'])])
    avg_profit_factor = np.mean([s['Profit Factor'] for s in all_stats_list if 'Profit Factor' in s and pd.notna(s['Profit Factor']) and s['Profit Factor'] != np.inf])

    logger.info(f"Average SQN (Out-of-Sample): {avg_sqn:.2f}")
    logger.info(f"Average Sharpe Ratio (Out-of-Sample): {avg_sharpe:.2f}")
    logger.info(f"Average Profit Factor (Out-of-Sample): {avg_profit_factor:.2f}")
    logger.info(f"Total Trades (Out-of-Sample): {len(all_trades_df)}")

    if not all_trades_df.empty:
        logger.info("Consolidated Out-of-Sample Trades:")
        print(all_trades_df.to_string(max_rows=10))

    logger.info("Walk-Forward Analysis Finished.")
    return all_stats_list, all_trades_df
```
