# core/constants.py

# Configuration Keys - Sections
CONFIG_GLOBAL_SETTINGS = "global_settings"
CONFIG_LOGGING = "logging"
CONFIG_METATRADER5 = "metatrader5"
CONFIG_DEFAULTS = "defaults"
CONFIG_SYMBOLS = "symbols"
CONFIG_STRATEGY = "strategy" # General strategy settings, often under global_settings
CONFIG_STRATEGY_PARAMS = "strategy_params" # Symbol-specific strategy parameters

# DataFrame column names for MT5 data
DATETIME_COL = "datetime"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low" 
CLOSE_COL = "close"
VOLUME_COL = "volume"

# Configuration Keys - Global Settings
CONFIG_PAPER_TRADING = "paper_trading"
CONFIG_MAX_SLIPPAGE_POINTS = "max_slippage_points"
CONFIG_MAGIC_NUMBER = "magic_number"
CONFIG_LOOP_INTERVAL = "loop_interval"
CONFIG_DEVIATION = "deviation"
CONFIG_MAX_TOTAL_TRADES = "max_total_trades"
CONFIG_MAX_SLIPPAGE_PIPS = "max_slippage_pips"
CONFIG_KILL_SWITCH_FILE_PATH = "kill_switch_file_path"
CONFIG_KILL_SWITCH_CLOSE_POSITIONS = "kill_switch_close_positions"
# Daily Drawdown Limit Config Keys
CONFIG_ENABLE_DAILY_DRAWDOWN_LIMIT = "enable_daily_drawdown_limit"
CONFIG_MAX_DAILY_DRAWDOWN_PERCENTAGE = "max_daily_drawdown_percentage"
CONFIG_CLOSE_POSITIONS_ON_DAILY_DRAWDOWN_LIMIT = "close_positions_on_daily_drawdown_limit"
# Account Drawdown Kill-Switch Config Keys
CONFIG_ENABLE_ACCOUNT_DRAWDOWN_KILL_SWITCH = "enable_account_drawdown_kill_switch"
CONFIG_MAX_ACCOUNT_DRAWDOWN_PERCENTAGE = "max_account_drawdown_percentage"
CONFIG_CLOSE_POSITIONS_ON_ACCOUNT_KILL_SWITCH = "close_positions_on_account_kill_switch"


# Configuration Keys - Logging
CONFIG_LOGGING_LEVEL = "level"
CONFIG_LOGGING_FILE = "file"
CONFIG_LOGGING_MAX_BYTES = "max_bytes"
CONFIG_LOGGING_BACKUP_COUNT = "backup_count"
CONFIG_LOGGING_FORMAT = "format"

# Configuration Keys - Metatrader5
CONFIG_MT5_PATH = "path"
CONFIG_MT5_SERVER = "server"
CONFIG_MT5_LOGIN = "login"
CONFIG_MT5_PASSWORD = "password"
CONFIG_MT5_TIMEOUT = "timeout"
CONFIG_MT5_PORTABLE = "portable"
CONFIG_MT5_CONNECTION_MAX_RETRIES = "connection_max_retries"
CONFIG_MT5_CONNECTION_RETRY_DELAY = "connection_retry_delay_seconds"

# Configuration Keys - Defaults & Symbols subsections
CONFIG_INDICATORS = "indicators"
CONFIG_RISK = "risk"
CONFIG_SR = "sr"

# Configuration Keys - Symbol specific
CONFIG_ENABLED = "enabled"
CONFIG_LOT_SIZE = "lot_size" # Also used as a key in RiskManager output
CONFIG_SPREAD_LIMIT_PIPS = "spread_limit_pips"

# Configuration Keys - Indicators (common examples)
CONFIG_INDICATOR_MACD_FAST = "macd_fast"
CONFIG_INDICATOR_MACD_SLOW = "macd_slow"
CONFIG_INDICATOR_MACD_SIGNAL = "macd_signal"
CONFIG_INDICATOR_RSI_PERIOD = "rsi_period"
CONFIG_INDICATOR_ATR_PERIOD = "atr_period"
CONFIG_INDICATOR_ATR_MULTIPLIER = "atr_multiplier"

# Configuration Keys - SR (Support/Resistance)
CONFIG_SR_METHOD = "method"
CONFIG_SR_PIVOT_TYPE = "pivot_type"
CONFIG_SR_FRACTAL_WINDOW = "fractal_window"

# Configuration Keys - Risk
CONFIG_RISK_PER_TRADE = "risk_per_trade"
CONFIG_MAX_RISK_PER_TRADE = "max_risk_per_trade"
CONFIG_RISK_REWARD_RATIO = "risk_reward_ratio"
CONFIG_OLD_SL_ATR_MULTIPLIER = "sl_atr_multiplier" # From existing risk config, might be fallback
OLD_SL_ATR_MULTIPLIER = "sl_atr_multiplier" # Para compatibilidad con versiones anteriores
CONFIG_OLD_TP_ATR_MULTIPLIER = "tp_atr_multiplier" # From existing risk config, might be fallback

# Configuration Keys - Strategy Params
CONFIG_USE_ATR_SL_TP = "use_atr_sl_tp"
CONFIG_ATR_SL_TP_ATR_PERIOD = "atr_sl_tp_atr_period"
CONFIG_ATR_SL_MULTIPLIER = "atr_sl_multiplier"
CONFIG_ATR_TP_MULTIPLIER = "atr_tp_multiplier"
CONFIG_DEFAULT_SL_PIPS = "default_stop_loss_pips"
CONFIG_DEFAULT_TP_PIPS = "default_take_profit_pips"
# Trailing Stop Loss Config Keys
CONFIG_ENABLE_TRAILING_STOP = "enable_trailing_stop"
CONFIG_TRAILING_START_PIPS_PROFIT = "trailing_start_pips_profit"
CONFIG_TRAILING_STEP_PIPS = "trailing_step_pips"
CONFIG_TRAILING_ACTIVATION_PRICE_DISTANCE_PIPS = "trailing_activation_price_distance_pips"
# Break-Even Stop Config Keys
CONFIG_ENABLE_BREAKEVEN_STOP = "enable_breakeven_stop"
CONFIG_BREAKEVEN_PIPS_PROFIT = "breakeven_pips_profit"
CONFIG_BREAKEVEN_EXTRA_PIPS = "breakeven_extra_pips"
# Time-Based Exit Config Keys
CONFIG_ENABLE_TIME_BASED_EXIT = "enable_time_based_exit"
CONFIG_MAX_TRADE_DURATION_HOURS = "max_trade_duration_hours"
# ADX Filter Config Keys
CONFIG_ENABLE_ADX_FILTER = "enable_adx_filter"
CONFIG_ADX_THRESHOLD = "adx_threshold"
# Note: ADX period for calculation is usually managed under CONFIG_INDICATORS section (e.g., "adx_period")
# News Filter Config Keys
CONFIG_ENABLE_NEWS_FILTER = "enable_news_filter"
CONFIG_HIGH_IMPACT_NEWS_WINDOWS = "high_impact_news_windows"

# Order Types
ORDER_TYPE_BUY = "BUY"
ORDER_TYPE_SELL = "SELL"

# Indicator Names / DataFrame Column Names
INDICATOR_OPEN_PRICE = "open"
INDICATOR_HIGH_PRICE = "high"
INDICATOR_LOW_PRICE = "low"
INDICATOR_CLOSE_PRICE = "close"
INDICATOR_VOLUME = "volume"

INDICATOR_RSI = "rsi"
INDICATOR_MACD = "macd"
INDICATOR_MACD_SIGNAL_LINE = "macd_signal"
INDICATOR_MACD_HISTOGRAM = "macd_hist"
INDICATOR_ATR = "atr"
INDICATOR_EMA_50 = "ema_50"
INDICATOR_EMA_200 = "ema_200"
INDICATOR_ADX = "adx"
INDICATOR_STOCH_K = "stoch_%k"
INDICATOR_STOCH_D = "stoch_%d"
INDICATOR_BB_UPPER = "bb_upper"
INDICATOR_BB_MIDDLE = "bb_middle"
INDICATOR_BB_LOWER = "bb_lower"
INDICATOR_OBV = "obv"
INDICATOR_VWAP = "vwap"
ICHIMOKU_TENKAN = "ichimoku_tenkan"
ICHIMOKU_KIJUN = "ichimoku_kijun"
ICHIMOKU_SENKOU_A = "ichimoku_senkou_a"
ICHIMOKU_SENKOU_B = "ichimoku_senkou_b"
ICHIMOKU_CHIKOU = "ichimoku_chikou"

INDICATOR_PLUS_DI = "plus_di"
INDICATOR_MINUS_DI = "minus_di"

# SR Level Types
SR_SUPPORT = "support"
SR_RESISTANCE = "resistance"

# Timeframes
TF_M15 = "M15"
TF_H1 = "H1"
TF_H4 = "H4"
TF_D1 = "D1"

# Strategy settings (often under global_settings.strategy or symbol_settings.strategy_params)
CONFIG_STRATEGY_TYPE = "type" # Old key, might be replaced by CONFIG_STRATEGY_NAME per symbol
CONFIG_STRATEGY_NAME = "strategy_name" # New key for per-symbol strategy selection
CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH = "min_signal_strength"

# Cache related
CACHE_PREFIX_CONFIG = "config"

# Position Info Keys (standardized for internal use, map from/to MT5 object keys)
POSITION_TICKET = "ticket"
POSITION_SYMBOL = "symbol"
POSITION_TYPE = "type"
POSITION_SL = "sl"
POSITION_TP = "tp"
POSITION_OPEN_PRICE = "price_open"
POSITION_VOLUME = "volume"
POSITION_MAGIC = "magic"
POSITION_COMMENT = "comment"
POSITION_TIME = "time" # Unix timestamp of position opening

# Request keys (for MT5 order_send, and for our simulated results)
REQUEST_ACTION = "action"
REQUEST_SYMBOL = "symbol"
REQUEST_VOLUME = "volume"
REQUEST_TYPE = "type"
REQUEST_PRICE = "price"
REQUEST_SL = "sl"
REQUEST_TP = "tp"
REQUEST_DEVIATION = "deviation"
REQUEST_MAGIC = "magic"
REQUEST_COMMENT = "comment"
REQUEST_TYPE_TIME = "type_time"
REQUEST_TYPE_FILLING = "type_filling"
REQUEST_POSITION = "position"

# Retcodes (MT5 TRADE_RETCODE_*)
RETCODE_DONE = 10009

# Logging & Messages
PAPER_TRADE_COMMENT_PREFIX = "Paper trade"
LOG_MSG_ORDER_OPENED = "Order opened successfully"
LOG_MSG_ORDER_FAILED = "Order failed"

# Keys for 'changes' dict in ConfigManager hot-reload
RELOAD_CHANGES_LOGGING_LEVEL = "logging_level"
RELOAD_CHANGES_GLOBAL_SETTINGS = "global_settings"
RELOAD_CHANGES_SYMBOLS = "symbols"
RELOAD_CHANGES_LOGGING_FILE_REQUIRES_RESTART = "logging_file_changed_requires_restart"

# Misc
PRIMARY_TIMEFRAME_KEY = "primary"

# File suffixes
YAML_SUFFIX_LOWER = ".yaml"
YAML_SUFFIX_YML = ".yml"

# Default values
DEFAULT_MAGIC_NUMBER = 123456
DEFAULT_DEVIATION = 10
DEFAULT_MAX_SLIPPAGE_POINTS = 10
DEFAULT_PAPER_TRADING = False
DEFAULT_LOGGING_LEVEL = "INFO"
DEFAULT_USE_ATR_SL_TP = False
DEFAULT_ATR_SL_TP_ATR_PERIOD = 14
DEFAULT_ATR_SL_MULTIPLIER = 1.5
DEFAULT_ATR_TP_MULTIPLIER = 3.0
DEFAULT_FALLBACK_TP_RR_RATIO = 1.5  # Valor predeterminado para la relación riesgo/recompensa
# Trailing Stop Loss Defaults
DEFAULT_ENABLE_TRAILING_STOP = False
DEFAULT_TRAILING_START_PIPS_PROFIT = 50
DEFAULT_TRAILING_STEP_PIPS = 10
DEFAULT_TRAILING_ACTIVATION_PRICE_DISTANCE_PIPS = 20
# Break-Even Defaults
DEFAULT_ENABLE_BREAKEVEN_STOP = False
DEFAULT_BREAKEVEN_PIPS_PROFIT = 30
DEFAULT_BREAKEVEN_EXTRA_PIPS = 2
# Time-Based Exit Defaults
DEFAULT_ENABLE_TIME_BASED_EXIT = False
DEFAULT_MAX_TRADE_DURATION_HOURS = 24 # e.g., 1 day
# Daily Drawdown Defaults
DEFAULT_ENABLE_DAILY_DRAWDOWN_LIMIT = False
DEFAULT_MAX_DAILY_DRAWDOWN_PERCENTAGE = 2.0 # Default 2%
DEFAULT_CLOSE_POSITIONS_ON_DAILY_DRAWDOWN_LIMIT = True
# ADX Filter Defaults
DEFAULT_ENABLE_ADX_FILTER = False
DEFAULT_ADX_THRESHOLD = 25.0
DEFAULT_ADX_PERIOD = 14 # Default period for ADX calculation if not specified elsewhere
# MT5 Connection Retry Defaults
DEFAULT_MT5_CONNECTION_MAX_RETRIES = 5
DEFAULT_MT5_CONNECTION_RETRY_DELAY_SECONDS = 5.0
# Account Drawdown Kill-Switch Defaults
DEFAULT_ENABLE_ACCOUNT_DRAWDOWN_KILL_SWITCH = False
DEFAULT_MAX_ACCOUNT_DRAWDOWN_PERCENTAGE = 10.0 # Default 10% account drawdown
DEFAULT_CLOSE_POSITIONS_ON_ACCOUNT_KILL_SWITCH = False
# News Filter Defaults
DEFAULT_ENABLE_NEWS_FILTER = True

# State Variable Keys
STATE_PEAK_EQUITY = "peak_account_equity"
STATE_INITIAL_DAILY_BALANCE = "initial_daily_balance"
STATE_DAILY_PNL_REALIZED = "daily_pnl_realized"
STATE_LAST_RESET_DATE = "last_reset_date"
# ACCOUNT_KILL_SWITCH_HIT_KEY = "account_kill_switch_hit" # Already string literal in main_bot.py

# Logging
LOGS_DIR = "logs" # Default directory for logs

# Default strategy name
DEFAULT_STRATEGY_NAME = "MACDStrategy" # Example default


LOT_SIZE = "lot_size" # Key for RiskManager output, matches CONFIG_LOT_SIZE

# Account Info Keys (from MT5 account_info object, used as dict keys)
ACCOUNT_BALANCE = "balance"
ACCOUNT_EQUITY = "equity"
ACCOUNT_MARGIN = "margin"
ACCOUNT_MARGIN_FREE = "free_margin" # or "margin_free" from MT5
ACCOUNT_MARGIN_LEVEL = "margin_level"

# Indicator Configuration Keys
CONFIG_INDICATOR_RSI_PERIOD = "rsi_period"
CONFIG_INDICATOR_MACD_FAST = "macd_fast"
CONFIG_INDICATOR_MACD_SLOW = "macd_slow"
CONFIG_INDICATOR_MACD_SIGNAL = "macd_signal"
CONFIG_INDICATOR_ATR_PERIOD = "atr_period"
CONFIG_INDICATOR_BB_PERIOD = "bb_period"
CONFIG_INDICATOR_BB_STD_DEV = "bb_std_dev"
CONFIG_INDICATOR_STOCH_K_PERIOD = "stoch_k_period"
CONFIG_INDICATOR_STOCH_D_PERIOD = "stoch_d_period"
CONFIG_INDICATOR_STOCH_SMOOTH_K = "stoch_smooth_k"
