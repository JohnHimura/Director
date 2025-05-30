# core/constants.py

# Configuration Keys - Sections
CONFIG_GLOBAL_SETTINGS = "global_settings"
CONFIG_LOGGING = "logging"
CONFIG_METATRADER5 = "metatrader5"
CONFIG_DEFAULTS = "defaults"
CONFIG_SYMBOLS = "symbols"
CONFIG_STRATEGY = "strategy"

# Configuration Keys - Global Settings
CONFIG_PAPER_TRADING = "paper_trading"
CONFIG_MAX_SLIPPAGE_POINTS = "max_slippage_points"
CONFIG_MAGIC_NUMBER = "magic_number"
CONFIG_LOOP_INTERVAL = "loop_interval"
CONFIG_DEVIATION = "deviation" # Retained for now, though slippage uses max_slippage_points
CONFIG_MAX_TOTAL_TRADES = "max_total_trades"
CONFIG_MAX_SLIPPAGE_PIPS = "max_slippage_pips"


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

# Configuration Keys - Defaults & Symbols subsections
CONFIG_INDICATORS = "indicators"
CONFIG_RISK = "risk"
CONFIG_SR = "sr" # Support/Resistance

# Configuration Keys - Symbol specific
CONFIG_ENABLED = "enabled"
CONFIG_LOT_SIZE = "lot_size"
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
# ... other SR params if needed

# Configuration Keys - Risk
CONFIG_RISK_PER_TRADE = "risk_per_trade"
CONFIG_MAX_RISK_PER_TRADE = "max_risk_per_trade"
# ... other risk params

# Order Types (used in trading_operations and potentially strategies if sending string types)
ORDER_TYPE_BUY = "BUY"
ORDER_TYPE_SELL = "SELL"

# Indicator Names (used in DataFrames, analysis results etc.)
INDICATOR_RSI = "rsi"
INDICATOR_MACD = "macd"
INDICATOR_MACD_SIGNAL_LINE = "macd_signal" # Name of the signal line column in MACD
INDICATOR_ATR = "atr"
INDICATOR_EMA_50 = "ema_50"
INDICATOR_EMA_200 = "ema_200"
# Add specific column names from indicator calculations if they are hardcoded elsewhere
# e.g. MACD_HISTOGRAM = "macd_histogram"

# SR Level Types
SR_SUPPORT = "support"
SR_RESISTANCE = "resistance"

# Timeframes - Keeping existing string usage for now, but could be constants
# TF_M15 = "M15"
# TF_H1 = "H1"
# TF_H4 = "H4"
# TF_D1 = "D1"

# Strategy settings (under global_settings.strategy)
CONFIG_STRATEGY_TYPE = "type"
CONFIG_STRATEGY_MIN_SIGNAL_STRENGTH = "min_signal_strength"

# Cache related
CACHE_PREFIX_CONFIG = "config" # Used in config_manager.py

# MT5 Trade Action types (if used explicitly beyond mt5.TRADE_ACTION_DEAL etc.)
# TRADE_ACTION_DEAL = "TRADE_ACTION_DEAL" (already mt5.TRADE_ACTION_DEAL)

# Default values if needed, though usually handled by .get(key, default_value)
DEFAULT_MAGIC_NUMBER = 123456
DEFAULT_DEVIATION = 10
DEFAULT_MAX_SLIPPAGE_POINTS = 10
DEFAULT_PAPER_TRADING = False # A safer default if not present
DEFAULT_LOGGING_LEVEL = "INFO"

# File paths
LOGS_DIR = "logs" # Example if used for constructing log paths

# Position Info Keys (if accessing position dicts with string keys)
POSITION_TICKET = "ticket"
POSITION_SYMBOL = "symbol"
POSITION_TYPE = "type" # Be careful if this clashes with mt5.POSITION_TYPE_*
POSITION_SL = "sl"
POSITION_TP = "tp"
POSITION_OPEN_PRICE = "open_price"
POSITION_VOLUME = "volume"
POSITION_MAGIC = "magic"

# Request keys (for MT5 order_send)
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
REQUEST_POSITION = "position" # For closing/modifying by ticket

# Retcodes (MT5 TRADE_RETCODE_*)
RETCODE_DONE = 10009 # mt5.TRADE_RETCODE_DONE, useful if mt5 is not always imported where checked

# String literals used in logging messages or comments
PAPER_TRADE_COMMENT_PREFIX = "Paper trade"
LOG_MSG_ORDER_OPENED = "Order opened successfully"
LOG_MSG_ORDER_FAILED = "Order failed"
# ... and so on. This can be extensive. Start with the most common ones.

# Keys for 'changes' dict in ConfigManager hot-reload
RELOAD_CHANGES_LOGGING_LEVEL = "logging_level"
RELOAD_CHANGES_GLOBAL_SETTINGS = "global_settings"
RELOAD_CHANGES_SYMBOLS = "symbols"
RELOAD_CHANGES_LOGGING_FILE_REQUIRES_RESTART = "logging_file_changed_requires_restart"
# ... and other keys from _identify_changes

# Misc
PRIMARY_TIMEFRAME_KEY = "primary" # If used for dict access like self.config.get('timeframes', {}).get('primary', 'M30')

# File suffixes
YAML_SUFFIX_LOWER = ".yaml"
YAML_SUFFIX_YML = ".yml"
