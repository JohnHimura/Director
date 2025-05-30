"""
Logging configuration for the trading bot.
"""

import os
import logging
import logging.config
import logging.handlers
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback
import threading # Import threading

from .utils.error_handler import retry, safe_operation
from .config_manager import ConfigManager, ConfigError
from . import constants as C # Import constants
from pythonjsonlogger import jsonlogger # Import JsonFormatter

logger = logging.getLogger(__name__)

# Create a thread-local context for storing correlation_id
_log_context = threading.local()

class ContextFilter(logging.Filter):
    """
    A logging filter that injects context information (e.g., correlation_id)
    into the log record.
    """
    def filter(self, record):
        record.correlation_id = getattr(_log_context, 'correlation_id', 'N/A') # Default to N/A if not set
        return True

class LoggerManager:
    """Manager class for logger configuration and operations."""
    
    def __init__(self):
        """Initialize the logger manager."""
        self.initialized = False
        self.log_config = {}
        self.root_logger = logging.getLogger()
        self.handlers = {}
    
    @safe_operation("setup_logging")
    def setup_logging(self, config_path: str = "config.json") -> None:
        """
        Configure logging based on the configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ConfigError: If error in configuration or setup
        """
        try:
            # Load configuration
            # config_path can also be a ConfigManager instance if already initialized
            if isinstance(config_path, ConfigManager):
                config = config_path
            else:
                config = ConfigManager(config_path)
            self.log_config = config.get_logging_config() # This returns a dict
            
            # Ensure log directory exists
            default_log_file = Path(C.LOGS_DIR) / "trading_bot.log" # Use constant for dir
            log_file = Path(self.log_config.get(C.CONFIG_LOGGING_FILE, default_log_file))
            log_dir = log_file.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Get format and datefmt from config, or use defaults
            # Add correlation_id to the default format string
            default_format_str = "%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(module)s %(funcName)s %(lineno)d %(message)s"
            log_format_str = self.log_config.get(C.CONFIG_LOGGING_FORMAT, default_format_str)
            date_format_str = self.log_config.get("datefmt", "%Y-%m-%d %H:%M:%S")

            # Create JsonFormatter
            formatter = jsonlogger.JsonFormatter(
                fmt=log_format_str,
                datefmt=date_format_str,
                rename_fields={"levelname": "level", "asctime": "timestamp"}
            )

            # Create and add context filter
            context_filter = ContextFilter()
            
            # Configure root logger
            self.root_logger.setLevel(logging.DEBUG)
            
            for handler in self.root_logger.handlers[:]:
                self.root_logger.removeHandler(handler)
            
            console_handler = logging.StreamHandler()
            console_level_str = self.log_config.get("console_level", C.DEFAULT_LOGGING_LEVEL)
            console_level = self._get_log_level(console_level_str)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(context_filter) # Add filter to handler
            self.root_logger.addHandler(console_handler)
            self.handlers["console"] = console_handler
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.log_config.get(C.CONFIG_LOGGING_MAX_BYTES, 10*1024*1024),
                backupCount=self.log_config.get(C.CONFIG_LOGGING_BACKUP_COUNT, 5),
                encoding='utf-8'
            )
            file_level_str = self.log_config.get("file_level", C.DEFAULT_LOGGING_LEVEL)
            file_level = self._get_log_level(file_level_str)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(context_filter) # Add filter to handler
            self.root_logger.addHandler(file_handler)
            self.handlers["file"] = file_handler
            
            # "module_levels" is not a standard constant
            module_levels = self.log_config.get("module_levels", {})
            for module_name, level in module_levels.items():
                self.set_module_log_level(module_name, level)
            
            root_level_str = self.log_config.get(C.CONFIG_LOGGING_LEVEL, C.DEFAULT_LOGGING_LEVEL)
            root_level = self._get_log_level(root_level_str)
            self.root_logger.setLevel(root_level)
            
            self.initialized = True
            logger.info("Logging configured successfully")
            
        except Exception as e:
            # Create basic fallback logging in case of error
            self._setup_fallback_logging()
            logger.error(f"Error setting up logging: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _setup_fallback_logging(self) -> None:
        """Set up basic fallback logging if main configuration fails."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logger.info("Using fallback logging configuration")
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        """
        Convert level string to logging level constant.
        
        Args:
            level: Level string or integer
            
        Returns:
            Logging level constant
        """
        if isinstance(level, int):
            return level
            
        try:
            return getattr(logging, level.upper())
        except (AttributeError, TypeError):
            logger.warning(f"Invalid log level '{level}', defaulting to INFO")
            return logging.INFO
    
    def set_global_log_level(self, level: Union[str, int]) -> None:
        """
        Set the global logging level.
        
        Args:
            level: New logging level
        """
        level_value = self._get_log_level(level)
        self.root_logger.setLevel(level_value)
        logger.info(f"Global log level set to {logging.getLevelName(level_value)}")
    
    def set_module_log_level(self, module_name: str, level: Union[str, int]) -> None:
        """
        Set logging level for a specific module.
        
        Args:
            module_name: Name of the module
            level: New logging level
        """
        module_logger = logging.getLogger(module_name)
        level_value = self._get_log_level(level)
        module_logger.setLevel(level_value)
        logger.debug(f"Log level for {module_name} set to {logging.getLevelName(level_value)}")
    
    def set_handler_log_level(self, handler_name: str, level: Union[str, int]) -> None:
        """
        Set logging level for a specific handler.
        
        Args:
            handler_name: Name of the handler (e.g., 'console', 'file')
            level: New logging level
        """
        if handler_name in self.handlers:
            level_value = self._get_log_level(level)
            self.handlers[handler_name].setLevel(level_value)
            logger.debug(f"Log level for {handler_name} handler set to {logging.getLevelName(level_value)}")
        else:
            logger.warning(f"Handler '{handler_name}' not found")
    
    @retry(max_retries=2, delay=0.5)
    def add_file_handler(self, filename: str, level: str = "DEBUG", max_bytes: int = 10*1024*1024, backup_count: int = 5) -> None:
        """
        Add a new file handler.
        
        Args:
            filename: Path to log file
            level: Logging level
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        if filename in self.handlers:
            logger.warning(f"Handler for {filename} already exists")
            return
            
        try:
            # Ensure directory exists
            file_path = Path(filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create handler
            handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            # Set level and formatter
            level_value = self._get_log_level(level)
            handler.setLevel(level_value)
            
            # Create a JsonFormatter for the new handler
            default_format_str_dynamic = "%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(module)s %(funcName)s %(lineno)d %(message)s"
            log_format_str = self.log_config.get(C.CONFIG_LOGGING_FORMAT, default_format_str_dynamic)
            date_format_str = self.log_config.get("datefmt", "%Y-%m-%d %H:%M:%S")

            current_formatter = jsonlogger.JsonFormatter(
                fmt=log_format_str,
                datefmt=date_format_str,
                rename_fields={"levelname": "level", "asctime": "timestamp"}
            )
            handler.setFormatter(current_formatter)

            # Add context filter to the new handler as well
            context_filter = ContextFilter()
            handler.addFilter(context_filter)
            
            # Add to root logger
            self.root_logger.addHandler(handler)
            self.handlers[filename] = handler
            
            logger.info(f"Added file handler for {filename}")
        except Exception as e:
            logger.error(f"Error adding file handler for {filename}: {str(e)}")
            raise
    
    def remove_handler(self, handler_name: str) -> None:
        """
        Remove a handler by name.
        
        Args:
            handler_name: Name of the handler to remove
        """
        if handler_name in self.handlers:
            handler = self.handlers[handler_name]
            self.root_logger.removeHandler(handler)
            del self.handlers[handler_name]
            logger.info(f"Removed handler: {handler_name}")
        else:
            logger.warning(f"Handler '{handler_name}' not found")

# Create a singleton instance
logger_manager = LoggerManager()

# Allow passing ConfigManager instance directly to setup_logging
# Also, make _log_context accessible for main_bot to set it.
def get_log_context():
    return _log_context

def setup_logging(config_manager: Optional[ConfigManager] = None, config_path: str = "config.json") -> None:
    """
    Configure logging.
    
    Args:
        config_manager: Optional ConfigManager instance.
        config_path: Path to the configuration file, used if config_manager is None.
    """
    if config_manager:
        logger_manager.setup_logging(config_path=config_manager)
    else:
        logger_manager.setup_logging(config_path=config_path)

def set_global_log_level(level: Union[str, int]) -> None:
    """
    Set the global logging level.
    
    Args:
        level: New logging level
    """
    logger_manager.set_global_log_level(level)

def set_module_log_level(module_name: str, level: Union[str, int]) -> None:
    """
    Set logging level for a specific module.
    
    Args:
        module_name: Name of the module
        level: New logging level
    """
    logger_manager.set_module_log_level(module_name, level)

def add_file_handler(filename: str, level: str = "DEBUG", max_bytes: int = 10*1024*1024, backup_count: int = 5) -> None:
    """
    Add a new file handler.
    
    Args:
        filename: Path to log file
        level: Logging level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
    """
    logger_manager.add_file_handler(filename, level, max_bytes, backup_count)
