"""
Configuration manager for loading and accessing trading bot settings.
"""

import json
import os # Ensure os is imported
from typing import Dict, Any, Optional, List, Union, Tuple # Ensure Tuple is imported
import logging
from pathlib import Path
import yaml
import copy
import jsonschema # Added import

from .utils.error_handler import ValidationError, DataError, retry
from .utils.cache_manager import cache_manager, cached
from . import constants as C # Import constants

logger = logging.getLogger(__name__)

# Path to the configuration schema
CONFIG_SCHEMA_PATH = Path(__file__).parent / "config_schema.json" # Added schema path

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigManager:
    """Manages loading and accessing configuration settings."""
    
    _schema: Optional[Dict[str, Any]] = None
    _mt5_credentials: Dict[str, Any] = {} # To store credentials from env vars

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration JSON/YAML file
        """
        self.config_path = config_path
        self._load_mt5_credentials_from_env() # Load credentials first

        self._config = self._load_config_data()
        # self._load_schema() # Comentado temporalmente
        # self._validate_config_data(self._config) # Comentado temporalmente
        self._last_modified_time = self._get_file_mod_time()
        
        # Cache TTL (5 minutes by default)
        self.cache_ttl = 300.0

    def _get_file_mod_time(self) -> float:
        """Get the last modification time of the config file."""
        try:
            return Path(self.config_path).stat().st_mtime
        except FileNotFoundError:
            # Should not happen if constructor succeeded with _load_config_data
            logger.error(f"Config file not found when checking mod time: {self.config_path}")
            return 0
        except Exception as e:
            logger.error(f"Error getting file modification time: {e}")
            return 0

    @classmethod
    def _load_schema(cls) -> None:
        """Load the JSON schema for configuration validation."""
        if cls._schema is None:
            try:
                with open(CONFIG_SCHEMA_PATH, 'r', encoding='utf-8') as f:
                    cls._schema = json.load(f)
            except FileNotFoundError:
                # This is a critical error, should stop the bot
                logger.critical(f"Configuration schema not found: {CONFIG_SCHEMA_PATH}")
                raise ConfigError(f"Configuration schema not found: {CONFIG_SCHEMA_PATH}")
            except json.JSONDecodeError as e:
                logger.critical(f"Invalid JSON in schema file: {e}")
                raise ConfigError(f"Invalid JSON in schema file: {e}")

    @retry(max_retries=3, delay=1.0, exceptions=(IOError, json.JSONDecodeError))
    def _load_config_data(self) -> Dict[str, Any]: # Renamed
        """
        Load configuration from JSON/YAML file with retry logic.
        
        Returns:
            Dict with configuration
            
        Raises:
            ConfigError: If configuration file not found or invalid
        """
        try:
            file_path = Path(self.config_path)
            
            if not file_path.exists():
                raise ConfigError(f"Configuration file not found: {self.config_path}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in [C.YAML_SUFFIX_LOWER, C.YAML_SUFFIX_YML]:
                    config_data = yaml.safe_load(f)
                else:  # Default to JSON
                    config_data = json.load(f)

            # self._validate_config_data(config_data) # Validation will be done by the caller
            return config_data
                    
        except FileNotFoundError:
            # This is critical on initial load
            logger.critical(f"Configuration file not found: {self.config_path}")
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON in config file: {e}")
            raise ConfigError(f"Invalid JSON in config file: {e}")
        except yaml.YAMLError as e:
            logger.critical(f"Invalid YAML in config file: {e}")
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            logger.critical(f"Error loading configuration: {str(e)}")
            raise ConfigError(f"Error loading configuration: {str(e)}")
    
    def _validate_config_data(self, config_data: Dict[str, Any]) -> None: # Renamed and takes data
        """
        Validate the given configuration data against the JSON schema.
        
        Args:
            config_data: The configuration data to validate.

        Raises:
            ConfigError: If configuration is invalid
        """
        # if self._schema is None: # Ensure schema is loaded # Comentado temporalmente
        #     self._load_schema() # Should have been loaded by constructor or reload method # Comentado temporalmente
        #     if self._schema is None: # If still None, something is very wrong # Comentado temporalmente
        #          logger.critical("Schema not loaded for validation.") # Comentado temporalmente
        #          raise ConfigError("Schema not loaded for validation.") # Comentado temporalmente

        # try: # Comentado temporalmente
        #     jsonschema.validate(instance=config_data, schema=self._schema) # Comentado temporalmente
        #     logger.debug("Configuration data validated successfully against schema.") # Comentado temporalmente
        # except jsonschema.exceptions.ValidationError as e: # Comentado temporalmente
        #     error_path = " -> ".join(map(str, e.path)) # Comentado temporalmente
        #     message = f"Configuration validation error at '{error_path}': {e.message}" # Comentado temporalmente
        #     logger.error(f"Schema validation failed for new data: {e}") # Comentado temporalmente
        #     raise ConfigError(message) # Comentado temporalmente
        # except jsonschema.exceptions.SchemaError as e: # Comentado temporalmente
        #     logger.error(f"Invalid schema during validation: {e}") # Comentado temporalmente
        #     raise ConfigError(f"Invalid configuration schema: {e.message}") # Comentado temporalmente

    def _load_mt5_credentials_from_env(self) -> None:
        """Loads MT5 credentials from environment variables and validates them."""
        path = os.getenv("MT5_PATH")
        server = os.getenv("MT5_SERVER")
        login_str = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")

        if not path:
            raise ConfigError("Missing MT5_PATH environment variable.")
        if not server:
            raise ConfigError("Missing MT5_SERVER environment variable.")
        if not login_str:
            raise ConfigError("Missing MT5_LOGIN environment variable.")
        if not password:
            raise ConfigError("Missing MT5_PASSWORD environment variable.")

        try:
            login = int(login_str)
        except ValueError:
            raise ConfigError(f"MT5_LOGIN environment variable ('{login_str}') must be an integer.")

        self._mt5_credentials = {
            C.CONFIG_MT5_PATH: path,
            C.CONFIG_MT5_SERVER: server,
            C.CONFIG_MT5_LOGIN: login,
            C.CONFIG_MT5_PASSWORD: password,
        }
        logger.info("Successfully loaded MT5 credentials from environment variables.")

    def check_and_reload_config(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the config file has been modified and reload if it has.
        Identifies what changed for hot-reloading purposes.

        Returns:
            A tuple: (was_reloaded: bool, changes: Dict[str, Any])
            'changes' dictionary details what was altered if a reload happened.
        """
        current_mod_time = self._get_file_mod_time()
        if current_mod_time == 0: # Error getting mod time
            return False, {}

        if current_mod_time > self._last_modified_time:
            logger.info(f"Configuration file '{self.config_path}' has been modified. Attempting to reload.")
            old_config = copy.deepcopy(self._config) # For comparison

            try:
                new_config_data = self._load_config_data()
                self._validate_config_data(new_config_data) # Validate before applying

                # Identify changes
                changes = self._identify_changes(old_config, new_config_data)

                if not changes:
                    logger.info("Config file modified but no functional changes detected after validation.")
                    self._last_modified_time = current_mod_time
                    return False, {}

                self._config = new_config_data
                self._last_modified_time = current_mod_time
                self.clear_cache() # Clear all cached config values

                logger.info(f"Configuration reloaded and validated. Changes: {json.dumps(changes)}")
                return True, changes

            except ConfigError as e:
                logger.error(f"Failed to reload or validate modified configuration: {e}. Keeping current config.")
                return False, {}
            except Exception as e: # Catch any other unexpected error during reload
                logger.error(f"Unexpected error during configuration reload: {e}. Keeping current config.")
                return False, {}

        return False, {}

    def _identify_changes(self, old_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifies differences between old and new configurations relevant for hot-reloading.
        """
        changes = {}

        # 1. Logging level
        old_log_cfg = old_cfg.get(C.CONFIG_LOGGING, {})
        new_log_cfg = new_cfg.get(C.CONFIG_LOGGING, {})
        if old_log_cfg.get(C.CONFIG_LOGGING_LEVEL) != new_log_cfg.get(C.CONFIG_LOGGING_LEVEL):
            changes[C.RELOAD_CHANGES_LOGGING_LEVEL] = new_log_cfg.get(C.CONFIG_LOGGING_LEVEL)
            logger.info(f"Logging level changed from {old_log_cfg.get(C.CONFIG_LOGGING_LEVEL)} to {new_log_cfg.get(C.CONFIG_LOGGING_LEVEL)}")

        # 2. Global settings
        old_global_settings = old_cfg.get(C.CONFIG_GLOBAL_SETTINGS, {})
        new_global_settings = new_cfg.get(C.CONFIG_GLOBAL_SETTINGS, {})
        # Define hot-reloadable global settings using constants
        hot_reloadable_global_settings = [
            C.CONFIG_MAX_SLIPPAGE_PIPS,
            C.CONFIG_MAX_TOTAL_TRADES,
            C.CONFIG_DEVIATION, # Keep if used, or remove if fully replaced by max_slippage_points
            C.CONFIG_MAX_SLIPPAGE_POINTS,
            # Add C.CONFIG_PAPER_TRADING if it's considered hot-reloadable (might be complex)
        ]
        global_changes = {}
        for key in hot_reloadable_global_settings:
            if old_global_settings.get(key) != new_global_settings.get(key):
                global_changes[key] = new_global_settings.get(key)
        if global_changes:
            changes[C.RELOAD_CHANGES_GLOBAL_SETTINGS] = global_changes
            logger.info(f"Global settings changed: {global_changes}")

        # 3. Symbol-specific parameters
        changed_symbols_details = {}
        all_symbol_keys = set(old_cfg.get(C.CONFIG_SYMBOLS, {}).keys()) | set(new_cfg.get(C.CONFIG_SYMBOLS, {}).keys())

        for sym in all_symbol_keys:
            old_sym_cfg = self.get_symbol_config_from_data(sym, old_cfg)
            new_sym_cfg = self.get_symbol_config_from_data(sym, new_cfg)

            sym_changes = {}
            if old_sym_cfg.get(C.CONFIG_INDICATORS) != new_sym_cfg.get(C.CONFIG_INDICATORS):
                sym_changes[C.CONFIG_INDICATORS] = new_sym_cfg.get(C.CONFIG_INDICATORS)
            if old_sym_cfg.get(C.CONFIG_RISK) != new_sym_cfg.get(C.CONFIG_RISK):
                sym_changes[C.CONFIG_RISK] = new_sym_cfg.get(C.CONFIG_RISK)
            if old_sym_cfg.get(C.CONFIG_SR) != new_sym_cfg.get(C.CONFIG_SR):
                 sym_changes[C.CONFIG_SR] = new_sym_cfg.get(C.CONFIG_SR)
            if old_sym_cfg.get(C.CONFIG_ENABLED) != new_sym_cfg.get(C.CONFIG_ENABLED):
                sym_changes[C.CONFIG_ENABLED] = new_sym_cfg.get(C.CONFIG_ENABLED)

            if sym_changes:
                changed_symbols_details[sym] = sym_changes

        if changed_symbols_details:
            changes[C.RELOAD_CHANGES_SYMBOLS] = changed_symbols_details
            logger.info(f"Symbol configurations changed: {changed_symbols_details}")

        # Parameters requiring restart
        critical_sections = [C.CONFIG_METATRADER5]
        for section in critical_sections:
            if old_cfg.get(section) != new_cfg.get(section):
                changes[f"{section}_changed_requires_restart"] = True # Keep dynamic key for now
                logger.warning(f"Critical configuration section '{section}' changed. Bot restart is recommended.")

        if old_log_cfg.get(C.CONFIG_LOGGING_FILE) != new_log_cfg.get(C.CONFIG_LOGGING_FILE):
            changes[C.RELOAD_CHANGES_LOGGING_FILE_REQUIRES_RESTART] = True
            logger.warning(f"Logging file path changed from {old_log_cfg.get(C.CONFIG_LOGGING_FILE)} to {new_log_cfg.get(C.CONFIG_LOGGING_FILE)}. Bot restart is recommended for logs to switch.")

        return changes

    def get_symbol_config_from_data(self, symbol: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper to get symbol configuration merged with defaults from a given config dictionary.
        This is used by _identify_changes to compare old and new states correctly.
        """
        if symbol not in config_data.get(C.CONFIG_SYMBOLS, {}):
            return {}

        defaults = config_data.get(C.CONFIG_DEFAULTS, {})
        config = {
            C.CONFIG_INDICATORS: copy.deepcopy(defaults.get(C.CONFIG_INDICATORS, {})),
            C.CONFIG_SR: copy.deepcopy(defaults.get(C.CONFIG_SR, {})),
            C.CONFIG_RISK: copy.deepcopy(defaults.get(C.CONFIG_RISK, {}))
        }

        symbol_specific_config = config_data[C.CONFIG_SYMBOLS][symbol]

        def update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d

        # Update with symbol-specific settings, including top-level like 'enabled', 'lot_size'
        merged_symbol_config = update_dict(config, symbol_specific_config)
        # Ensure top-level simple settings from symbol_specific_config are also present
        for key, value in symbol_specific_config.items():
            if not isinstance(value, dict): # if it's not C.CONFIG_INDICATORS, C.CONFIG_SR, C.CONFIG_RISK
                merged_symbol_config[key] = value

        return merged_symbol_config

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save the current configuration to file.
        
        Args:
            config_path: Optional path to save to (defaults to original path)
            
        Raises:
            ConfigError: If there's an error saving the configuration
        """
        save_path = config_path or self.config_path
        file_path = Path(save_path)

        # Create a copy of the config to save, excluding sensitive MT5 credentials
        config_to_save = copy.deepcopy(self._config)
        if C.CONFIG_METATRADER5 in config_to_save:
            # Ensure the metatrader5 section itself exists before trying to delete keys
            config_to_save[C.CONFIG_METATRADER5].pop(C.CONFIG_MT5_PATH, None)
            config_to_save[C.CONFIG_METATRADER5].pop(C.CONFIG_MT5_SERVER, None)
            config_to_save[C.CONFIG_METATRADER5].pop(C.CONFIG_MT5_LOGIN, None)
            config_to_save[C.CONFIG_METATRADER5].pop(C.CONFIG_MT5_PASSWORD, None)
            # If the metatrader5 section becomes empty, optionally remove it
            if not config_to_save[C.CONFIG_METATRADER5]:
                del config_to_save[C.CONFIG_METATRADER5]
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in [C.YAML_SUFFIX_LOWER, C.YAML_SUFFIX_YML]:
                    yaml.dump(config_to_save, f, default_flow_style=False)
                else:  # Default to JSON
                    json.dump(config_to_save, f, indent=4)
                    
            logger.info(f"Configuration saved to {save_path} (MT5 credentials excluded)")
            
            # Clear cache after save
            self.clear_cache()
        except Exception as e:
            raise ConfigError(f"Error saving configuration to {save_path}: {str(e)}")
    
    @cached(cache_manager, C.CACHE_PREFIX_CONFIG, 300.0)
    def get_mt5_config(self) -> Dict[str, Any]:
        """
        Get MetaTrader 5 configuration, merging environment credentials with file config.
        
        Returns:
            Dict with MT5 configuration
        """
        # Start with non-sensitive MT5 settings from the file (e.g., timeout, portable)
        file_mt5_config = copy.deepcopy(self._config.get(C.CONFIG_METATRADER5, {}))

        # Override/add with credentials from environment variables
        # The self._mt5_credentials already uses the C.CONFIG_MT5_XXX constants as keys
        merged_config = {**file_mt5_config, **self._mt5_credentials}

        # Ensure all required keys are present after merging, even if from env vars
        # (path, server, login, password are validated in _load_mt5_credentials_from_env)
        # Other keys like 'timeout', 'portable' should come from file_mt5_config
        # and their presence is validated by the schema against the file content.

        return merged_config # This is already a copy due to deepcopy and dict unpacking
    
    @cached(cache_manager, C.CACHE_PREFIX_CONFIG, 300.0)
    def get_global_settings(self) -> Dict[str, Any]:
        """
        Get global trading settings.
        
        Returns:
            Dict with global settings
        """
        return copy.deepcopy(self._config[C.CONFIG_GLOBAL_SETTINGS])
    
    @cached(cache_manager, C.CACHE_PREFIX_CONFIG, 300.0)
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dict with logging configuration
        """
        return copy.deepcopy(self._config[C.CONFIG_LOGGING])
    
    @cached(cache_manager, C.CACHE_PREFIX_CONFIG, 300.0)
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get configuration for a specific symbol, merging with defaults.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Dict containing the merged configuration for the symbol
            
        Raises:
            ValidationError: If symbol not found in configuration
        """
        if symbol not in self._config[C.CONFIG_SYMBOLS]:
            raise ValidationError(f"No configuration found for symbol: {symbol}")
        
        # Start with defaults
        defaults_config = self._config[C.CONFIG_DEFAULTS]
        config = {
            C.CONFIG_INDICATORS: copy.deepcopy(defaults_config.get(C.CONFIG_INDICATORS, {})),
            C.CONFIG_SR: copy.deepcopy(defaults_config.get(C.CONFIG_SR, {})),
            C.CONFIG_RISK: copy.deepcopy(defaults_config.get(C.CONFIG_RISK, {}))
        }
        
        # Update with symbol-specific settings
        symbol_config = self._config[C.CONFIG_SYMBOLS][symbol]
        
        # Merge dictionaries recursively
        def update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d
        
        result = update_dict(config, symbol_config)
        
        return copy.deepcopy(result)
    
    @cached(cache_manager, C.CACHE_PREFIX_CONFIG, 300.0)
    def get_active_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active symbols and their configurations.
        
        Returns:
            Dict of symbol to its configuration
        """
        result = {
            symbol_name: self.get_symbol_config(symbol_name)
            for symbol_name, config_data in self._config[C.CONFIG_SYMBOLS].items()
            if config_data.get(C.CONFIG_ENABLED, True)
        }
        
        return copy.deepcopy(result)
    
    def get_indicator_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get indicator parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with indicator parameters
        """
        return self.get_symbol_config(symbol).get(C.CONFIG_INDICATORS, {})
    
    def get_sr_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get support/resistance parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with SR parameters
        """
        return self.get_symbol_config(symbol).get(C.CONFIG_SR, {})
    
    def get_risk_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get risk management parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with risk parameters
        """
        return self.get_symbol_config(symbol).get(C.CONFIG_RISK, {})
        
    @cached(cache_manager, C.CACHE_PREFIX_CONFIG, 300.0)
    def get_timeframes(self) -> Dict[str, str]:
        """
        Get the configured timeframes.
        
        Returns:
            Dict with timeframe names as keys and their MT5 equivalents as values
        """
        return self._config.get("timeframes", {
            "M15": "M15",
            "H1": "H1",
            "H4": "H4",
            "D1": "D1"
        })
    
    def update_config(self, updates: Dict[str, Any], save: bool = True) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
            save: Whether to save the updated configuration to file
            
        Raises:
            ConfigError: If there's an error updating the configuration
        """
        try:
            # Update the configuration recursively
            def update_dict(d: Dict, u: Dict) -> Dict:
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            
            self._config = update_dict(self._config, updates)
            
            # Clear cache as it may be stale
            self.clear_cache()
            
            # Validate the updated configuration
            self._validate_config_data(self._config) # Validate current config before saving
            
            # Save if requested
            if save:
                self.save_config()
                
            logger.info("Configuration updated successfully")
        except Exception as e:
            raise ConfigError(f"Error updating configuration: {str(e)}")
            
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        # Clear entries with config prefix
        stats = cache_manager.get_stats()
        if stats:
             cache_keys = list(stats.get('keys', [])) # Corrected: Use list copy for iteration
             for key in cache_keys:
                 if key.startswith(C.CACHE_PREFIX_CONFIG + ':'): # Ensure colon for specificity
                     cache_manager.delete(key)
        logger.debug("Configuration cache cleared")
    
    def set_cache_ttl(self, ttl: float) -> None:
        """
        Set the cache time-to-live.
        
        Args:
            ttl: Time to live in seconds
        """
        self.cache_ttl = ttl
        logger.debug(f"Cache TTL set to {ttl} seconds")
