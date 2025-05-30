"""
Configuration manager for loading and accessing trading bot settings.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import yaml
import copy

from .utils.error_handler import ValidationError, DataError, retry
from .utils.cache_manager import cache_manager, cached

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigManager:
    """Manages loading and accessing configuration settings."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration JSON/YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
        
        # Cache TTL (5 minutes by default)
        self.cache_ttl = 300.0
    
    @retry(max_retries=3, delay=1.0, exceptions=(IOError, json.JSONDecodeError))
    def _load_config(self) -> Dict[str, Any]:
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
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:  # Default to JSON
                    return json.load(f)
                    
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {str(e)}")
    
    def _validate_config(self) -> None:
        """
        Perform basic validation of the configuration.
        
        Raises:
            ConfigError: If required sections or settings are missing
        """
        required_sections = ["metatrader5", "global_settings", "logging", "defaults", "symbols"]
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigError(f"Missing required section in config: {section}")
        
        # Validate MT5 settings
        mt5_required = ["path", "server", "login", "password"]
        for key in mt5_required:
            if key not in self._config["metatrader5"]:
                raise ConfigError(f"Missing required MT5 setting: {key}")
                
        # Validate symbol configurations
        if not self._config["symbols"]:
            logger.warning("No symbols configured in configuration file.")
    
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
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False)
                else:  # Default to JSON
                    json.dump(self._config, f, indent=4)
                    
            logger.info(f"Configuration saved to {save_path}")
            
            # Clear cache after save
            self.clear_cache()
        except Exception as e:
            raise ConfigError(f"Error saving configuration to {save_path}: {str(e)}")
    
    @cached(cache_manager, "config", 300.0)
    def get_mt5_config(self) -> Dict[str, Any]:
        """
        Get MetaTrader 5 configuration.
        
        Returns:
            Dict with MT5 configuration
        """
        return copy.deepcopy(self._config["metatrader5"])
    
    @cached(cache_manager, "config", 300.0)
    def get_global_settings(self) -> Dict[str, Any]:
        """
        Get global trading settings.
        
        Returns:
            Dict with global settings
        """
        return copy.deepcopy(self._config["global_settings"])
    
    @cached(cache_manager, "config", 300.0)
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dict with logging configuration
        """
        return copy.deepcopy(self._config["logging"])
    
    @cached(cache_manager, "config", 300.0)
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
        if symbol not in self._config["symbols"]:
            raise ValidationError(f"No configuration found for symbol: {symbol}")
        
        # Start with defaults
        config = {
            "indicators": copy.deepcopy(self._config["defaults"]["indicators"]),
            "sr": copy.deepcopy(self._config["defaults"]["sr"]),
            "risk": copy.deepcopy(self._config["defaults"]["risk"])
        }
        
        # Update with symbol-specific settings
        symbol_config = self._config["symbols"][symbol]
        
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
    
    @cached(cache_manager, "config", 300.0)
    def get_active_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active symbols and their configurations.
        
        Returns:
            Dict of symbol to its configuration
        """
        result = {
            symbol: self.get_symbol_config(symbol)
            for symbol, config in self._config["symbols"].items()
            if config.get("enabled", True)
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
        return self.get_symbol_config(symbol)["indicators"]
    
    def get_sr_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get support/resistance parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with SR parameters
        """
        return self.get_symbol_config(symbol)["sr"]
    
    def get_risk_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get risk management parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with risk parameters
        """
        return self.get_symbol_config(symbol)["risk"]
        
    @cached(cache_manager, "config", 300.0)
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
            self._validate_config()
            
            # Save if requested
            if save:
                self.save_config()
                
            logger.info("Configuration updated successfully")
        except Exception as e:
            raise ConfigError(f"Error updating configuration: {str(e)}")
            
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        # Clear entries with config prefix
        cache_keys = cache_manager.get_stats()['keys']
        for key in cache_keys:
            if key.startswith('config:'):
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
