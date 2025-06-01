import pytest
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from core.logging_setup import LoggerManager, ContextFilter, _log_context, setup_logging
from core.config_manager import ConfigManager, ConfigError
from core import constants as C
from pythonjsonlogger import jsonlogger

@pytest.fixture
def mock_config_manager_for_logging(tmp_path):
    mock_cm = MagicMock(spec=ConfigManager)
    log_file_path = tmp_path / "test_bot.log"
    mock_cm.get_logging_config.return_value = {
        C.CONFIG_LOGGING_LEVEL: "INFO",
        C.CONFIG_LOGGING_FILE: str(log_file_path),
        C.CONFIG_LOGGING_MAX_BYTES: 1024, # Small for testing
        C.CONFIG_LOGGING_BACKUP_COUNT: 1,
        C.CONFIG_LOGGING_FORMAT: "%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(message)s",
        "datefmt": "%Y-%m-%dT%H:%M:%S",
        "console_level": "DEBUG",
        "file_level": "INFO",
        "module_levels": {
            "test_module": "WARNING"
        }
    }
    return mock_cm

@pytest.fixture
def logger_manager_instance():
    # Reset shared state for logger_manager singleton if tests modify it globally
    # This might involve re-importing or specific reset methods if it were a complex singleton.
    # For this LoggerManager, creating a new instance for each test is cleaner.
    # However, the module uses a global `logger_manager = LoggerManager()`.
    # To test it cleanly, we might need to mock the global instance or test functions directly.
    # For this test, we'll create a new instance and work with it.
    # The global functions setup_logging, etc., call methods on the module-level logger_manager.
    # We can patch that module-level instance.

    # For simplicity, we'll test the class methods directly by instantiating LoggerManager.
    # Testing the global functions would require patching the global `logger_manager` instance.
    manager = LoggerManager()
    # Ensure it's clean before each test that uses this fixture
    manager.initialized = False
    manager.log_config = {}
    manager.root_logger = logging.getLogger(f"test_root_{os.urandom(4).hex()}") # Unique logger per test
    manager.root_logger.handlers = [] # Clear handlers
    manager.handlers = {}
    return manager

def test_logger_manager_init(logger_manager_instance: LoggerManager):
    assert not logger_manager_instance.initialized
    assert logger_manager_instance.log_config == {}
    assert len(logger_manager_instance.root_logger.handlers) == 0

def test_get_log_level(logger_manager_instance: LoggerManager):
    assert logger_manager_instance._get_log_level("DEBUG") == logging.DEBUG
    assert logger_manager_instance._get_log_level("INFO") == logging.INFO
    assert logger_manager_instance._get_log_level("WARNING") == logging.WARNING
    assert logger_manager_instance._get_log_level("ERROR") == logging.ERROR
    assert logger_manager_instance._get_log_level("CRITICAL") == logging.CRITICAL
    assert logger_manager_instance._get_log_level(logging.DEBUG) == logging.DEBUG
    assert logger_manager_instance._get_log_level("INVALID_LEVEL") == logging.INFO # Defaults to INFO

@patch('core.logging_setup.Path.mkdir')
@patch('core.logging_setup.logging.handlers.RotatingFileHandler')
@patch('core.logging_setup.logging.StreamHandler')
def test_setup_logging_success(mock_stream_handler, mock_rotating_file_handler, mock_mkdir,
                               logger_manager_instance: LoggerManager, mock_config_manager_for_logging, tmp_path):
    # Ensure the global logger_manager used by the setup_logging function is our instance
    with patch('core.logging_setup.logger_manager', logger_manager_instance):
        setup_logging(config_manager=mock_config_manager_for_logging)

    assert logger_manager_instance.initialized
    mock_config_manager_for_logging.get_logging_config.assert_called_once()
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Check console handler
    mock_stream_handler.assert_called_once()
    console_h = mock_stream_handler.return_value
    console_h.setLevel.assert_called_with(logging.DEBUG) # From "console_level" in mock config
    # Check that addFilter was called with a ContextFilter instance
    assert console_h.addFilter.call_count == 1
    assert isinstance(console_h.addFilter.call_args[0][0], ContextFilter)
    assert isinstance(console_h.formatter, jsonlogger.JsonFormatter)

    # Check file handler
    mock_rotating_file_handler.assert_called_once()
    file_h = mock_rotating_file_handler.return_value
    expected_log_file = tmp_path / "test_bot.log"
    args, kwargs = mock_rotating_file_handler.call_args
    assert kwargs['filename'] == expected_log_file
    assert kwargs['maxBytes'] == 1024
    assert kwargs['backupCount'] == 1
    file_h.setLevel.assert_called_with(logging.INFO) # From "file_level" in mock config
    assert any(isinstance(f, ContextFilter) for f in file_h.filters)
    assert isinstance(file_h.formatter, jsonlogger.JsonFormatter)

    # Check root logger level
    assert logger_manager_instance.root_logger.level == logging.INFO # From "level" in mock config

    # Check module level
    test_module_logger = logging.getLogger("test_module")
    assert test_module_logger.level == logging.WARNING


@patch('core.logging_setup.LoggerManager._setup_fallback_logging')
def test_setup_logging_config_error(mock_fallback, logger_manager_instance: LoggerManager):
    mock_cm_bad = MagicMock(spec=ConfigManager)
    mock_cm_bad.get_logging_config.side_effect = ConfigError("Failed to load logging config")

    with patch('core.logging_setup.logger_manager', logger_manager_instance):
        setup_logging(config_manager=mock_cm_bad)

    assert not logger_manager_instance.initialized # Should fail initialization
    mock_fallback.assert_called_once()

def test_context_filter():
    context_filter = ContextFilter()
    record = logging.LogRecord(name='test', level=logging.INFO, pathname='path', lineno=10, msg='message', args=(), exc_info=None, func='func')

    # Test with correlation_id set
    _log_context.correlation_id = "test_id_123"
    context_filter.filter(record)
    assert record.correlation_id == "test_id_123"

    # Test with correlation_id not set (should default to 'N/A')
    del _log_context.correlation_id
    record_no_id = logging.LogRecord(name='test', level=logging.INFO, pathname='path', lineno=10, msg='message', args=(), exc_info=None, func='func')
    context_filter.filter(record_no_id)
    assert record_no_id.correlation_id == "N/A"

def test_set_global_log_level(logger_manager_instance: LoggerManager):
    logger_manager_instance.set_global_log_level("DEBUG")
    assert logger_manager_instance.root_logger.level == logging.DEBUG
    logger_manager_instance.set_global_log_level(logging.WARNING)
    assert logger_manager_instance.root_logger.level == logging.WARNING

def test_set_module_log_level(logger_manager_instance: LoggerManager):
    logger_manager_instance.set_module_log_level("my_module", "ERROR")
    assert logging.getLogger("my_module").level == logging.ERROR

def test_set_handler_log_level(logger_manager_instance: LoggerManager):
    # First, setup a dummy handler to test against
    mock_handler = MagicMock(spec=logging.Handler)
    logger_manager_instance.handlers['dummy_handler'] = mock_handler

    logger_manager_instance.set_handler_log_level('dummy_handler', "CRITICAL")
    mock_handler.setLevel.assert_called_once_with(logging.CRITICAL)

    logger_manager_instance.set_handler_log_level('nonexistent_handler', "DEBUG") # Should log a warning

@patch('core.logging_setup.Path.mkdir')
@patch('core.logging_setup.logging.handlers.RotatingFileHandler')
def test_add_file_handler(mock_rfh, mock_mkdir, logger_manager_instance: LoggerManager, tmp_path):
    log_file = tmp_path / "another_test.log"

    # Ensure log_config is populated for formatter creation within add_file_handler
    logger_manager_instance.log_config = {"datefmt": "%Y-%m-%d", C.CONFIG_LOGGING_FORMAT: "%(message)s"}

    logger_manager_instance.add_file_handler(str(log_file), level="WARNING", max_bytes=2048, backup_count=2)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_rfh.assert_called_once()
    args, kwargs = mock_rfh.call_args
    assert kwargs['filename'] == log_file
    assert kwargs['maxBytes'] == 2048
    assert kwargs['backupCount'] == 2

    new_handler_instance = mock_rfh.return_value
    new_handler_instance.setLevel.assert_called_with(logging.WARNING)
    # Check that setFormatter was called with a JsonFormatter instance
    assert new_handler_instance.setFormatter.call_count == 1
    formatter_arg = new_handler_instance.setFormatter.call_args[0][0]
    assert isinstance(formatter_arg, jsonlogger.JsonFormatter)
    # Check that addFilter was called with a ContextFilter instance
    assert new_handler_instance.addFilter.call_count == 1
    assert isinstance(new_handler_instance.addFilter.call_args[0][0], ContextFilter)
    assert str(log_file) in logger_manager_instance.handlers
    assert new_handler_instance in logger_manager_instance.root_logger.handlers # Check actual instance

def test_remove_handler(logger_manager_instance: LoggerManager):
    mock_handler = MagicMock(spec=logging.Handler)
    logger_manager_instance.handlers['to_remove'] = mock_handler
    logger_manager_instance.root_logger.addHandler(mock_handler)

    assert 'to_remove' in logger_manager_instance.handlers
    assert mock_handler in logger_manager_instance.root_logger.handlers # Check membership

    logger_manager_instance.remove_handler('to_remove')

    assert 'to_remove' not in logger_manager_instance.handlers
    assert mock_handler not in logger_manager_instance.root_logger.handlers # Check membership

    logger_manager_instance.remove_handler('nonexistent') # Should log warning
