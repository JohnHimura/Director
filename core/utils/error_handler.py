"""
Utility module for error handling and retry mechanisms.
"""

import logging
import time
import threading
from typing import Callable, TypeVar, Any, Optional, Dict, List, Union, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')

class TradingError(Exception):
    """Base exception class for all trading related errors."""
    pass

class ConnectionError(TradingError):
    """Exception raised when connection to trading platform fails."""
    pass

class TimeoutError(TradingError):
    """Exception raised when a trading operation times out."""
    pass

class ValidationError(TradingError):
    """Exception raised when validation of parameters fails."""
    pass

class OperationError(TradingError):
    """Exception raised when a trading operation fails."""
    pass

class DataError(TradingError):
    """Exception raised when there is an issue with trading data."""
    pass

def retry(
    max_retries: int = 3, 
    delay: float = 2.0, 
    backoff_factor: float = 1.5,
    exceptions: Tuple[Exception, ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for increasing delay between retries
        exceptions: Tuple of exceptions that should trigger a retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}")
            
            # If all retries failed, re-raise the last exception
            if last_exception:
                raise last_exception
            
            # This should never happen, but added for type checking completeness
            raise RuntimeError("Unexpected error in retry mechanism")
        
        return wrapper
    
    return decorator

def validate_parameters(required_params: List[str]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to validate required parameters in function calls.
    
    Args:
        required_params: List of required parameter names
        
    Returns:
        Decorated function with parameter validation
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get function parameter names
            import inspect
            sig = inspect.signature(func)
            
            # Check for missing parameters
            missing_params = []
            for param_name in required_params:
                if param_name not in kwargs:
                    # Check if it's a positional parameter
                    param_found = False
                    for i, param in enumerate(sig.parameters.values()):
                        if param.name == param_name and i < len(args):
                            param_found = True
                            break
                    
                    if not param_found:
                        missing_params.append(param_name)
            
            if missing_params:
                raise ValidationError(f"Missing required parameters: {', '.join(missing_params)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def safe_operation(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to safely execute operations with proper logging and error handling.
    
    Args:
        operation_name: Name of the operation for logging purposes
        
    Returns:
        Decorated function with enhanced logging and error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger.debug(f"Starting operation: {operation_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Operation completed successfully: {operation_name}")
                return result
            except Exception as e:
                logger.error(f"Operation failed: {operation_name}. Error: {str(e)}")
                raise OperationError(f"Operation '{operation_name}' failed: {str(e)}") from e
        
        return wrapper
    
    return decorator

def timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add timeout to function calls.
    
    Args:
        seconds: Maximum execution time in seconds
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import platform
            
            # Check if running on Windows
            if platform.system() == 'Windows':
                return _timeout_windows(func, seconds, *args, **kwargs)
            else:
                return _timeout_unix(func, seconds, *args, **kwargs)
        
        return wrapper
    
    return decorator 

def _timeout_unix(func: Callable[..., T], seconds: float, *args: Any, **kwargs: Any) -> T:
    """Implement timeout for Unix-based systems using signal."""
    import signal
    
    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")
    
    # Set the timeout handler
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    
    # Set the timeout alarm
    signal.alarm(int(seconds))
    
    try:
        result = func(*args, **kwargs)
    finally:
        # Cancel the alarm and restore the original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
    
    return result

def _timeout_windows(func: Callable[..., T], seconds: float, *args: Any, **kwargs: Any) -> T:
    """Implement timeout for Windows using threading.Timer."""
    result = None
    exception = None
    completed = False
    
    def target():
        nonlocal result, exception, completed
        try:
            result = func(*args, **kwargs)
            completed = True
        except Exception as e:
            exception = e
    
    # Start function in a thread
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete or timeout
    thread.join(timeout=seconds)
    
    if not completed:
        raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")
    
    if exception:
        raise exception
    
    return result 