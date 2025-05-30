"""
Utility module for caching with expiration.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, TypeVar, Generic

logger = logging.getLogger(__name__)

# Type variable for cached values
T = TypeVar('T')

class CachedValue(Generic[T]):
    """Class to store a cached value with expiration time."""
    
    def __init__(self, value: T, ttl: float):
        """
        Initialize the cached value.
        
        Args:
            value: Value to cache
            ttl: Time to live in seconds
        """
        self.value = value
        self.expiration = time.time() + ttl
    
    def is_expired(self) -> bool:
        """
        Check if the cached value has expired.
        
        Returns:
            True if expired, False otherwise
        """
        return time.time() > self.expiration

class CacheManager:
    """Manager for cache with expiration."""
    
    def __init__(self, default_ttl: float = 300.0):
        """
        Initialize the cache manager.
        
        Args:
            default_ttl: Default time to live in seconds (5 minutes)
        """
        self._cache: Dict[str, CachedValue] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self._cache:
            return None
            
        cached = self._cache[key]
        if cached.is_expired():
            # Remove expired entry
            del self._cache[key]
            return None
            
        return cached.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        ttl_value = ttl if ttl is not None else self.default_ttl
        self._cache[key] = CachedValue(value, ttl_value)
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
    
    def clear_expired(self) -> int:
        """
        Clear all expired items from the cache.
        
        Returns:
            Number of items removed
        """
        expired_keys = [key for key, cached in self._cache.items() if cached.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Clear expired entries first
        self.clear_expired()
        
        return {
            'size': len(self._cache),
            'keys': list(self._cache.keys())
        }
    
    def __len__(self) -> int:
        """Get cache size (not counting expired items)."""
        # Clear expired entries first
        self.clear_expired()
        return len(self._cache)

# Create a decorator for caching function results
def cached(cache_manager: CacheManager, key_prefix: str, ttl: Optional[float] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_manager: Cache manager instance
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Create cache key from function name, args and kwargs
            key_parts = [key_prefix, func.__name__]
            
            # Add args and kwargs to key
            if args:
                key_parts.extend([str(arg) for arg in args])
            
            # Add sorted kwargs to key
            if kwargs:
                for k in sorted(kwargs.keys()):
                    key_parts.append(f"{k}={kwargs[k]}")
            
            cache_key = ":".join(key_parts)
            
            # Check cache first
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        return wrapper
    
    return decorator

# Singleton instance
cache_manager = CacheManager() 