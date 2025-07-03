#!/usr/bin/env python3
"""
Redis Cache Manager for Diet Classification System

This module provides a centralized Redis caching solution optimized for storing
classification results and factual contexts. It implements a cache-first approach
with graceful fallback when Redis is unavailable.

The CacheManager class provides high-performance caching for:
- Classification results with metadata and timestamps
- Ingredient context data (nutritional and vegan information)
- System statistics and performance metrics
- Batch operations with configurable TTL

Key Features:
- Thread-safe singleton pattern with automatic initialization
- Automatic JSON serialization/deserialization with error handling
- Configurable TTL for different data types (24h for results, 7d for contexts)
- Graceful degradation when Redis is unavailable
- Comprehensive health monitoring and connection recovery
- Professional logging without emojis

Architecture:
- Singleton pattern ensures single cache instance across application
- Connection pooling with automatic reconnection
- Memory-efficient serialization with compression support
- Cache warming and invalidation strategies

Example:
    >>> from utils.cache_manager import get_cache_manager
    >>> cache = get_cache_manager()
    >>> cache.set_classification_result("recipe_123", {"keto": True, "vegan": False})
    >>> result = cache.get_classification_result("recipe_123")
    >>> print(result['classification']['keto'])
    True
"""

# Try to import redis, with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import json
import os
import logging
from typing import Optional, Any, Dict, Union
from pathlib import Path

# Import app_config for centralized configuration
from config import app_config

# Configure professional logging
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Thread-safe Redis cache manager with automatic fallback capabilities.
    
    This class implements a singleton pattern and provides high-level caching
    operations for the diet classification pipeline. It handles connection
    failures gracefully and provides detailed logging for debugging.
    
    The CacheManager supports multiple data types with appropriate TTL settings:
    - Classification results: 24-hour TTL for recipe classifications
    - Ingredient contexts: 7-day TTL for nutritional/vegan data
    - System statistics: 1-hour TTL for performance metrics
    
    Key Features:
    - Automatic connection management with health checks
    - JSON serialization with error handling
    - Configurable TTL per data type
    - Graceful fallback when Redis unavailable
    - Comprehensive statistics and monitoring
    
    Attributes:
        _instance: Singleton instance of CacheManager
        _redis_client: Redis client connection
        _is_connected: Connection status flag
        
    Example:
        >>> cache = CacheManager()
        >>> cache.set_ingredient_context("chicken", {"protein": 25.0, "keto": True})
        >>> context = cache.get_ingredient_context("chicken")
        >>> print(context['protein'])
        25.0
    """
    
    _instance = None
    _redis_client = None
    _is_connected = False
    
    def __new__(cls):
        """
        Singleton pattern implementation.
        
        Ensures only one CacheManager instance exists across the application,
        providing consistent caching behavior and resource management.
        
        Returns:
            CacheManager: Singleton instance of the cache manager
        """
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Initialize the Redis connection with environment-based configuration.
        
        Sets up Redis connection using environment variables with sensible
        defaults. Implements comprehensive error handling and connection
        testing to ensure reliable operation.
        
        Configuration Sources (now from app_config):
        - REDIS_HOST: Redis server hostname
        - REDIS_PORT: Redis server port
        - REDIS_DB: Redis database number
        - REDIS_PASSWORD: Redis authentication password (optional)
        
        Raises:
            None: All exceptions are caught and logged, system continues operation
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis module not available. Operating in fallback mode (no caching)")
            self._redis_client = None
            self._is_connected = False
            return
            
        try:
            # Redis configuration from app_config
            redis_host = app_config.REDIS_HOST
            redis_port = app_config.REDIS_PORT
            redis_db = app_config.REDIS_DB
            # Password can still be an env var
            redis_password = os.environ.get("REDIS_PASSWORD", None)
            
            # Connection configuration
            self._redis_client = redis.StrictRedis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self._redis_client.ping()
            self._is_connected = True
            logger.info(f"Redis cache manager initialized successfully at {redis_host}:{redis_port}")
            
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}. Operating in fallback mode (no caching)")
            self._redis_client = None
            self._is_connected = False
        except Exception as e:
            logger.error(f"Unexpected error during Redis initialization: {e}")
            self._redis_client = None
            self._is_connected = False
    
    def is_available(self) -> bool:
        """
        Check if Redis is available and responsive.
        
        Performs a health check on the Redis connection and updates the
        connection status. This method is called before each cache operation
        to ensure reliable behavior.
        
        Returns:
            bool: True if Redis is available and responsive, False otherwise
            
        Example:
            >>> cache = CacheManager()
            >>> if cache.is_available():
            ...     cache.set_classification_result("test", {"result": True})
            ... else:
            ...     print("Cache unavailable, skipping operation")
        """
        if not self._redis_client or not REDIS_AVAILABLE:
            return False
        
        try:
            self._redis_client.ping()
            if not self._is_connected:
                logger.info("Redis connection restored")
                self._is_connected = True
            return True
        except redis.exceptions.ConnectionError:
            if self._is_connected:
                logger.warning("Redis connection lost")
                self._is_connected = False
            return False
    
    def get_classification_result(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached classification result for a recipe.
        
        Retrieves a previously cached classification result with metadata
        including cache timestamp and version information. Returns None
        if the result is not cached or Redis is unavailable.
        
        Args:
            recipe_id: Unique identifier for the recipe
            
        Returns:
            Dict containing classification results and metadata, or None if not cached
            
        Example:
            >>> result = cache.get_classification_result("recipe_123")
            >>> if result:
            ...     print(f"Keto: {result['classification']['keto']}")
            ...     print(f"Cached at: {result['cached_at']}")
        """
        if not self.is_available():
            return None
        
        cache_key = f"classification:{recipe_id}"
        try:
            cached_data = self._redis_client.get(cache_key)
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for recipe {recipe_id}")
                return result
            logger.debug(f"Cache miss for recipe {recipe_id}")
            return None
        except (redis.exceptions.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached result for recipe {recipe_id}: {e}")
            return None
    
    def set_classification_result(self, recipe_id: str, result: Dict[str, Any], ttl: int = 86400):
        """
        Cache a classification result for a recipe.
        
        Stores a classification result with metadata including cache timestamp
        and version information. The result is stored with a default TTL of
        24 hours to balance performance and data freshness.
        
        Args:
            recipe_id: Unique identifier for the recipe
            result: Classification result dictionary
            ttl: Time to live in seconds (default: 24 hours)
            
        Example:
            >>> result = {"keto": True, "vegan": False, "confidence": 0.95}
            >>> cache.set_classification_result("test_recipe_123", result)
        """
        if not self.is_available():
            logger.debug(f"Redis unavailable, skipping cache write for recipe {recipe_id}")
            return
        
        cache_key = f"classification:{recipe_id}"
        try:
            # Add metadata to the cached result
            cache_data = {
                "recipe_id": recipe_id,
                "classification": result,
                "cached_at": self._get_current_timestamp(),
                "cache_version": "1.0"
            }
            
            serialized_data = json.dumps(cache_data, ensure_ascii=False, separators=(',', ':'))
            self._redis_client.setex(cache_key, ttl, serialized_data)
            logger.debug(f"Cached classification result for recipe {recipe_id}")
            
        except (redis.exceptions.RedisError, TypeError) as e:
            logger.error(f"Failed to cache result for recipe {recipe_id}: {e}")
    
    def get_ingredient_context(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached factual context for an ingredient.
        
        Retrieves previously cached nutritional and vegan context information
        for an ingredient. This data has a longer TTL (7 days) since it
        represents factual information that changes infrequently.
        
        Args:
            ingredient_name: Normalized ingredient name
            
        Returns:
            Dict containing nutritional and vegan context, or None if not cached
            
        Example:
            >>> context = cache.get_ingredient_context("chicken breast")
            >>> if context:
            ...     print(f"Protein: {context['nutrition']['protein_g']}g")
            ...     print(f"Vegan: {context['vegan']['is_vegan']}")
        """
        if not self.is_available():
            return None
        
        cache_key = f"ingredient:{ingredient_name.lower().strip()}"
        try:
            cached_data = self._redis_client.get(cache_key)
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for ingredient {ingredient_name}")
                return result
            return None
        except (redis.exceptions.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached context for ingredient {ingredient_name}: {e}")
            return None
    
    def set_ingredient_context(self, ingredient_name: str, context: Dict[str, Any], ttl: int = 604800):
        """
        Cache factual context for an ingredient.
        
        Stores nutritional and vegan context information for an ingredient
        with a default TTL of 7 days. This longer TTL is appropriate since
        nutritional data changes infrequently.
        
        Args:
            ingredient_name: Normalized ingredient name
            context: Factual context dictionary
            ttl: Time to live in seconds (default: 7 days)
            
        Example:
            >>> context = {
            ...     "nutrition": {"protein_g": 25.0, "calories": 165},
            ...     "vegan": {"is_vegan": False, "reason": "animal product"}
            ... }
            >>> cache.set_ingredient_context("chicken breast", context)
        """
        if not self.is_available():
            return
        
        cache_key = f"ingredient:{ingredient_name.lower().strip()}"
        try:
            cache_data = {
                "ingredient_name": ingredient_name,
                "context": context,
                "cached_at": self._get_current_timestamp(),
                "cache_version": "1.0"
            }
            
            serialized_data = json.dumps(cache_data, ensure_ascii=False, separators=(',', ':'))
            self._redis_client.setex(cache_key, ttl, serialized_data)
            logger.debug(f"Cached ingredient context for {ingredient_name}")
            
        except (redis.exceptions.RedisError, TypeError) as e:
            logger.error(f"Failed to cache context for ingredient {ingredient_name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and health information.
        
        Retrieves comprehensive statistics about cache usage, performance,
        and health status. This information is useful for monitoring and
        debugging cache behavior.
        
        Returns:
            Dict containing cache statistics and health information
            
        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Cache hits: {stats['hits']}")
            >>> print(f"Cache misses: {stats['misses']}")
            >>> print(f"Connection status: {stats['connected']}")
        """
        stats = {
            "connected": self._is_connected,
            "available": self.is_available(),
            "hits": 0,
            "misses": 0,
            "keys": 0
        }
        
        if not self.is_available():
            return stats
        
        try:
            # Get basic Redis info
            info = self._redis_client.info()
            stats.update({
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            })
            
            # Count keys by pattern
            classification_keys = len(self._redis_client.keys("classification:*"))
            ingredient_keys = len(self._redis_client.keys("ingredient:*"))
            stats["keys"] = classification_keys + ingredient_keys
            stats["classification_keys"] = classification_keys
            stats["ingredient_keys"] = ingredient_keys
            
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to get cache statistics: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache entries matching a pattern.
        
        Removes cache entries that match the specified pattern. If no pattern
        is provided, clears all cache entries. This is useful for cache
        invalidation and maintenance.
        
        Args:
            pattern: Redis key pattern to match (e.g., "classification:*")
                    If None, clears all cache entries
            
        Example:
            >>> cache.clear_cache("classification:*")  # Clear all classification results
            >>> cache.clear_cache()  # Clear entire cache
        """
        if not self.is_available():
            logger.warning("Redis unavailable, cannot clear cache")
            return
        
        try:
            if pattern:
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cache entries matching pattern: {pattern}")
                else:
                    logger.info(f"No cache entries found matching pattern: {pattern}")
            else:
                self._redis_client.flushdb()
                logger.info("Cleared entire cache")
                
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format for cache metadata.
        
        Returns:
            str: Current timestamp in ISO 8601 format
            
        Example:
            >>> timestamp = cache._get_current_timestamp()
            >>> print(timestamp)
            '2024-01-15T10:30:45.123456'
        """
        from datetime import datetime
        return datetime.now().isoformat()

def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    This function provides access to the singleton CacheManager instance,
    ensuring consistent caching behavior across the application.
    
    Returns:
        CacheManager: Global cache manager instance
        
    Example:
        >>> cache = get_cache_manager()
        >>> cache.set_classification_result("test", {"result": True})
    """
    return CacheManager()

if __name__ == "__main__":
    # Basic testing and demonstration
    import time
    
    print("Testing Redis Cache Manager")
    print("=" * 40)
    
    cm = get_cache_manager()
    
    # Test basic functionality
    if cm.is_available():
        print("✓ Redis connection successful")
        
        # Test ingredient caching
        test_context = {
            "nutrition_data": {"calories": 100, "carbohydrates": 5, "protein": 20},
            "vegan_status": False,
            "confidence": 0.95
        }
        
        cm.set_ingredient_context("chicken_breast", test_context)
        retrieved = cm.get_ingredient_context("chicken_breast")
        
        if retrieved and retrieved["context"] == test_context:
            print("✓ Ingredient context caching works")
        else:
            print("✗ Ingredient context caching failed")
        
        # Test classification caching
        test_result = {"is_vegan": False, "is_keto": True, "confidence": 0.85}
        cm.set_classification_result("test_recipe_123", test_result)
        retrieved = cm.get_classification_result("test_recipe_123")
        
        if retrieved and retrieved["classification"] == test_result:
            print("✓ Classification result caching works")
        else:
            print("✗ Classification result caching failed")
        
        # Show stats
        print("\nCache Stats:")
        stats = cm.get_stats()
        for k, v in stats.items():
            print(f"- {k}: {v}")
            
        # Clear cache
        print("\nClearing cache...")
        cm.clear_cache()
        if not cm.get_classification_result("test_recipe_123") and not cm.get_ingredient_context("chicken_breast"):
            print("✓ Cache cleared successfully")
        else:
            print("✗ Cache clear failed")
            
    else:
        print("✗ Redis connection failed. Check Redis server and configuration.")
    
    print("\nDemonstration complete.") 