#!/usr/bin/env python3
"""
Redis Cache Manager for Diet Classification System

This module provides a centralized Redis caching solution optimized for storing
classification results and factual contexts. It implements a cache-first approach
with graceful fallback when Redis is unavailable.

Key Features:
- Thread-safe singleton pattern
- Automatic JSON serialization/deserialization 
- Configurable TTL for different data types
- Graceful degradation when Redis is unavailable
- Professional logging without emojis
"""

import redis
import json
import os
import logging
from typing import Optional, Any, Dict, Union
from pathlib import Path

# Configure professional logging
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Thread-safe Redis cache manager with automatic fallback capabilities.
    
    This class implements a singleton pattern and provides high-level caching
    operations for the diet classification pipeline. It handles connection
    failures gracefully and provides detailed logging for debugging.
    """
    
    _instance = None
    _redis_client = None
    _is_connected = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Redis connection with environment-based configuration."""
        try:
            # Redis configuration from environment variables
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            redis_db = int(os.environ.get("REDIS_DB", 0))
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
        
        Returns:
            bool: True if Redis is available, False otherwise
        """
        if not self._redis_client:
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
        
        Args:
            recipe_id: Unique identifier for the recipe
            
        Returns:
            Dict containing classification results and metadata, or None if not cached
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
        
        Args:
            recipe_id: Unique identifier for the recipe
            result: Classification result dictionary
            ttl: Time to live in seconds (default: 24 hours)
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
        
        Args:
            ingredient_name: Normalized ingredient name
            
        Returns:
            Dict containing nutritional and vegan context, or None if not cached
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
        
        Args:
            ingredient_name: Normalized ingredient name
            context: Factual context dictionary
            ttl: Time to live in seconds (default: 7 days)
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
            logger.debug(f"Cached context for ingredient {ingredient_name}")
            
        except (redis.exceptions.RedisError, TypeError) as e:
            logger.error(f"Failed to cache context for ingredient {ingredient_name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and health information.
        
        Returns:
            Dict containing cache statistics
        """
        if not self.is_available():
            return {
                "status": "unavailable",
                "total_keys": 0,
                "classification_keys": 0,
                "ingredient_keys": 0
            }
        
        try:
            info = self._redis_client.info()
            
            # Count different key types
            classification_keys = len(self._redis_client.keys("classification:*"))
            ingredient_keys = len(self._redis_client.keys("ingredient:*"))
            total_keys = info.get('db0', {}).get('keys', 0) if 'db0' in info else 0
            
            return {
                "status": "connected",
                "total_keys": total_keys,
                "classification_keys": classification_keys,
                "ingredient_keys": ingredient_keys,
                "memory_usage": info.get('used_memory_human', 'unknown'),
                "connected_clients": info.get('connected_clients', 0),
                "uptime_seconds": info.get('uptime_in_seconds', 0)
            }
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "classification:*"). If None, clears all cache.
        """
        if not self.is_available():
            logger.warning("Cannot clear cache: Redis unavailable")
            return
        
        try:
            if pattern:
                keys = self._redis_client.keys(pattern)
                if keys:
                    deleted = self._redis_client.delete(*keys)
                    logger.info(f"Cleared {deleted} cache entries matching pattern '{pattern}'")
            else:
                self._redis_client.flushdb()
                logger.info("Cleared entire cache database")
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

# Global cache manager instance
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager: The singleton cache manager instance
    """
    return cache_manager

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
        stats = cm.get_stats()
        print(f"✓ Cache stats: {stats['total_keys']} total keys")
        
    else:
        print("✗ Redis connection failed - operating in fallback mode")
    
    print("\nCache manager ready for use!") 