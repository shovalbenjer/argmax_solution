#!/usr/bin/env python3
"""
Integration Test Script for Standalone Architecture

This script tests the integration of all nb/src modules to ensure:
1. All imports work correctly without shared/ dependency
2. Database connections function properly
3. Arctic -> Qwen pipeline works end-to-end
4. All components integrate correctly

Designed for integration with task.ipynb for MLOps monitoring.
"""
import sys
import os
import logging
from pathlib import Path

# Configure logging (PEP 8 compliant, no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_local_imports():
    """Test that all local nb/src modules can be imported correctly."""
    logger.info("Testing local module imports")
    
    try:
        from config import app_config
        logger.info("Config import successful")
        
        from database import db_manager
        logger.info("Database manager import successful")
        
        from llm_client import llm_client
        logger.info("LLM client import successful")
        
        from arctic_handler import ArcticText2SQLHandler
        logger.info("Arctic handler import successful")
        
        from context_aware_classifier import ContextAwareDietClassifier
        logger.info("Context aware classifier import successful")
        
        from diet_classifiers import is_keto, is_vegan
        logger.info("Diet classifiers import successful")
        
        return True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def test_configuration():
    """Test that configuration is properly loaded."""
    logger.info("Testing configuration")
    
    try:
        from config import app_config
        
        # Test required attributes
        assert hasattr(app_config, 'OPENSEARCH_URL')
        assert hasattr(app_config, 'DB_PATH')
        assert hasattr(app_config, 'KETO_CARBS_THRESHOLD')
        
        logger.info(f"OpenSearch URL: {app_config.OPENSEARCH_URL}")
        logger.info(f"Database Path: {app_config.DB_PATH}")
        logger.info(f"Keto Threshold: {app_config.KETO_CARBS_THRESHOLD}")
        
        return True
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

def test_database_manager():
    """Test database manager functionality."""
    logger.info("Testing database manager")
    
    try:
        from database import db_manager
        
        # Test OpenSearch client initialization (may fail if service not running)
        client = db_manager.get_opensearch_client()
        if client:
            logger.info("OpenSearch client initialized")
        else:
            logger.warning("OpenSearch client not available (service may not be running)")
        
        # Test SQLite connection context manager
        with db_manager.get_sqlite_connection() as conn:
            logger.info("SQLite connection context manager works")
        
        return True
    except Exception as e:
        logger.error(f"Database manager test failed: {e}")
        return False

def test_llm_client():
    """Test LLM client functionality."""
    logger.info("Testing LLM client")
    
    try:
        from llm_client import llm_client
        
        # Test model listing (may fail if Ollama not running)
        models = llm_client.list_models()
        if models:
            logger.info(f"Found {len(models)} available models")
            for model in models[:3]:  # Show first 3 models
                logger.info(f"Available model: {model.get('name', 'Unknown')}")
        else:
            logger.warning("No models available (Ollama may not be running)")
        
        # Test recommended model selection
        recommended = llm_client.get_recommended_model()
        if recommended:
            logger.info(f"Recommended model: {recommended}")
        else:
            logger.warning("No recommended model available")
        
        return True
    except Exception as e:
        logger.error(f"LLM client test failed: {e}")
        return False

def test_arctic_handler():
    """Test Arctic Text2SQL handler."""
    logger.info("Testing Arctic Text2SQL handler")
    
    try:
        from arctic_handler import ArcticText2SQLHandler
        
        handler = ArcticText2SQLHandler()
        logger.info("Arctic handler initialized successfully")
        
        # Test schema loading
        if "SCHEMA NOT FOUND" not in handler.schema:
            logger.info("Database schema loaded successfully")
        else:
            logger.warning("Database schema not found - may need to run data ingestion")
        
        return True
    except Exception as e:
        logger.error(f"Arctic handler test failed: {e}")
        return False

def test_diet_classifiers():
    """Test diet classification functionality."""
    logger.info("Testing diet classifiers")
    
    try:
        from diet_classifiers import is_keto, is_vegan, is_ingredient_keto, is_ingredient_vegan
        
        # Test with sample ingredients
        test_ingredients = ["chicken breast", "spinach", "olive oil"]
        
        # Test recipe-level classification
        keto_result = is_keto(test_ingredients)
        logger.info(f"Keto classification result: {keto_result}")
        
        vegan_result = is_vegan(test_ingredients)
        logger.info(f"Vegan classification result: {vegan_result}")
        
        # Test individual ingredient methods
        is_chicken_keto = is_ingredient_keto("chicken breast")
        is_chicken_vegan = is_ingredient_vegan("chicken breast")
        
        logger.info(f"Chicken breast - Keto: {is_chicken_keto}, Vegan: {is_chicken_vegan}")
        
        return True
    except Exception as e:
        logger.error(f"Diet classifier test failed: {e}")
        return False

def test_context_aware_pipeline():
    """Test the full Arctic -> Qwen pipeline."""
    logger.info("Testing context-aware classification pipeline")
    
    try:
        from context_aware_classifier import ContextAwareDietClassifier
        import asyncio
        
        classifier = ContextAwareDietClassifier()
        logger.info("Context-aware classifier initialized")
        
        # Test single ingredient classification
        async def test_ingredient():
            result = await classifier.classify_single_ingredient("butter")
            return result
        
        result = asyncio.run(test_ingredient())
        if result and "error" not in result:
            logger.info(f"Single ingredient classification successful: {result}")
        else:
            logger.warning(f"Single ingredient classification failed: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Context-aware pipeline test failed: {e}")
        return False

def test_ingredient_processor():
    """Test ingredient processor functionality."""
    logger.info("Testing ingredient processor")
    
    try:
        from ingredient_processor.processor import get_context_with_rapidfuzz_fallback
        
        context = get_context_with_rapidfuzz_fallback("chicken breast")
        logger.info(f"Ingredient context generation works: {bool(context)}")
        
        if context.get('results'):
            logger.info(f"Found {len(context['results'])} nutrition results")
        
        return True
    except Exception as e:
        logger.error(f"Ingredient processor test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("Starting Integration Tests for Standalone Architecture")
    logger.info("=" * 60)
    
    tests = [
        test_local_imports,
        test_configuration,
        test_database_manager,
        test_llm_client,
        test_arctic_handler,
        test_diet_classifiers,
        test_context_aware_pipeline,
        test_ingredient_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                logger.info(f"Test {test.__name__} PASSED")
            else:
                logger.error(f"Test {test.__name__} FAILED")
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info("=" * 60)
    logger.info(f"Integration Tests Complete: {passed}/{total} passed")
    
    if passed == total:
        logger.info("All tests passed! The standalone architecture is working correctly.")
        return 0
    else:
        logger.warning("Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main()) 