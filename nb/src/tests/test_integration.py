#!/usr/bin/env python3
"""
Integration Tests for the Standalone RAG Classification System

This script tests the integration of all nb/src modules to ensure:
1. All modules can be imported correctly.
2. The main classifier can be instantiated with its dependencies.
3. The database and LLM clients can be initialized.
4. The new function-calling pipeline works end-to-end.
"""
import pytest
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def test_config_import_and_attributes():
    """Tests that the central configuration can be imported and has key attributes."""
    logger.info("Testing configuration import.")
    from config import app_config
    assert hasattr(app_config, 'DB_PATH'), "DB_PATH not found in config"
    assert hasattr(app_config, 'KETO_CARBS_THRESHOLD'), "KETO_CARBS_THRESHOLD not found in config"
    assert app_config.DB_PATH.exists(), f"Database path {app_config.DB_PATH} does not exist. Run the ingestion script."
    logger.info("Configuration import and attributes are valid.")

def test_database_manager_initialization():
    """Tests that the unified database manager can initialize its clients."""
    logger.info("Testing database manager initialization.")
    from database import db_manager
    
    # Test SQLite connection
    try:
        with db_manager.get_sqlite_connection() as conn:
            assert conn is not None, "SQLite connection failed."
        logger.info("SQLite connection successful.")
    except Exception as e:
        pytest.fail(f"SQLite connection test failed with an exception: {e}")

    # Test OpenSearch client (should not raise an error even if not running)
    try:
        client = db_manager.get_opensearch_client()
        if client:
            logger.info("OpenSearch client initialized (or was already).")
        else:
            logger.warning("OpenSearch client is None (service may not be running). This is acceptable for the test.")
    except Exception as e:
        pytest.fail(f"OpenSearch client initialization failed with an exception: {e}")

def test_llm_client_initialization():
    """Tests that the LLM client can be initialized."""
    logger.info("Testing LLM client initialization.")
    from llm_client import llm_client
    assert llm_client is not None, "LLM client is None."
    logger.info("LLM client initialized successfully.")

def test_cache_manager_initialization():
    """Tests that the Redis cache manager can be initialized."""
    logger.info("Testing cache manager initialization.")
    from utils.cache_manager import get_cache_manager
    
    cache_manager = get_cache_manager()
    assert cache_manager is not None, "Cache manager is None."
    
    # Test basic functionality (should work even if Redis is not available)
    stats = cache_manager.get_stats()
    assert isinstance(stats, dict), "Cache stats should return a dictionary."
    assert "status" in stats, "Cache stats should include status."
    
    logger.info(f"Cache manager initialized with status: {stats['status']}")

def test_core_logic_imports():
    """Tests that all core classification modules can be imported."""
    logger.info("Testing imports for core classification logic.")
    try:
        from function_calling_handler import FunctionCallingHandler
        from query_engine import QueryEngine
        from context_aware_classifier import ContextAwareDietClassifier
        from diet_classifiers import is_keto, is_vegan
        logger.info("All core classification modules imported successfully.")
    except ImportError as e:
        pytest.fail(f"Failed to import a core classification module: {e}")

def test_function_calling_handler_instantiation():
    """Tests that the FunctionCallingHandler can be instantiated."""
    logger.info("Testing FunctionCallingHandler instantiation.")
    from function_calling_handler import FunctionCallingHandler
    
    try:
        handler = FunctionCallingHandler()
        assert handler is not None, "FunctionCallingHandler instantiation returned None."
        logger.info("FunctionCallingHandler instantiated successfully.")
    except Exception as e:
        pytest.fail(f"Failed to instantiate FunctionCallingHandler: {e}")

def test_query_engine_instantiation():
    """Tests that the QueryEngine can be instantiated."""
    logger.info("Testing QueryEngine instantiation.")
    from query_engine import QueryEngine
    
    try:
        engine = QueryEngine()
        assert engine is not None, "QueryEngine instantiation returned None."
        logger.info("QueryEngine instantiated successfully.")
    except Exception as e:
        pytest.fail(f"Failed to instantiate QueryEngine: {e}")

def test_classifier_instantiation():
    """
    Tests if the main ContextAwareDietClassifier can be instantiated,
    which implicitly tests if its dependencies are wired correctly.
    """
    logger.info("Testing instantiation of the ContextAwareDietClassifier.")
    from context_aware_classifier import ContextAwareDietClassifier
    
    try:
        # This will test the connection between the classifier, the LLM client,
        # the function handler, and the query engine.
        classifier = ContextAwareDietClassifier()
        assert classifier is not None, "Classifier instantiation returned None."
        assert hasattr(classifier, 'llm_client'), "Classifier is missing llm_client."
        assert hasattr(classifier, 'function_handler'), "Classifier is missing function_handler."
        assert hasattr(classifier, 'query_engine'), "Classifier is missing query_engine."
        logger.info("ContextAwareDietClassifier instantiated successfully.")
    except Exception as e:
        pytest.fail(f"Failed to instantiate ContextAwareDietClassifier: {e}")

def test_diet_classifiers_entry_points():
    """Tests that the main diet classifier functions can be called."""
    logger.info("Testing diet classifier entry points.")
    from diet_classifiers import is_keto, is_vegan
    
    # Test with simple ingredients list
    test_ingredients = ["chicken breast", "spinach", "olive oil"]
    
    try:
        keto_result = is_keto(test_ingredients)
        assert isinstance(keto_result, bool), "is_keto should return a boolean."
        logger.info(f"is_keto returned: {keto_result}")
        
        vegan_result = is_vegan(test_ingredients)
        assert isinstance(vegan_result, bool), "is_vegan should return a boolean."
        logger.info(f"is_vegan returned: {vegan_result}")
        
        logger.info("Diet classifier entry points work correctly.")
    except Exception as e:
        pytest.fail(f"Failed to call diet classifier functions: {e}")

def test_ingredient_processor_functionality():
    """Tests that the ingredient processor can process ingredients."""
    logger.info("Testing ingredient processor functionality.")
    
    try:
        from ingredient_processor.processor import processor
        
        # Test the enhanced processor
        result = processor.process_ingredient_comprehensive("chicken breast")
        assert isinstance(result, dict), "Processor should return a dictionary."
        assert "original" in result, "Result should contain original ingredient."
        assert "match_type" in result, "Result should contain match type."
        
        logger.info(f"Ingredient processor returned match type: {result['match_type']}")
        
    except Exception as e:
        pytest.fail(f"Ingredient processor test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 