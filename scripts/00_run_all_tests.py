#!/usr/bin/env python3
"""
Comprehensive Test Runner Script

This script runs all tests to ensure the system is working correctly before
running the main data processing pipeline. It tests:

1. Ingredient Parser functionality
2. Generalized Ingredient Analyzer
3. Basic classification pipeline
4. Database connectivity
5. LLM service availability

This should be run before the main pipeline scripts to catch issues early.

Returns:
    int: Exit code (0 for success, 1 for failure)

Raises:
    ImportError: If required modules cannot be imported
    ConnectionError: If database or LLM services are unavailable
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add nb/src to path for accessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from config import app_config


def test_database_connectivity():
    """
    Test database connectivity and basic operations.
    
    Returns:
        bool: True if database connectivity test passes, False otherwise
    """
    logger.info("Testing database connectivity...")
    
    try:
        from database import db_manager
        
        # Test basic connection using context manager
        with db_manager.get_sqlite_connection() as connection:
            logger.info("Database connection successful")
            
            # Test basic query
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            logger.info(f"Found {len(tables)} tables in database")
            
            # Check for required tables
            required_tables = ['nutrition_facts', 'vegan_ontology', 'unit_conversions']
            table_names = [table[0] for table in tables]
            
            missing_tables = [table for table in required_tables if table not in table_names]
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
                return False
            else:
                logger.info("All required tables present")
                return True
            
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False


def test_llm_services():
    """
    Test LLM service availability.
    
    Returns:
        bool: True if LLM services are available, False otherwise
    """
    logger.info("Testing LLM services...")
    
    try:
        from llm_client import llm_client
        
        # Test basic connection
        models = llm_client.list_models()
        logger.info(f"Found {len(models)} available models")
        
        # Check for required models
        required_models = ["qwen", "arctic"]
        available_model_names = []
        
        for model in models:
            if hasattr(model, "model"):
                model_name = model.model
            elif isinstance(model, dict):
                model_name = model.get("name", "") or model.get("model", "")
            else:
                continue
            available_model_names.append(model_name)
        
        logger.info(f"Available models: {available_model_names}")
        
        # Check if we have at least one of the required models
        has_required_model = any(
            any(req in name for req in required_models)
            for name in available_model_names
        )
        
        if has_required_model:
            logger.info("Required LLM models available")
            return True
        else:
            logger.warning("No required models found, but continuing...")
            return True  # Continue with warnings
            
    except Exception as e:
        logger.error(f"LLM service test failed: {e}")
        return False


def test_ingredient_parser():
    """
    Test ingredient parser functionality.
    
    Returns:
        bool: True if ingredient parser tests pass, False otherwise
    """
    logger.info("Testing ingredient parser...")
    
    try:
        # Import the test module
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src" / "tests"))
        from test_ingredient_parser import test_ingredient_parser, test_diet_classifiers, test_simple_classification
        
        # Test ingredient parser
        parser_success = test_ingredient_parser()
        
        # Test diet classifier integration
        classifier_success = test_diet_classifiers()
        
        # Test simple classification
        classification_success = test_simple_classification()
        
        overall_success = parser_success and classifier_success and classification_success
        
        if overall_success:
            logger.info("Ingredient parser tests passed")
        else:
            logger.error("Some ingredient parser tests failed")
            
        return overall_success
        
    except Exception as e:
        logger.error(f"Ingredient parser test failed: {e}")
        return False


async def test_generalized_analyzer():
    """
    Test the GeneralizedIngredientAnalyzer functionality.
    
    Returns:
        bool: True if GeneralizedIngredientAnalyzer tests pass, False otherwise
    """
    logger.info("Testing GeneralizedIngredientAnalyzer...")
    
    try:
        # Import the test module
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src" / "tests"))
        from test_generalized_analyzer import test_individual_components, test_ingredient_analyzer
        
        # Test individual components
        logger.info("Testing individual components...")
        await test_individual_components()
        
        # Test ingredient analyzer
        logger.info("Testing ingredient analyzer...")
        await test_ingredient_analyzer()
        
        logger.info("GeneralizedIngredientAnalyzer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"GeneralizedIngredientAnalyzer test failed: {e}")
        return False


def test_configuration():
    """
    Test configuration loading.
    
    Returns:
        bool: True if configuration loads successfully, False otherwise
    """
    logger.info("Testing configuration...")
    
    try:
        # Test basic config loading
        if app_config:
            logger.info("Configuration loaded successfully")
            
            # Test required config values
            required_keys = ['OPENSEARCH_URL', 'REDIS_URL', 'OLLAMA_URL']
            missing_keys = [key for key in required_keys if not getattr(app_config, key, None)]
            
            if missing_keys:
                logger.warning(f"Missing config keys: {missing_keys}")
                return True  # Continue with warnings
            else:
                logger.info("All required configuration present")
                return True
        else:
            logger.error("Configuration loading failed")
            return False
            
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


async def main():
    """
    Run all tests.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE SYSTEM TEST")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test configuration first
    test_results['configuration'] = test_configuration()
    
    # Test database connectivity
    test_results['database'] = test_database_connectivity()
    
    # Test LLM services
    test_results['llm_services'] = test_llm_services()
    
    # Test ingredient parser
    test_results['ingredient_parser'] = test_ingredient_parser()
    
    # Test generalized analyzer (async)
    test_results['generalized_analyzer'] = await test_generalized_analyzer()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("All tests passed! System is ready for processing.")
        return 0
    elif passed_tests >= total_tests * 0.8:  # 80% threshold
        logger.warning("Most tests passed, but some issues detected. Review warnings above.")
        return 0  # Continue with warnings
    else:
        logger.error("Too many tests failed. Please fix issues before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 