"""
Debug script for the evaluation pipeline.

This script helps identify and fix issues in the evaluation pipeline
by testing each component individually and providing detailed diagnostics.
"""
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
import json

# Import nest_asyncio for Jupyter compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("Warning: nest_asyncio not available")

from loguru import logger

# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import app_config
    from llm_client import llm_client
    from context_aware_classifier import ContextAwareDietClassifier
    from function_calling_handler import FunctionCallingHandler
    from database import db_manager
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

async def test_ollama_connection():
    """Test Ollama connection and model availability."""
    logger.info("=== Testing Ollama Connection ===")
    
    try:
        # Test basic connection
        models = llm_client.list_models()
        logger.info(f"✅ Ollama connection successful. Found {len(models)} models")
        
        # List all models
        model_names = []
        for model in models:
            if hasattr(model, 'model'):
                model_name = model.model
            elif isinstance(model, dict):
                model_name = model.get('name', '') or model.get('model', '')
            else:
                continue
            model_names.append(model_name)
        
        logger.info(f"Available models: {model_names}")
        
        # Check required models
        required_models = ["qwen", "arctic"]
        available_required = []
        
        for req_model in required_models:
            if llm_client.is_model_available(req_model):
                available_required.append(req_model)
                logger.info(f"✅ Required model '{req_model}' is available")
            else:
                logger.error(f"❌ Required model '{req_model}' is NOT available")
        
        return len(available_required) > 0
        
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return False

async def test_database_connection():
    """Test database connection and tables."""
    logger.info("=== Testing Database Connection ===")
    
    try:
        with db_manager.get_sqlite_connection() as conn:
            # Test basic connection
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            logger.info(f"✅ Database connection successful. Found {len(tables)} tables")
            
            # Check required tables
            required_tables = ["nutrition_facts", "vegan_ontology"]
            for table in required_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ Table '{table}' exists with {count} records")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def test_function_calling():
    """Test function calling handler."""
    logger.info("=== Testing Function Calling Handler ===")
    
    try:
        handler = FunctionCallingHandler()
        logger.info("✅ FunctionCallingHandler initialized successfully")
        
        # Test a simple query
        test_question = "Find chicken in nutrition_facts"
        logger.info(f"Testing function calling with: '{test_question}'")
        
        result = await asyncio.wait_for(
            handler.generate_json_query(test_question),
            timeout=30.0
        )
        
        if "error" in result:
            logger.error(f"❌ Function calling failed: {result['error']}")
            return False
        else:
            logger.info(f"✅ Function calling successful: {result}")
            return True
            
    except asyncio.TimeoutError:
        logger.error("❌ Function calling timed out after 30 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Function calling failed: {e}")
        return False

async def test_classifier_initialization():
    """Test classifier initialization."""
    logger.info("=== Testing Classifier Initialization ===")
    
    try:
        logger.info("Initializing ContextAwareDietClassifier...")
        classifier = ContextAwareDietClassifier()
        logger.info("✅ ContextAwareDietClassifier initialized successfully")
        
        # Test basic functionality
        test_ingredient = "chicken breast"
        logger.info(f"Testing single ingredient classification: '{test_ingredient}'")
        
        result = await asyncio.wait_for(
            classifier.classify_single_ingredient(test_ingredient),
            timeout=60.0
        )
        
        if "error" in result:
            logger.error(f"❌ Single ingredient classification failed: {result['error']}")
            return False
        else:
            logger.info(f"✅ Single ingredient classification successful: {result}")
            return True
            
    except asyncio.TimeoutError:
        logger.error("❌ Classifier initialization timed out after 60 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Classifier initialization failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

async def test_data_loading():
    """Test data loading functionality."""
    logger.info("=== Testing Data Loading ===")
    
    try:
        eval_data_dir = Path(app_config.PROJECT_ROOT) / "eval_data"
        logger.info(f"Checking eval_data directory: {eval_data_dir}")
        
        if not eval_data_dir.exists():
            logger.error(f"❌ Eval data directory does not exist: {eval_data_dir}")
            return False
        
        # Check for required files
        required_files = ["borderline_keto.csv", "strict_keto.csv", "strict_vegan.csv", "ground_truth_sample.csv"]
        missing_files = []
        
        for file_name in required_files:
            file_path = eval_data_dir / file_name
            if file_path.exists():
                logger.info(f"✅ Found {file_name}")
            else:
                missing_files.append(file_name)
                logger.error(f"❌ Missing {file_name}")
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ All required data files found")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False

async def run_comprehensive_debug():
    """Run comprehensive debugging of the evaluation pipeline."""
    logger.info("🚀 Starting Comprehensive Debug of Evaluation Pipeline")
    logger.info("=" * 60)
    
    # Test results
    results = {}
    
    # Test 1: Ollama Connection
    results['ollama'] = await test_ollama_connection()
    
    # Test 2: Database Connection
    results['database'] = await test_database_connection()
    
    # Test 3: Function Calling
    results['function_calling'] = await test_function_calling()
    
    # Test 4: Classifier Initialization
    results['classifier'] = await test_classifier_initialization()
    
    # Test 5: Data Loading
    results['data_loading'] = await test_data_loading()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 DEBUG SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.success("🎉 All tests passed! The evaluation pipeline should work.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix the issues above before running evaluation.")
        return False

def main():
    """Main debug function."""
    try:
        success = asyncio.run(run_comprehensive_debug())
        return 0 if success else -1
    except Exception as e:
        logger.error(f"Critical error during debugging: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return -1

if __name__ == "__main__":
    sys.exit(main()) 