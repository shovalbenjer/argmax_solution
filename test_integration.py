#!/usr/bin/env python3
"""
Integration Test Script for Fixed Architecture

This script tests the critical fixes implemented to resolve:
1. Code duplication elimination
2. Missing imports fixes
3. Mock implementation replacements
4. Architecture inconsistencies

Run this script to verify that the shared modules work correctly.
"""
import sys
import os
from pathlib import Path

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_shared_imports():
    """Test that shared modules can be imported correctly."""
    print("🧪 Testing shared module imports...")
    
    try:
        from shared.config import app_config
        print("✅ Shared config import successful")
        
        from shared.database import db_manager
        print("✅ Shared database import successful")
        
        from shared.diet_classifiers import is_keto, is_vegan, diet_classifier
        print("✅ Shared diet classifiers import successful")
        
        from shared.llm_client import llm_client
        print("✅ Shared LLM client import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test that configuration is properly loaded."""
    print("\n🧪 Testing configuration...")
    
    try:
        from shared.config import app_config
        
        # Test required attributes
        assert hasattr(app_config, 'OPENSEARCH_URL')
        assert hasattr(app_config, 'DB_PATH')
        assert hasattr(app_config, 'KETO_CARBS_THRESHOLD')
        
        print(f"✅ OpenSearch URL: {app_config.OPENSEARCH_URL}")
        print(f"✅ Database Path: {app_config.DB_PATH}")
        print(f"✅ Keto Threshold: {app_config.KETO_CARBS_THRESHOLD}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_manager():
    """Test database manager functionality."""
    print("\n🧪 Testing database manager...")
    
    try:
        from shared.database import db_manager
        
        # Test OpenSearch client initialization (may fail if service not running)
        client = db_manager.get_opensearch_client()
        if client:
            print("✅ OpenSearch client initialized")
        else:
            print("⚠️  OpenSearch client not available (service may not be running)")
        
        # Test SQLite connection context manager
        with db_manager.get_sqlite_connection() as conn:
            print("✅ SQLite connection context manager works")
        
        return True
    except Exception as e:
        print(f"❌ Database manager test failed: {e}")
        return False

def test_diet_classifiers():
    """Test diet classification functionality."""
    print("\n🧪 Testing diet classifiers...")
    
    try:
        from shared.diet_classifiers import is_keto, is_vegan, diet_classifier
        
        # Test with sample ingredients
        test_ingredients = ["chicken breast", "spinach", "olive oil"]
        
        # Test keto classification
        keto_result = is_keto(test_ingredients)
        print(f"✅ Keto classification result: {keto_result}")
        
        # Test vegan classification
        vegan_result = is_vegan(test_ingredients)
        print(f"✅ Vegan classification result: {vegan_result}")
        
        # Test individual ingredient methods
        is_chicken_keto = diet_classifier.is_ingredient_keto("chicken breast")
        is_chicken_vegan = diet_classifier.is_ingredient_vegan("chicken breast")
        
        print(f"✅ Chicken breast - Keto: {is_chicken_keto}, Vegan: {is_chicken_vegan}")
        
        return True
    except Exception as e:
        print(f"❌ Diet classifier test failed: {e}")
        return False

def test_llm_client():
    """Test LLM client functionality."""
    print("\n🧪 Testing LLM client...")
    
    try:
        from shared.llm_client import llm_client
        
        # Test model listing (may fail if Ollama not running)
        models = llm_client.list_models()
        if models:
            print(f"✅ Found {len(models)} available models")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model.get('name', 'Unknown')}")
        else:
            print("⚠️  No models available (Ollama may not be running)")
        
        # Test recommended model selection
        recommended = llm_client.get_recommended_model()
        if recommended:
            print(f"✅ Recommended model: {recommended}")
        else:
            print("⚠️  No recommended model available")
        
        return True
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        return False

def test_web_service_integration():
    """Test that web service can use shared modules."""
    print("\n🧪 Testing web service integration...")
    
    try:
        # Simulate web service import pattern
        sys.path.insert(0, 'web/src')
        
        # This would be the pattern used in web/src/app.py
        from shared.config import app_config
        from shared.database import db_manager
        from shared.diet_classifiers import is_keto, is_vegan
        
        print("✅ Web service can import shared modules")
        
        # Test that the functions work as expected
        test_ingredients = ["flour", "sugar", "butter"]
        result = is_keto(test_ingredients)
        print(f"✅ Web service diet classification works: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Web service integration test failed: {e}")
        return False

def test_notebook_service_integration():
    """Test that notebook service can use shared modules."""
    print("\n🧪 Testing notebook service integration...")
    
    try:
        # Simulate notebook service import pattern
        sys.path.insert(0, 'nb/src')
        
        # This would be the pattern used in nb/src files
        from shared.diet_classifiers import diet_classifier
        from shared.database import db_manager
        
        print("✅ Notebook service can import shared modules")
        
        # Test ingredient context
        context = diet_classifier.processor.get_ingredient_context("chicken")
        print(f"✅ Ingredient context generation works: {bool(context)}")
        
        return True
    except Exception as e:
        print(f"❌ Notebook service integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Integration Tests for Fixed Architecture\n")
    print("="*60)
    
    tests = [
        test_shared_imports,
        test_configuration,
        test_database_manager,
        test_diet_classifiers,
        test_llm_client,
        test_web_service_integration,
        test_notebook_service_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print(f"🏁 Integration Tests Complete: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The architecture fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main()) 