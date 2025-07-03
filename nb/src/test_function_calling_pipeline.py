import asyncio
import json
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from function_calling_handler import FunctionCallingHandler
from query_engine import translate_json_to_sql
from context_aware_classifier import ContextAwareDietClassifier

async def test_pipeline_components():
    """Tests each component of the function-calling pipeline individually."""
    print("--- Testing Pipeline Components ---")
    
    # 1. Test Function Calling Handler (Text-to-JSON)
    print("\n1. Testing FunctionCallingHandler (Text-to-JSON)...")
    fc_handler = FunctionCallingHandler()
    question = "What are the nutritional values for spinach?"
    json_query = await fc_handler.generate_json_query(question)
    print(f"   Generated JSON for '{question}':\n{json.dumps(json_query, indent=2)}")
    assert "operation" in json_query and json_query["operation"] == "search"
    assert "table" in json_query and json_query["table"] == "nutrition_facts"
    print("   ✅ FunctionCallingHandler OK")

    # 2. Test Query Engine (JSON-to-SQL)
    print("\n2. Testing QueryEngine (JSON-to-SQL)...")
    sql, params = translate_json_to_sql(json_query)
    print(f"   Translated SQL: {sql}")
    print(f"   Parameters: {params}")
    assert "SELECT" in sql and "FROM nutrition_facts" in sql and "WHERE name" in sql
    assert params[0] == '%spinach%'
    print("   ✅ QueryEngine OK")

async def test_end_to_end_classification():
    """Tests the full end-to-end classification pipeline."""
    print("\n--- Testing End-to-End Classification ---")
    classifier = ContextAwareDietClassifier()

    # Test Case 1: A simple, non-vegan, keto-friendly ingredient
    ingredient1 = "egg"
    print(f"\n1. Testing ingredient: '{ingredient1}' (Expected: Keto=True, Vegan=False)")
    result1 = await classifier.classify_single_ingredient(ingredient1)
    print(f"   Result: {json.dumps(result1, indent=2)}")
    assert result1.get("is_keto") is True and result1.get("is_vegan") is False
    print("   ✅ Test Case 1 OK")

    # Test Case 2: A simple, vegan, non-keto ingredient
    ingredient2 = "white rice"
    print(f"\n2. Testing ingredient: '{ingredient2}' (Expected: Keto=False, Vegan=True)")
    result2 = await classifier.classify_single_ingredient(ingredient2)
    print(f"   Result: {json.dumps(result2, indent=2)}")
    assert result2.get("is_keto") is False and result2.get("is_vegan") is True
    print("   ✅ Test Case 2 OK")

    # Test Case 3: A full recipe
    recipe = ["100g chicken breast", "50g of cheddar cheese", "1 slice of white bread"]
    print(f"\n3. Testing recipe: {recipe} (Expected: Keto=False, Vegan=False)")
    recipe_result = await classifier.classify_recipe(recipe)
    print(f"   Result: {json.dumps(recipe_result, indent=2)}")
    assert recipe_result.get("recipe_is_keto") is False and recipe_result.get("recipe_is_vegan") is False
    print("   ✅ Test Case 3 OK")


if __name__ == "__main__":
    print("======================================================")
    print("  Running Full Test of Function-Calling RAG Pipeline  ")
    print("======================================================")
    
    # Run component tests
    asyncio.run(test_pipeline_components())
    
    # Run end-to-end tests
    asyncio.run(test_end_to_end_classification())
    
    print("\n--- All Tests Completed Successfully! ---") 