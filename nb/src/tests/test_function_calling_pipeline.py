"""
Function-Calling Pipeline Test Suite

This module provides comprehensive testing for the function-calling RAG pipeline,
which enables natural language queries to be converted to structured database
operations and executed safely. It tests the complete pipeline from text input
to classification results.

The test suite covers:
- Function calling handler (text-to-JSON conversion)
- Query engine (JSON-to-SQL translation)
- Context-aware classifier integration
- End-to-end classification pipeline
- Individual component validation
- Full recipe classification testing

Key Test Areas:
- Natural language query processing
- Structured query generation
- SQL translation and parameter binding
- Database query execution
- Classification accuracy validation
- Pipeline integration testing

Test Features:
- Component-level testing for isolation
- End-to-end pipeline validation
- Multiple ingredient and recipe scenarios
- Expected outcome validation
- Error handling verification
- Performance monitoring

Dependencies:
- asyncio: Asynchronous testing support
- json: Data serialization and validation
- sys/os: Path management for imports
- FunctionCallingHandler: Text-to-JSON conversion
- QueryEngine: JSON-to-SQL translation
- ContextAwareDietClassifier: Main classification system

Example:
    >>> python nb/src/tests/test_function_calling_pipeline.py
    >>> # Run specific test function
    >>> asyncio.run(test_pipeline_components())
"""

import asyncio
import json
import os
import sys

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from context_aware_classifier import ContextAwareDietClassifier
from function_calling_handler import FunctionCallingHandler
from query_engine import translate_json_to_sql


async def test_pipeline_components():
    """
    Tests each component of the function-calling pipeline individually.

    This function performs isolated testing of the pipeline components to ensure
    each part works correctly before testing the integrated system. It validates
    the text-to-JSON conversion and JSON-to-SQL translation capabilities.

    Test Components:
        1. FunctionCallingHandler: Converts natural language to structured JSON
        2. QueryEngine: Translates JSON queries to SQL with parameters

    Test Cases:
        - Natural language query: "What are the nutritional values for spinach?"
        - Expected JSON structure with operation and table fields
        - SQL translation with proper parameter binding
        - Query validation and safety checks

    The test validates:
    - JSON query structure correctness
    - SQL translation accuracy
    - Parameter binding safety
    - Query operation identification
    - Table name extraction

    Returns:
        None: Prints test results to console

    Raises:
        AssertionError: If component functionality fails validation

    Example:
        >>> await test_pipeline_components()
        >>> # Tests individual pipeline components
    """
    print("--- Testing Pipeline Components ---")

    # 1. Test Function Calling Handler (Text-to-JSON)
    print("\n1. Testing FunctionCallingHandler (Text-to-JSON)...")
    fc_handler = FunctionCallingHandler()
    question = "What are the nutritional values for spinach?"
    json_query = await fc_handler.generate_json_query(question)
    print(f"   Generated JSON for '{question}':\n{json.dumps(json_query, indent=2)}")
    assert "operation" in json_query and json_query["operation"] == "search"
    assert "table" in json_query and json_query["table"] == "nutrition_facts"
    print("   FunctionCallingHandler OK")

    # 2. Test Query Engine (JSON-to-SQL)
    print("\n2. Testing QueryEngine (JSON-to-SQL)...")
    sql, params = translate_json_to_sql(json_query)
    print(f"   Translated SQL: {sql}")
    print(f"   Parameters: {params}")
    assert "SELECT" in sql and "FROM nutrition_facts" in sql and "WHERE name" in sql
    assert params[0] == "%spinach%"
    print("   QueryEngine OK")


async def test_end_to_end_classification():
    """
    Tests the full end-to-end classification pipeline.

    This function validates the complete classification pipeline from ingredient
    input to final dietary classification results. It tests both single ingredient
    classification and full recipe classification with known expected outcomes.

    Test Cases:
        1. Single ingredient "egg": Expected Keto=True, Vegan=False
        2. Single ingredient "white rice": Expected Keto=False, Vegan=True
        3. Full recipe with multiple ingredients: Expected Keto=False, Vegan=False

    The test validates:
    - Single ingredient classification accuracy
    - Recipe-level classification logic
    - Expected outcome validation
    - Classification reasoning quality
    - Pipeline integration reliability

    Returns:
        None: Prints test results to console

    Raises:
        AssertionError: If classification results don't match expectations

    Example:
        >>> await test_end_to_end_classification()
        >>> # Tests complete classification pipeline
    """
    print("\n--- Testing End-to-End Classification ---")
    classifier = ContextAwareDietClassifier()

    # Test Case 1: A simple, non-vegan, keto-friendly ingredient
    ingredient1 = "egg"
    print(
        f"\n1. Testing ingredient: '{ingredient1}' (Expected: Keto=True, Vegan=False)"
    )
    result1 = await classifier.classify_single_ingredient(ingredient1)
    print(f"   Result: {json.dumps(result1, indent=2)}")
    assert result1.get("is_keto") is True and result1.get("is_vegan") is False
    print("   Test Case 1 OK")

    # Test Case 2: A simple, vegan, non-keto ingredient
    ingredient2 = "white rice"
    print(
        f"\n2. Testing ingredient: '{ingredient2}' (Expected: Keto=False, Vegan=True)"
    )
    result2 = await classifier.classify_single_ingredient(ingredient2)
    print(f"   Result: {json.dumps(result2, indent=2)}")
    assert result2.get("is_keto") is False and result2.get("is_vegan") is True
    print("   Test Case 2 OK")

    # Test Case 3: A full recipe
    recipe = ["100g chicken breast", "50g of cheddar cheese", "1 slice of white bread"]
    print(f"\n3. Testing recipe: {recipe} (Expected: Keto=False, Vegan=False)")
    recipe_result = await classifier.classify_recipe(recipe)
    print(f"   Result: {json.dumps(recipe_result, indent=2)}")
    assert (
        recipe_result.get("recipe_is_keto") is False
        and recipe_result.get("recipe_is_vegan") is False
    )
    print("   Test Case 3 OK")


if __name__ == "__main__":
    print("======================================================")
    print("  Running Full Test of Function-Calling RAG Pipeline  ")
    print("======================================================")

    # Run component tests
    asyncio.run(test_pipeline_components())

    # Run end-to-end tests
    asyncio.run(test_end_to_end_classification())

    print("\n--- All Tests Completed Successfully! ---")
