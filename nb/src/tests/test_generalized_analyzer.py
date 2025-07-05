#!/usr/bin/env python3
"""
Test script for the integrated GeneralizedIngredientAnalyzer and RecipeClassifier.
This tests the complete multi-stage pipeline from the plan.
"""

import asyncio
import json
import logging
from typing import Dict, Any

from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generalized_ingredient_analyzer import (
    GeneralizedIngredientAnalyzer, 
    RecipeClassifier, 
    opensearch_client,
    ingredient_analyzer,
    judge_llm_handler
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ingredient_analyzer():
    """Test the GeneralizedIngredientAnalyzer with various ingredients."""
    print("\n" + "="*60)
    print("TESTING GENERALIZED INGREDIENT ANALYZER")
    print("="*60)
    
    test_ingredients = [
        "cheddar cheese",
        "almond milk", 
        "chicken breast",
        "tofu",
        "honey",
        "vegetable stock cube"
    ]
    
    for ingredient in test_ingredients:
        print(f"\n--- Testing ingredient: {ingredient} ---")
        try:
            result = await ingredient_analyzer.analyze_ingredient_generalized(
                ingredient, recipe_context="Test recipe context"
            )
            
            print(f"Vegan Status: {result.get('vegan_status')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Reason: {result.get('reason', 'N/A')}")
            
            if result.get('synonyms_found'):
                print(f"Synonyms found: {len(result['synonyms_found'])}")
            
            if result.get('hierarchies_found'):
                print(f"Hierarchies found: {len(result['hierarchies_found'])}")
                
            if result.get('composite_analysis'):
                print(f"Composite analysis: {result['composite_analysis']}")
                
        except Exception as e:
            print(f"Error analyzing {ingredient}: {e}")


async def test_recipe_classifier():
    """Test the RecipeClassifier with sample recipe IDs."""
    print("\n" + "="*60)
    print("TESTING RECIPE CLASSIFIER")
    print("="*60)
    
    classifier = RecipeClassifier(opensearch_client, ingredient_analyzer, judge_llm_handler)
    
    # Test recipe IDs (you may need to adjust these based on your OpenSearch data)
    test_recipe_ids = [
        'KBehQJcBmKMcD7RGUg3P',  # "Genuine Egg Noodles"
        'Ph2hQJcBmKMcD7RGpSI4',  # "Eggs Creole Over Toast"
    ]
    
    for recipe_id in test_recipe_ids:
        print(f"\n--- Testing recipe ID: {recipe_id} ---")
        try:
            result = await classifier.classify_recipe(recipe_id)
            
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Overall Classification: {result.get('overall_classification')}")
            print(f"Vegan Status: {result.get('vegan_status')}")
            print(f"Keto Status: {result.get('keto_status')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            if result.get('aggregated_data'):
                agg_data = result['aggregated_data']
                print(f"Total Net Carbs: {agg_data.get('total_net_carbs_recipe_g', 0):.2f}g")
                print(f"Non-vegan ingredients: {len(agg_data.get('potential_non_vegan_ingredients', []))}")
                print(f"Keto disqualifiers: {len(agg_data.get('potential_keto_disqualifiers', []))}")
                
        except Exception as e:
            print(f"Error classifying recipe {recipe_id}: {e}")


async def test_individual_components():
    """Test individual components of the pipeline."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Test SQL generation and execution
    print("\n--- Testing SQL Generation ---")
    try:
        test_prompt = "Get the first 5 rows from nutrition_facts table"
        results = await ingredient_analyzer._generate_and_execute_with_retry(test_prompt)
        print(f"SQL execution successful, returned {len(results)} rows")
    except Exception as e:
        print(f"SQL generation failed: {e}")
    
    # Test synonym discovery
    print("\n--- Testing Synonym Discovery ---")
    try:
        synonyms = await ingredient_analyzer._discover_synonyms("cheese")
        print(f"Found {len(synonyms)} synonyms for 'cheese'")
        for synonym in synonyms[:3]:  # Show first 3
            print(f"  - {synonym.get('term')}: {synonym.get('aliases', [])}")
    except Exception as e:
        print(f"Synonym discovery failed: {e}")
    
    # Test hierarchy discovery
    print("\n--- Testing Hierarchy Discovery ---")
    try:
        hierarchies = await ingredient_analyzer._discover_hierarchies("cheddar cheese")
        print(f"Found {len(hierarchies)} hierarchies for 'cheddar cheese'")
        for hierarchy in hierarchies[:3]:  # Show first 3
            print(f"  - {hierarchy.get('parent')} -> {hierarchy.get('child')} ({hierarchy.get('relationship')})")
    except Exception as e:
        print(f"Hierarchy discovery failed: {e}")


async def main():
    """Run all tests."""
    print("Starting comprehensive test of Generalized Ingredient Analyzer and Recipe Classifier")
    print("This tests the complete multi-stage pipeline from the plan.")
    
    try:
        # Test individual components first
        await test_individual_components()
        
        # Test ingredient analyzer
        await test_ingredient_analyzer()
        
        # Test recipe classifier (may fail if OpenSearch is not available)
        await test_recipe_classifier()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"\nTest failed: {e}")
        print("This might be due to:")
        print("- OpenSearch not running")
        print("- Database not initialized")
        print("- LLM services not available")
        print("- Missing dependencies")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main()) 