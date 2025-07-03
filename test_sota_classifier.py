#!/usr/bin/env python3
"""
Test script for SOTA Semantic Classifier

Tests the complete pipeline:
Recipe Ingredient → ingredient-parser → Arctic Semantic SQL → Database → Fuzzy Fallback → Qwen
"""

import asyncio
import json
import time
from context_aware_classifier import SOTASemanticClassifier

async def test_single_ingredients():
    """Test individual ingredient classification with realistic recipe ingredients."""
    classifier = SOTASemanticClassifier()
    
    # Test ingredients that require semantic matching
    test_cases = [
        {
            "ingredient": "3 pounds pork shoulder, cut into chunks",
            "expected_extraction": "pork shoulder", 
            "expected_keto": True,  # Meat is typically keto
            "expected_vegan": False  # Meat is not vegan
        },
        {
            "ingredient": "2 tbsp extra virgin olive oil",
            "expected_extraction": "olive oil",
            "expected_keto": True,  # Pure fat is keto
            "expected_vegan": True   # Plant oil is vegan
        },
        {
            "ingredient": "1 cup diced tomatoes", 
            "expected_extraction": "tomatoes",
            "expected_keto": False,  # Tomatoes have carbs
            "expected_vegan": True   # Vegetables are vegan
        },
        {
            "ingredient": "100g fresh spinach leaves",
            "expected_extraction": "spinach",
            "expected_keto": True,   # Low carb vegetable
            "expected_vegan": True   # Vegetables are vegan
        },
        {
            "ingredient": "2 cups whole milk",
            "expected_extraction": "milk",
            "expected_keto": False,  # Milk has lactose (carbs)
            "expected_vegan": False  # Dairy is not vegan
        }
    ]
    
    print("TESTING SOTA SEMANTIC CLASSIFIER")
    print("=" * 60)
    
    results = []
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        ingredient = test_case["ingredient"]
        print(f"\n{i}. Testing: {ingredient}")
        print("-" * 40)
        
        start_time = time.time()
        result = await classifier.classify_single_ingredient(ingredient)
        end_time = time.time()
        classification_time = end_time - start_time
        total_time += classification_time
        
        # Extract results
        extracted_name = result.get("extracted_ingredient", "")
        is_keto = result.get("is_keto", False)
        is_vegan = result.get("is_vegan", False)
        confidence = result.get("confidence", "unknown")
        semantic_quality = result.get("semantic_match_quality", "unknown")
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # Check for errors
        if "error" in result:
            print(f"ERROR: {result['error']}")
            results.append({"ingredient": ingredient, "status": "error", "error": result["error"]})
            continue
        
        # Display results
        print(f"Extracted: '{ingredient}' → '{extracted_name}'")
        print(f"Keto: {is_keto} (expected: {test_case['expected_keto']})")
        print(f"Vegan: {is_vegan} (expected: {test_case['expected_vegan']})")
        print(f"Confidence: {confidence}")
        print(f"Semantic Quality: {semantic_quality}")
        print(f"Time: {classification_time:.2f}s")
        print(f"Reasoning: {reasoning}")
        
        # Check accuracy
        keto_correct = is_keto == test_case["expected_keto"]
        vegan_correct = is_vegan == test_case["expected_vegan"]
        overall_correct = keto_correct and vegan_correct
        
        status_emoji = "✅" if overall_correct else "⚠️"
        print(f"{status_emoji} Overall: {'CORRECT' if overall_correct else 'NEEDS REVIEW'}")
        
        results.append({
            "ingredient": ingredient,
            "extracted": extracted_name,
            "keto_result": is_keto,
            "keto_expected": test_case["expected_keto"],
            "keto_correct": keto_correct,
            "vegan_result": is_vegan,
            "vegan_expected": test_case["expected_vegan"], 
            "vegan_correct": vegan_correct,
            "confidence": confidence,
            "semantic_quality": semantic_quality,
            "time": classification_time,
            "status": "correct" if overall_correct else "review"
        })
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    correct_count = len([r for r in results if r.get("status") == "correct"])
    total_count = len([r for r in results if r.get("status") != "error"])
    error_count = len([r for r in results if r.get("status") == "error"])
    
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        avg_time = total_time / total_count
        
        print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
        print(f"Time: {avg_time:.2f}s per ingredient")
        print(f"Errors: {error_count}")
        
        # Semantic quality distribution
        quality_dist = {}
        for r in results:
            if r.get("semantic_quality"):
                quality = r["semantic_quality"]
                quality_dist[quality] = quality_dist.get(quality, 0) + 1
        
        print(f"Semantic Quality: {quality_dist}")
    
    # Performance stats
    print(f"\nCLASSIFIER PERFORMANCE:")
    stats = classifier.get_performance_stats()
    print(f"   Cache available: {stats['cache_available']}")
    print(f"   ingredient-parser available: {stats['ingredient_parser_available']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    return results

async def test_recipe_classification():
    """Test complete recipe classification."""
    classifier = SOTASemanticClassifier()
    
    print("\n" + "=" * 60)
    print("🍽️  TESTING RECIPE CLASSIFICATION")
    print("=" * 60)
    
    # Test recipe with mixed keto/vegan compliance
    recipe = [
        "3 pounds pork shoulder, cut into chunks",  # Keto: Yes, Vegan: No
        "2 tbsp extra virgin olive oil",            # Keto: Yes, Vegan: Yes  
        "1 cup diced tomatoes",                     # Keto: No,  Vegan: Yes
        "100g fresh spinach leaves"                 # Keto: Yes, Vegan: Yes
    ]
    
    print("Recipe ingredients:")
    for i, ingredient in enumerate(recipe, 1):
        print(f"  {i}. {ingredient}")
    
    print(f"\nExpected results:")
    print(f"  🥩 Recipe is Keto: False (tomatoes have carbs)")
    print(f"  🌱 Recipe is Vegan: False (pork is animal product)")
    
    start_time = time.time()
    result = await classifier.classify_recipe(recipe)
    end_time = time.time()
    
    print(f"\nRECIPE RESULTS:")
    print(f"  🥩 Keto-friendly: {result['recipe_is_keto']}")
    print(f"  🌱 Vegan-friendly: {result['recipe_is_vegan']}")
    print(f"  📝 Ingredients processed: {result['successful_classifications']}/{result['ingredient_count']}")
    print(f"  ⏱️  Total time: {end_time - start_time:.2f}s")
    
    # Semantic quality breakdown
    quality_dist = result.get('semantic_quality_distribution', {})
    print(f"  🔍 Semantic quality: {quality_dist}")
    
    # Individual ingredient breakdown
    print(f"\n📋 INGREDIENT BREAKDOWN:")
    for detail in result['ingredient_analysis']:
        ingredient = detail['ingredient']
        extracted = detail.get('extracted_name', '')
        keto = detail['is_keto']
        vegan = detail['is_vegan']
        quality = detail.get('semantic_quality', 'unknown')
        
        print(f"  • {ingredient}")
        print(f"    → Extracted: {extracted}")
        print(f"    → Keto: {keto}, Vegan: {vegan} ({quality})")
    
    return result

async def test_basic_functionality():
    """Test basic functionality of the SOTA classifier."""
    print("🧪 Testing SOTA Semantic Classifier")
    print("=" * 50)
    
    classifier = SOTASemanticClassifier()
    
    # Test simple ingredients first
    test_ingredients = [
        "spinach",
        "chicken breast", 
        "olive oil"
    ]
    
    for ingredient in test_ingredients:
        print(f"\n🔍 Testing: {ingredient}")
        
        try:
            start_time = time.time()
            result = await classifier.classify_single_ingredient(ingredient)
            end_time = time.time()
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"SUCCESS Keto: {result.get('is_keto', 'unknown')}")
                print(f"SUCCESS Vegan: {result.get('is_vegan', 'unknown')}")
                print(f"Time: {end_time - start_time:.2f}s")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Test performance stats
    print(f"\n📊 Performance Stats:")
    stats = classifier.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

async def main():
    """Run all tests."""
    print("🚀 STARTING SOTA SEMANTIC CLASSIFIER TESTS")
    
    try:
        # Test individual ingredients
        ingredient_results = await test_single_ingredients()
        
        # Test recipe classification  
        recipe_result = await test_recipe_classification()
        
        # Test basic functionality
        await test_basic_functionality()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 