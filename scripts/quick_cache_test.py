#!/usr/bin/env python3
"""
Quick Cache Pre-computation Test

Simplified version that tests our SOTA classifier without complex threading.

This script provides a lightweight test of the SOTA Semantic Classifier
functionality to validate that the classification pipeline is working
correctly before running larger-scale pre-computation tasks.

The script performs basic testing including:
- SOTA Semantic Classifier initialization
- Individual ingredient classification testing
- Result validation and error handling
- Performance monitoring and reporting

Returns:
    int: Exit code (0 for success, 1 for failure)

Raises:
    ImportError: If SOTA classifier module cannot be imported
    Exception: If classification testing fails
"""
import os
import sys
from pathlib import Path

# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

import asyncio

from context_aware_classifier import SOTASemanticClassifier


async def test_cache_precompute():
    """
    Test cache pre-computation on a small sample.
    
    This function tests the SOTA Semantic Classifier with a small set of
    test ingredients to validate functionality and performance before
    running larger-scale pre-computation tasks.
    
    Returns:
        tuple: (successful_count, error_count) - Number of successful and failed tests
        
    Raises:
        ImportError: If SOTA classifier cannot be imported
        Exception: If classification testing fails
    """
    print("SOTA Semantic Classifier Cache Test")
    print("=" * 50)

    classifier = SOTASemanticClassifier()

    # Test ingredients
    test_ingredients = [
        "spinach, raw",
        "chicken breast, skinless",
        "olive oil, extra virgin",
        "sugar, granulated",
        "butter, salted",
    ]

    print(f"Testing {len(test_ingredients)} ingredients...")

    results = []

    for i, ingredient in enumerate(test_ingredients, 1):
        print(f"\n{i}. {ingredient}")

        try:
            result = await classifier.classify_single_ingredient(ingredient)

            if "error" in result:
                print(f"   Error: {result['error']}")
                results.append({"ingredient": ingredient, "status": "error"})
            else:
                keto = result.get("is_keto", False)
                vegan = result.get("is_vegan", False)
                confidence = result.get("confidence", "unknown")
                semantic_quality = result.get("semantic_match_quality", "unknown")

                print(f"   Keto: {keto}")
                print(f"   Vegan: {vegan}")
                print(f"   Confidence: {confidence}")
                print(f"   Quality: {semantic_quality}")

                results.append(
                    {
                        "ingredient": ingredient,
                        "keto": keto,
                        "vegan": vegan,
                        "confidence": confidence,
                        "semantic_quality": semantic_quality,
                        "status": "success",
                    }
                )

        except Exception as e:
            print(f"   Exception: {e}")
            results.append(
                {"ingredient": ingredient, "status": "exception", "error": str(e)}
            )

    # Summary
    successful = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] in ["error", "exception"]]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successful: {len(successful)}/{len(test_ingredients)}")
    print(f"Errors: {len(errors)}/{len(test_ingredients)}")

    if successful:
        print("\nSOTA Classifier is working!")
        print("   Ready for larger-scale pre-computation")
    else:
        print("\nIssues need to be resolved")

    return len(successful), len(errors)


if __name__ == "__main__":
    successful, errors = asyncio.run(test_cache_precompute())
    print(f"\nResult: {successful} successful, {errors} errors")
    sys.exit(0 if successful >= 3 else 1)
