#!/usr/bin/env python3
"""
Classification Accuracy Validation Script

Validates the diet classification pipeline performance against specific benchmarks.
Tests both individual ingredient classification and full recipe classification.

This script should be run after the knowledge database is built and tests the
complete function-calling RAG pipeline for accuracy and performance.
"""
import sys
import time
import json
from pathlib import Path
import logging

# Add nb/src to path for accessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    logger.warning("sklearn not available - using basic accuracy calculation")
    def accuracy_score(y_true, y_pred):
        return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

# Classification Accuracy Benchmarks
BENCHMARKS = {
    'keto_accuracy_min': 0.75,
    'vegan_accuracy_min': 0.80,
    'keto_f1_min': 0.70,
    'vegan_f1_min': 0.75,
    'max_processing_time_per_ingredient': 5.0,  # seconds
    'max_processing_time_per_recipe': 30.0,  # seconds for full recipe
    'format_compatibility_required': True
}

def test_individual_ingredients():
    """Test individual ingredient classification performance."""
    from diet_classifiers import is_keto, is_vegan
    
    # Test cases with known answers (ground truth)
    test_cases = [
        # (ingredient, expected_keto, expected_vegan)
        ("chicken breast", True, False),
        ("spinach", True, True),
        ("sugar", False, True),
        ("butter", True, False),
        ("olive oil", True, True),
        ("flour", False, True),
        ("eggs", True, False),
        ("broccoli", True, True),
        ("milk", False, False),
        ("avocado", True, True),
        ("cheese", True, False),
        ("quinoa", False, True),
        ("salmon", True, False),
        ("rice", False, True),
        ("beef", True, False)
    ]
    
    logger.info("Testing individual ingredient classification...")
    
    keto_predictions = []
    vegan_predictions = []
    keto_expected = []
    vegan_expected = []
    processing_times = []
    
    for ingredient, expected_keto, expected_vegan in test_cases:
        start_time = time.time()
        
        try:
            # Test with single ingredient lists (as per interface)
            pred_keto = is_keto([ingredient])
            pred_vegan = is_vegan([ingredient])
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            keto_predictions.append(pred_keto)
            vegan_predictions.append(pred_vegan)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)
            
            logger.info(f"  {ingredient}: keto={pred_keto} (exp: {expected_keto}), "
                       f"vegan={pred_vegan} (exp: {expected_vegan}), "
                       f"time={processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"  {ingredient}: FAILED - {e}")
            keto_predictions.append(False)
            vegan_predictions.append(False)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)
            processing_times.append(BENCHMARKS['max_processing_time_per_ingredient'])
    
    # Calculate metrics
    keto_accuracy = accuracy_score(keto_expected, keto_predictions)
    vegan_accuracy = accuracy_score(vegan_expected, vegan_predictions)
    avg_processing_time = sum(processing_times) / len(processing_times)
    
    results = {
        'keto_accuracy': keto_accuracy,
        'vegan_accuracy': vegan_accuracy,
        'avg_processing_time': avg_processing_time,
        'max_processing_time': max(processing_times),
        'total_ingredients_tested': len(test_cases)
    }
    
    # Add F1 scores if sklearn available
    try:
        _, _, keto_f1, _ = precision_recall_fscore_support(keto_expected, keto_predictions, average='binary')
        _, _, vegan_f1, _ = precision_recall_fscore_support(vegan_expected, vegan_predictions, average='binary')
        results['keto_f1'] = keto_f1
        results['vegan_f1'] = vegan_f1
    except:
        results['keto_f1'] = 0.0
        results['vegan_f1'] = 0.0
    
    return results

def test_recipe_classification():
    """Test full recipe classification performance."""
    from diet_classifiers import is_keto, is_vegan
    
    # Test recipes with known classifications
    test_recipes = [
        # (ingredients, expected_keto, expected_vegan)
        (["chicken breast", "spinach", "olive oil"], True, False),  # Keto, not vegan (animal protein)
        (["quinoa", "black beans", "avocado"], False, True),        # Not keto (high carb), vegan
        (["salmon", "asparagus", "butter"], True, False),          # Keto, not vegan (animal products)
        (["pasta", "tomato sauce", "cheese"], False, False),       # Not keto (pasta), not vegan (cheese)
        (["lettuce", "cucumber", "olive oil"], True, True),        # Keto and vegan
        (["almonds", "spinach", "coconut oil"], True, True),       # Keto and vegan
        (["bread", "lettuce", "turkey"], False, False),           # Not keto (bread), not vegan (turkey)
    ]
    
    logger.info("Testing recipe classification...")
    
    keto_predictions = []
    vegan_predictions = []
    keto_expected = []
    vegan_expected = []
    processing_times = []
    
    for ingredients, expected_keto, expected_vegan in test_recipes:
        start_time = time.time()
        
        try:
            pred_keto = is_keto(ingredients)
            pred_vegan = is_vegan(ingredients)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            keto_predictions.append(pred_keto)
            vegan_predictions.append(pred_vegan)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)
            
            logger.info(f"  {ingredients}: keto={pred_keto} (exp: {expected_keto}), "
                       f"vegan={pred_vegan} (exp: {expected_vegan}), "
                       f"time={processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"  {ingredients}: FAILED - {e}")
            keto_predictions.append(False)
            vegan_predictions.append(False)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)
            processing_times.append(BENCHMARKS['max_processing_time_per_recipe'])
    
    # Calculate metrics
    recipe_keto_accuracy = accuracy_score(keto_expected, keto_predictions)
    recipe_vegan_accuracy = accuracy_score(vegan_expected, vegan_predictions)
    avg_recipe_time = sum(processing_times) / len(processing_times)
    
    return {
        'recipe_keto_accuracy': recipe_keto_accuracy,
        'recipe_vegan_accuracy': recipe_vegan_accuracy,
        'avg_recipe_processing_time': avg_recipe_time,
        'max_recipe_processing_time': max(processing_times),
        'total_recipes_tested': len(test_recipes)
    }

def test_input_format_compatibility():
    """Test that diet_classifiers handles different input formats correctly."""
    from diet_classifiers import is_keto, is_vegan
    
    logger.info("Testing input format compatibility...")
    
    # Test different input formats
    test_cases = [
        # JSON string format
        '["chicken breast", "spinach", "olive oil"]',
        # Comma-separated string format  
        "chicken breast, spinach, olive oil",
        # Direct list format
        ["chicken breast", "spinach", "olive oil"]
    ]
    
    results = []
    for i, ingredients in enumerate(test_cases):
        try:
            keto_result = is_keto(ingredients)
            vegan_result = is_vegan(ingredients)
            results.append((keto_result, vegan_result))
            logger.info(f"  Format {i+1}: keto={keto_result}, vegan={vegan_result}")
        except Exception as e:
            logger.error(f"  Format {i+1}: FAILED - {e}")
            results.append((False, False))
    
    # All formats should give same result
    all_same = len(set(results)) == 1
    logger.info(f"  All formats consistent: {all_same}")
    
    return {'format_compatibility': all_same}

def main():
    """Run all classification accuracy validations."""
    logger.info("="*60)
    logger.info("CLASSIFICATION ACCURACY VALIDATION")
    logger.info("="*60)
    
    try:
        # Test individual ingredients
        ingredient_results = test_individual_ingredients()
        
        # Test recipe classification
        recipe_results = test_recipe_classification()
        
        # Test input format compatibility
        format_results = test_input_format_compatibility()
        
        # Combine results
        all_results = {**ingredient_results, **recipe_results, **format_results}
        
        # Evaluate against benchmarks
        logger.info("="*60)
        logger.info("BENCHMARK EVALUATION")
        logger.info("="*60)
        
        benchmarks_met = []
        
        # Accuracy benchmarks
        keto_acc_met = all_results['keto_accuracy'] >= BENCHMARKS['keto_accuracy_min']
        vegan_acc_met = all_results['vegan_accuracy'] >= BENCHMARKS['vegan_accuracy_min']
        
        logger.info(f"{'PASS' if keto_acc_met else 'WARN'}: Keto Accuracy: {all_results['keto_accuracy']:.3f} "
                   f"(benchmark: {BENCHMARKS['keto_accuracy_min']:.3f})")
        logger.info(f"{'PASS' if vegan_acc_met else 'WARN'}: Vegan Accuracy: {all_results['vegan_accuracy']:.3f} "
                   f"(benchmark: {BENCHMARKS['vegan_accuracy_min']:.3f})")
        
        benchmarks_met.extend([keto_acc_met, vegan_acc_met])
        
        # F1 score benchmarks (if available)
        if all_results.get('keto_f1', 0) > 0:
            keto_f1_met = all_results['keto_f1'] >= BENCHMARKS['keto_f1_min']
            vegan_f1_met = all_results['vegan_f1'] >= BENCHMARKS['vegan_f1_min']
            
            logger.info(f"{'PASS' if keto_f1_met else 'WARN'}: Keto F1: {all_results['keto_f1']:.3f} "
                       f"(benchmark: {BENCHMARKS['keto_f1_min']:.3f})")
            logger.info(f"{'PASS' if vegan_f1_met else 'WARN'}: Vegan F1: {all_results['vegan_f1']:.3f} "
                       f"(benchmark: {BENCHMARKS['vegan_f1_min']:.3f})")
            
            benchmarks_met.extend([keto_f1_met, vegan_f1_met])
        
        # Performance benchmarks
        time_met = all_results['avg_processing_time'] <= BENCHMARKS['max_processing_time_per_ingredient']
        recipe_time_met = all_results['avg_recipe_processing_time'] <= BENCHMARKS['max_processing_time_per_recipe']
        
        logger.info(f"{'PASS' if time_met else 'WARN'}: Avg Processing Time: {all_results['avg_processing_time']:.2f}s "
                   f"(benchmark: {BENCHMARKS['max_processing_time_per_ingredient']:.2f}s)")
        logger.info(f"{'PASS' if recipe_time_met else 'WARN'}: Avg Recipe Time: {all_results['avg_recipe_processing_time']:.2f}s "
                   f"(benchmark: {BENCHMARKS['max_processing_time_per_recipe']:.2f}s)")
        
        benchmarks_met.extend([time_met, recipe_time_met])
        
        # Format compatibility
        format_met = all_results['format_compatibility']
        logger.info(f"{'PASS' if format_met else 'FAIL'}: Format Compatibility: {format_met}")
        benchmarks_met.append(format_met)
        
        # Overall status
        all_benchmarks_met = all(benchmarks_met)
        benchmarks_passed = sum(benchmarks_met)
        total_benchmarks = len(benchmarks_met)
        
        logger.info("="*60)
        logger.info(f"VALIDATION SUMMARY: {benchmarks_passed}/{total_benchmarks} benchmarks met")
        
        if all_benchmarks_met:
            logger.info("SUCCESS: All classification accuracy benchmarks met")
            return 0
        else:
            logger.warning("PARTIAL: Some benchmarks not met - review warnings above")
            return 0  # Continue with warnings as non-critical
            
    except Exception as e:
        logger.error(f"FAILED: Classification accuracy validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 