#!/usr/bin/env python3
"""
Phase 6 Validation: Performance Benchmarking KPIs

Validates system performance against throughput and latency benchmarks.
"""
import sys
import time
import statistics
from pathlib import Path
import logging
import json
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

# Performance Benchmarks for Phase 6
BENCHMARKS = {
    'min_recipes_per_second': 2.0,
    'max_avg_latency_ms': 500.0,
    'max_memory_usage_gb': 2.0,
    'max_error_rate': 0.05,  # 5%
    'min_concurrent_requests': 5,
    'max_p95_latency_ms': 1000.0,  # 95th percentile
}

def generate_test_recipes(count=50):
    """Generate test recipes for performance testing."""
    
    # Base ingredients for different recipe types
    keto_ingredients = ["chicken breast", "spinach", "olive oil", "avocado", "cheese", "eggs"]
    vegan_ingredients = ["quinoa", "black beans", "tomatoes", "lettuce", "olive oil", "nutritional yeast"]
    mixed_ingredients = ["pasta", "ground beef", "onions", "garlic", "tomato sauce", "parmesan cheese"]
    
    recipes = []
    
    for i in range(count):
        if i % 3 == 0:
            # Keto recipe
            ingredients = keto_ingredients[:3] + [keto_ingredients[i % len(keto_ingredients)]]
        elif i % 3 == 1:
            # Vegan recipe
            ingredients = vegan_ingredients[:3] + [vegan_ingredients[i % len(vegan_ingredients)]]
        else:
            # Mixed recipe
            ingredients = mixed_ingredients[:3] + [mixed_ingredients[i % len(mixed_ingredients)]]
        
        # Convert to JSON string format (original submission format)
        recipe_json = json.dumps(ingredients)
        recipes.append({
            'id': f'test_recipe_{i}',
            'ingredients': recipe_json,
            'ingredients_list': ingredients
        })
    
    return recipes

def measure_single_classification_performance():
    """Measure performance of single ingredient classification."""
    from diet_classifiers import is_ingredient_keto, is_ingredient_vegan
    
    logger.info("Measuring single ingredient classification performance...")
    
    test_ingredients = [
        "chicken breast", "spinach", "olive oil", "butter", "flour", 
        "eggs", "milk", "sugar", "avocado", "broccoli", "salmon", 
        "quinoa", "black beans", "cheese", "tomatoes"
    ]
    
    keto_times = []
    vegan_times = []
    errors = 0
    
    for ingredient in test_ingredients:
        # Measure keto classification
        try:
            start_time = time.time()
            result = is_ingredient_keto(ingredient)
            keto_time = (time.time() - start_time) * 1000  # Convert to ms
            keto_times.append(keto_time)
        except Exception as e:
            logger.error(f"Keto classification failed for {ingredient}: {e}")
            errors += 1
            keto_times.append(5000)  # 5 second penalty
        
        # Measure vegan classification
        try:
            start_time = time.time()
            result = is_ingredient_vegan(ingredient)
            vegan_time = (time.time() - start_time) * 1000  # Convert to ms
            vegan_times.append(vegan_time)
        except Exception as e:
            logger.error(f"Vegan classification failed for {ingredient}: {e}")
            errors += 1
            vegan_times.append(5000)  # 5 second penalty
    
    return {
        'keto_avg_latency_ms': statistics.mean(keto_times),
        'keto_p95_latency_ms': statistics.quantiles(keto_times, n=20)[18] if len(keto_times) > 1 else keto_times[0],
        'vegan_avg_latency_ms': statistics.mean(vegan_times),
        'vegan_p95_latency_ms': statistics.quantiles(vegan_times, n=20)[18] if len(vegan_times) > 1 else vegan_times[0],
        'total_ingredients_tested': len(test_ingredients),
        'single_ingredient_errors': errors
    }

def measure_recipe_classification_performance():
    """Measure performance of full recipe classification."""
    from diet_classifiers import is_keto, is_vegan
    
    logger.info("Measuring recipe classification performance...")
    
    # Generate test recipes
    test_recipes = generate_test_recipes(30)
    
    processing_times = []
    throughput_measurements = []
    errors = 0
    
    # Measure individual recipe processing
    for recipe in test_recipes:
        start_time = time.time()
        
        try:
            keto_result = is_keto(recipe['ingredients'])
            vegan_result = is_vegan(recipe['ingredients'])
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            processing_times.append(processing_time)
            
        except Exception as e:
            logger.error(f"Recipe classification failed for {recipe['id']}: {e}")
            errors += 1
            processing_times.append(30000)  # 30 second penalty
    
    # Measure throughput (batch processing)
    batch_start = time.time()
    batch_recipes = test_recipes[:10]  # Smaller batch for throughput test
    batch_errors = 0
    
    for recipe in batch_recipes:
        try:
            is_keto(recipe['ingredients'])
            is_vegan(recipe['ingredients'])
        except:
            batch_errors += 1
    
    batch_time = time.time() - batch_start
    recipes_per_second = len(batch_recipes) / batch_time if batch_time > 0 else 0
    
    return {
        'avg_recipe_latency_ms': statistics.mean(processing_times),
        'p95_recipe_latency_ms': statistics.quantiles(processing_times, n=20)[18] if len(processing_times) > 1 else processing_times[0],
        'max_recipe_latency_ms': max(processing_times),
        'recipes_per_second': recipes_per_second,
        'total_recipes_tested': len(test_recipes),
        'recipe_errors': errors,
        'batch_errors': batch_errors,
        'error_rate': (errors + batch_errors) / (len(test_recipes) + len(batch_recipes))
    }

def measure_memory_usage():
    """Measure memory usage during classification."""
    logger.info("Measuring memory usage...")
    
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
    
    # Memory during classification
    from diet_classifiers import is_keto, is_vegan
    
    test_recipes = generate_test_recipes(20)
    memory_measurements = []
    
    for recipe in test_recipes[:10]:  # Test subset
        try:
            is_keto(recipe['ingredients'])
            is_vegan(recipe['ingredients'])
            
            current_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
            memory_measurements.append(current_memory)
            
        except Exception as e:
            logger.error(f"Memory test failed: {e}")
    
    peak_memory = max(memory_measurements) if memory_measurements else baseline_memory
    avg_memory = statistics.mean(memory_measurements) if memory_measurements else baseline_memory
    
    return {
        'baseline_memory_gb': baseline_memory,
        'peak_memory_gb': peak_memory,
        'avg_memory_gb': avg_memory,
        'memory_increase_gb': peak_memory - baseline_memory
    }

def test_input_format_performance():
    """Test performance with different input formats."""
    from diet_classifiers import is_keto, is_vegan
    
    logger.info("Testing input format performance...")
    
    # Same recipe in different formats
    ingredients_json = '["chicken breast", "spinach", "olive oil"]'
    ingredients_csv = "chicken breast, spinach, olive oil"
    ingredients_list = ["chicken breast", "spinach", "olive oil"]
    
    formats = [
        ("JSON", ingredients_json),
        ("CSV", ingredients_csv), 
        ("List", ingredients_list)
    ]
    
    format_performance = {}
    
    for format_name, ingredients in formats:
        times = []
        
        for _ in range(5):  # Multiple runs for average
            start_time = time.time()
            try:
                is_keto(ingredients)
                is_vegan(ingredients)
                processing_time = (time.time() - start_time) * 1000
                times.append(processing_time)
            except Exception as e:
                logger.error(f"Format {format_name} failed: {e}")
                times.append(5000)  # Penalty
        
        format_performance[format_name] = {
            'avg_time_ms': statistics.mean(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times)
        }
        
        logger.info(f"  {format_name}: {statistics.mean(times):.1f}ms avg")
    
    return format_performance

def main():
    """Run all Phase 6 validations."""
    logger.info("="*60)
    logger.info("PHASE 6 VALIDATION: Performance Benchmarking KPIs")
    logger.info("="*60)
    
    try:
        # Single ingredient performance
        single_results = measure_single_classification_performance()
        
        # Recipe classification performance
        recipe_results = measure_recipe_classification_performance()
        
        # Memory usage
        memory_results = measure_memory_usage()
        
        # Input format performance
        format_results = test_input_format_performance()
        
        # Combine results
        all_results = {**single_results, **recipe_results, **memory_results}
        
        # Evaluate against benchmarks
        logger.info("="*60)
        logger.info("PHASE 6 BENCHMARK EVALUATION")
        logger.info("="*60)
        
        benchmarks_met = []
        
        # Throughput benchmark
        throughput_met = all_results['recipes_per_second'] >= BENCHMARKS['min_recipes_per_second']
        logger.info(f"{'✅' if throughput_met else '⚠️'} Throughput: {all_results['recipes_per_second']:.2f} recipes/sec "
                   f"(benchmark: {BENCHMARKS['min_recipes_per_second']:.2f})")
        benchmarks_met.append(throughput_met)
        
        # Latency benchmarks
        avg_latency_met = all_results['avg_recipe_latency_ms'] <= BENCHMARKS['max_avg_latency_ms']
        p95_latency_met = all_results['p95_recipe_latency_ms'] <= BENCHMARKS['max_p95_latency_ms']
        
        logger.info(f"{'✅' if avg_latency_met else '⚠️'} Avg Latency: {all_results['avg_recipe_latency_ms']:.1f}ms "
                   f"(benchmark: {BENCHMARKS['max_avg_latency_ms']:.1f}ms)")
        logger.info(f"{'✅' if p95_latency_met else '⚠️'} P95 Latency: {all_results['p95_recipe_latency_ms']:.1f}ms "
                   f"(benchmark: {BENCHMARKS['max_p95_latency_ms']:.1f}ms)")
        
        benchmarks_met.extend([avg_latency_met, p95_latency_met])
        
        # Memory benchmark
        memory_met = all_results['peak_memory_gb'] <= BENCHMARKS['max_memory_usage_gb']
        logger.info(f"{'✅' if memory_met else '⚠️'} Peak Memory: {all_results['peak_memory_gb']:.2f}GB "
                   f"(benchmark: {BENCHMARKS['max_memory_usage_gb']:.2f}GB)")
        benchmarks_met.append(memory_met)
        
        # Error rate benchmark
        error_rate_met = all_results['error_rate'] <= BENCHMARKS['max_error_rate']
        logger.info(f"{'✅' if error_rate_met else '⚠️'} Error Rate: {all_results['error_rate']:.3f} "
                   f"(benchmark: {BENCHMARKS['max_error_rate']:.3f})")
        benchmarks_met.append(error_rate_met)
        
        # Format performance summary
        logger.info("\nInput Format Performance:")
        for format_name, perf in format_results.items():
            logger.info(f"  {format_name}: {perf['avg_time_ms']:.1f}ms avg")
        
        # Overall status
        all_benchmarks_met = all(benchmarks_met)
        benchmarks_passed = sum(benchmarks_met)
        total_benchmarks = len(benchmarks_met)
        
        logger.info("="*60)
        logger.info(f"PHASE 6 SUMMARY: {benchmarks_passed}/{total_benchmarks} benchmarks met")
        
        if all_benchmarks_met:
            logger.info("🎉 ALL PHASE 6 BENCHMARKS MET")
            return 0
        else:
            logger.warning("⚠️ Some benchmarks not met - continuing with warnings")
            return 0  # Continue with warnings as requested
            
    except Exception as e:
        logger.error(f"❌ Phase 6 validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 