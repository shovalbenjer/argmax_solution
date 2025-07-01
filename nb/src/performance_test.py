"""
Performance Test Suite for Recipe Classification System

This script performs comprehensive performance testing on 5,000 random recipe samples,
measuring throughput, accuracy, and system performance metrics.

Phase 3.2: Performance Test on 5,000 Random Samples
- Fetches 5,000 random recipes from available data sources
- Runs classify_recipe function on each sample
- Measures execution time and calculates throughput
- Displays distribution of predictions
- Generates performance report with visualizations
"""

import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import json
from collections import Counter
import polars as pl

# Performance metrics
class PerformanceMetrics:
    """A container for collecting and calculating performance metrics during a test run.

    Detailed Description:
        - This class provides a structured way to manage metrics for a performance test.
        - It includes methods to handle timing (start, end, and individual recipe times),
          and to collect predictions and errors.
        - It also contains methods to calculate summary statistics like total time,
          average time, and throughput.

    Libraries Used:
        - numpy: Used for calculating the mean of recipe processing times. It's a standard and
          efficient library for numerical operations in Python.
        - time: Used for basic timing operations.
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.recipe_times = []
        self.predictions = []
        self.errors = []
        self.total_recipes = 0
        
    def start_timer(self):
        self.start_time = time.time()
        
    def end_timer(self):
        self.end_time = time.time()
        
    def add_recipe_time(self, recipe_time: float):
        self.recipe_times.append(recipe_time)
        
    def add_prediction(self, prediction: Dict[str, Any]):
        self.predictions.append(prediction)
        
    def add_error(self, error: str):
        self.errors.append(error)
        
    def get_total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time and self.start_time else 0
        
    def get_average_time_per_recipe(self) -> float:
        return np.mean(self.recipe_times) if self.recipe_times else 0
        
    def get_throughput_per_second(self) -> float:
        total_time = self.get_total_time()
        return len(self.recipe_times) / total_time if total_time > 0 else 0

def load_random_recipes(n_samples: int = 5000) -> List[Dict[str, Any]]:
    """Loads a specified number of random recipes for testing.

    Detailed Description:
        - This function is responsible for sourcing the data for the performance test.
        - It first attempts to load recipes from a primary ground truth CSV file.
        - If the number of loaded recipes is less than `n_samples`, it proceeds to generate
          synthetic recipes using a base list of ingredients from a nutrition CSV file.
        - This hybrid approach ensures that the test can run even if the primary data source is small.

    Parameters:
        - n_samples (int): The target number of random recipes to load.

    Returns:
        - List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a recipe.

    Libraries Used:
        - polars: A fast DataFrame library used for reading the CSV files. It is chosen over
          pandas here for its potential performance advantages in I/O operations.
        - pandas: Used to convert a polars DataFrame to a list of names for synthetic data generation.
        - random: For selecting random samples and templates for synthetic data.
        - loguru: For logging the progress of data loading.
    """
    logger.info(f"Loading {n_samples} random recipes from available sources...")
    
    # Try to load from different sources
    data_sources = [
        Path("nb/src/evaluation_data/ground_truth_sample.csv"),
        Path("nb/src/raw_data/nutrition.csv"),
        Path("nb/src/evaluation_data/strict_keto.csv"),
        Path("nb/src/evaluation_data/strict_vegan.csv"),
        Path("nb/src/evaluation_data/borderline_keto.csv")
    ]
    
    all_recipes = []
    
    # Load ground truth sample (main source)
    ground_truth_path = Path("nb/src/evaluation_data/ground_truth_sample.csv")
    if ground_truth_path.exists():
        df = pl.read_csv(ground_truth_path)
        logger.info(f"Loaded {len(df)} recipes from ground_truth_sample.csv")
        
        # Convert to recipe format
        for row in df.to_dicts():
            recipe = {
                "title": row.get("title", f"Recipe_{len(all_recipes)}"),
                "ingredients": row.get("ingredients", "[]"),
                "description": row.get("description", ""),
                "source": "ground_truth_sample"
            }
            all_recipes.append(recipe)
    
    # If we need more recipes, generate synthetic ones based on nutrition data
    if len(all_recipes) < n_samples:
        nutrition_path = Path("nb/src/raw_data/nutrition.csv")
        if nutrition_path.exists():
            nutrition_df = pl.read_csv(nutrition_path)
            logger.info(f"Generating synthetic recipes from {len(nutrition_df)} nutrition entries")
            
            # Generate synthetic recipes
            ingredients_templates = [
                ["1 cup {}", "2 tbsp olive oil", "salt to taste"],
                ["1 lb {}", "1 onion diced", "2 cloves garlic"],
                ["2 cups {}", "1 tsp herbs", "pepper to taste"],
                ["1/2 cup {}", "1 egg", "1 tbsp butter"],
                ["1 can {}", "1 tsp spices", "1 cup water"]
            ]
            
            # Get the first column (likely ingredient names)
            nutrition_names = nutrition_df.to_pandas().iloc[:, 0].tolist()
            
            for i in range(min(n_samples - len(all_recipes), len(nutrition_names) * 3)):
                base_ingredient = random.choice(nutrition_names)
                template = random.choice(ingredients_templates)
                ingredients = [template[0].format(base_ingredient)] + template[1:]
                
                recipe = {
                    "title": f"Synthetic Recipe with {base_ingredient}",
                    "ingredients": str(ingredients),
                    "description": f"A recipe featuring {base_ingredient}",
                    "source": "synthetic"
                }
                all_recipes.append(recipe)
    
    # Randomly sample the requested number
    if len(all_recipes) > n_samples:
        all_recipes = random.sample(all_recipes, n_samples)
    
    logger.info(f"Successfully loaded {len(all_recipes)} recipes for performance testing")
    return all_recipes

def mock_classify_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates the recipe classification function for performance testing.

    Detailed Description:
        - This function acts as a stand-in for the actual, potentially slow, `classify_recipe` function.
        - It introduces a small, random delay to mimic the processing time (latency) of a real model inference.
        - It uses a simple, rule-based heuristic based on keywords in the recipe's title and ingredients
          to generate a plausible-looking classification (vegan/keto).
        - This allows for testing the throughput and performance of the overall system without the
          computational expense or variability of a real ML model.

    Parameters:
        - recipe (Dict[str, Any]): A dictionary representing the recipe to be classified.

    Returns:
        - Dict[str, Any]: A dictionary containing the mock classification results, including confidence scores
          and the simulated processing time.
    """
    # Simulate processing time (actual classifier would take time for inference)
    time.sleep(random.uniform(0.001, 0.005))  # 1-5ms per recipe
    
    # Mock classification logic based on ingredients
    ingredients_str = str(recipe.get("ingredients", "")).lower()
    title = recipe.get("title", "").lower()
    
    # Simple heuristic for mock classification
    vegan_indicators = ["vegetable", "fruit", "bean", "rice", "pasta", "salad", "vegan"]
    keto_indicators = ["meat", "fish", "cheese", "egg", "butter", "oil", "avocado"]
    non_keto_indicators = ["bread", "flour", "sugar", "pasta", "rice", "potato"]
    
    # Calculate scores
    vegan_score = sum(1 for indicator in vegan_indicators 
                     if indicator in ingredients_str or indicator in title)
    keto_score = sum(1 for indicator in keto_indicators 
                    if indicator in ingredients_str or indicator in title)
    non_keto_score = sum(1 for indicator in non_keto_indicators 
                        if indicator in ingredients_str or indicator in title)
    
    # Add some randomness for realistic distribution
    vegan_score += random.uniform(-0.5, 0.5)
    keto_score += random.uniform(-0.5, 0.5)
    
    # Determine classifications
    is_vegan = vegan_score > 0.5 and "meat" not in ingredients_str and "cheese" not in ingredients_str
    is_keto = keto_score > non_keto_score and non_keto_score < 1
    
    return {
        "recipe_title": recipe.get("title"),
        "is_vegan": is_vegan,
        "is_keto": is_keto,
        "vegan_confidence": min(max(vegan_score / 3, 0), 1),
        "keto_confidence": min(max(keto_score / 3, 0), 1),
        "processing_time": random.uniform(0.001, 0.005)
    }

def run_performance_test(n_samples: int = 5000, save_results: bool = True) -> Dict[str, Any]:
    """Runs the core performance test on a sample of recipes.

    Detailed Description:
        - This function orchestrates the main performance test loop.
        - It initializes the `PerformanceMetrics` container.
        - It loads the recipe data using `load_random_recipes`.
        - It then iterates through each recipe, calling `mock_classify_recipe` to get a simulated
          classification and timing each call.
        - After the loop, it calculates and aggregates all performance statistics and prediction distributions.

    Parameters:
        - n_samples (int): The number of recipes to test.
        - save_results (bool): If True, the results will be saved to files.

    Returns:
        - Dict[str, Any]: A comprehensive dictionary containing all performance metrics,
          test parameters, and prediction distributions.
    """
    logger.info(f"🚀 Starting Performance Test on {n_samples} Random Samples")
    
    # Initialize metrics
    metrics = PerformanceMetrics()
    
    # Load random recipes
    recipes = load_random_recipes(n_samples)
    actual_samples = len(recipes)
    
    logger.info(f"📊 Testing {actual_samples} recipes...")
    
    # Start performance test
    metrics.start_timer()
    
    # Process each recipe
    for i, recipe in enumerate(recipes):
        try:
            # Time individual recipe processing
            recipe_start = time.time()
            
            # Classify recipe (using mock function for testing)
            from nb.src.diet_classifiers import classify_recipe_with_context # Assuming this function exists
            result = classify_recipe_with_context(recipe)             
            recipe_end = time.time()
            recipe_time = recipe_end - recipe_start
            
            # Record metrics
            metrics.add_recipe_time(recipe_time)
            metrics.add_prediction(result)
            
            # Progress logging
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{actual_samples} recipes...")
                
        except Exception as e:
            logger.error(f"Error processing recipe {i}: {e}")
            metrics.add_error(str(e))
    
    # End performance test
    metrics.end_timer()
    
    # Calculate performance statistics
    total_time = metrics.get_total_time()
    avg_time_per_recipe = metrics.get_average_time_per_recipe()
    throughput = metrics.get_throughput_per_second()
    
    # Analyze predictions
    vegan_predictions = [p["is_vegan"] for p in metrics.predictions]
    keto_predictions = [p["is_keto"] for p in metrics.predictions]
    
    vegan_count = sum(vegan_predictions)
    keto_count = sum(keto_predictions)
    both_count = sum(1 for i in range(len(vegan_predictions)) 
                    if vegan_predictions[i] and keto_predictions[i])
    
    # Compile results
    results = {
        "test_parameters": {
            "requested_samples": n_samples,
            "actual_samples": actual_samples,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance_metrics": {
            "total_execution_time_seconds": total_time,
            "average_time_per_recipe_seconds": avg_time_per_recipe,
            "throughput_recipes_per_second": throughput,
            "successful_classifications": len(metrics.predictions),
            "errors": len(metrics.errors),
            "success_rate": len(metrics.predictions) / actual_samples * 100
        },
        "prediction_distribution": {
            "total_vegan": vegan_count,
            "total_keto": keto_count,
            "both_vegan_and_keto": both_count,
            "vegan_percentage": vegan_count / len(vegan_predictions) * 100,
            "keto_percentage": keto_count / len(keto_predictions) * 100,
            "both_percentage": both_count / len(vegan_predictions) * 100
        },
        "timing_statistics": {
            "min_recipe_time": min(metrics.recipe_times) if metrics.recipe_times else 0,
            "max_recipe_time": max(metrics.recipe_times) if metrics.recipe_times else 0,
            "median_recipe_time": np.median(metrics.recipe_times) if metrics.recipe_times else 0,
            "std_recipe_time": np.std(metrics.recipe_times) if metrics.recipe_times else 0
        }
    }
    
    # Log summary
    logger.success("🎯 Performance Test Complete!")
    logger.info(f"📈 Processed {actual_samples} recipes in {total_time:.2f} seconds")
    logger.info(f"⚡ Average time per recipe: {avg_time_per_recipe*1000:.2f}ms")
    logger.info(f"🔥 Throughput: {throughput:.1f} recipes/second")
    logger.info(f"🌱 Vegan recipes: {vegan_count} ({vegan_count/len(vegan_predictions)*100:.1f}%)")
    logger.info(f"🥓 Keto recipes: {keto_count} ({keto_count/len(keto_predictions)*100:.1f}%)")
    logger.info(f"✨ Both vegan & keto: {both_count} ({both_count/len(vegan_predictions)*100:.1f}%)")
    
    if save_results:
        save_performance_results(results, metrics)
    
    return results

def save_performance_results(results: Dict[str, Any], metrics: PerformanceMetrics):
    """Saves performance test results to a JSON file and triggers visualization.

    Detailed Description:
        - This function handles the output of the performance test.
        - It creates the output directory if it doesn't exist.
        - It saves the detailed results dictionary to a `performance_test_results.json` file.
        - It then calls the functions to generate the visual charts and the text summary report.

    Parameters:
        - results (Dict[str, Any]): The dictionary of results from `run_performance_test`.
        - metrics (PerformanceMetrics): The metrics object containing raw timing data needed for visualizations.
    """
    
    output_dir = Path("nb/src/data/performance_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save JSON results
    json_path = output_dir / "performance_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Saved results to {json_path}")
    
    # Generate visualizations
    create_performance_visualizations(results, metrics, output_dir)

def create_performance_visualizations(results: Dict[str, Any], metrics: PerformanceMetrics, output_dir: Path):
    """Generates and saves a set of plots summarizing the performance results.

    Detailed Description:
        - This function creates a 2x2 subplot figure to visualize the performance test results.
        - The visualizations include:
            1. A histogram of recipe processing times.
            2. A pie chart showing the distribution of dietary classifications.
            3. A bar chart of key performance metrics (throughput, latency).
            4. A line plot of cumulative processing time.
        - This provides a quick, visual summary of the system's performance.

    Parameters:
        - results (Dict[str, Any]): The main results dictionary.
        - metrics (PerformanceMetrics): The metrics object containing the raw timing data.
        - output_dir (Path): The directory where the output PNG file will be saved.

    Libraries Used:
        - matplotlib & seaborn: These are standard libraries for data visualization in Python,
          used here to create clear and informative plots.
    """
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Recipe Classification Performance Test Results', fontsize=16, fontweight='bold')
    
    # 1. Processing Time Distribution
    ax1.hist(metrics.recipe_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Processing Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Recipe Processing Times')
    ax1.axvline(np.mean(metrics.recipe_times), color='red', linestyle='--', 
               label=f'Mean: {np.mean(metrics.recipe_times)*1000:.1f}ms')
    ax1.legend()
    
    # 2. Prediction Distribution Pie Chart
    pred_dist = results["prediction_distribution"]
    vegan_only = pred_dist["total_vegan"] - pred_dist["both_vegan_and_keto"]
    keto_only = pred_dist["total_keto"] - pred_dist["both_vegan_and_keto"]
    both = pred_dist["both_vegan_and_keto"]
    neither = results["test_parameters"]["actual_samples"] - vegan_only - keto_only - both
    
    labels = ['Vegan Only', 'Keto Only', 'Both', 'Neither']
    sizes = [vegan_only, keto_only, both, neither]
    colors = ['lightgreen', 'lightcoral', 'gold', 'lightgray']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Recipe Classification Distribution')
    
    # 3. Performance Metrics Bar Chart
    metrics_names = ['Throughput\n(recipes/sec)', 'Avg Time\n(ms)', 'Success Rate\n(%)']
    metrics_values = [
        results["performance_metrics"]["throughput_recipes_per_second"],
        results["performance_metrics"]["average_time_per_recipe_seconds"] * 1000,
        results["performance_metrics"]["success_rate"]
    ]
    
    bars = ax3.bar(metrics_names, metrics_values, color=['lightblue', 'lightgreen', 'lightyellow'])
    ax3.set_title('Key Performance Metrics')
    ax3.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 4. Cumulative Processing Time
    cumulative_times = np.cumsum(metrics.recipe_times)
    ax4.plot(range(len(cumulative_times)), cumulative_times, color='purple', linewidth=2)
    ax4.set_xlabel('Recipe Number')
    ax4.set_ylabel('Cumulative Time (seconds)')
    ax4.set_title('Cumulative Processing Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot of individual recipe times vs. throughput at that point in time
    throughputs = [i / (metrics.recipe_times[i] + 1e-9) for i in range(1, len(metrics.recipe_times))]
    latencies_ms = [t * 1000 for t in metrics.recipe_times[1:]]

    sns.scatterplot(x=latencies_ms, y=throughputs, alpha=0.5)
    plt.title('Performance Benchmark: Latency vs. Throughput', fontsize=18, weight='bold')
    plt.xlabel('Latency per Recipe (ms)')
    plt.ylabel('Instantaneous Throughput (recipes/sec)')
    plt.xscale('log') # Latency often has a long tail, log scale helps visualization
    
    # Add average lines
    avg_latency = results["performance_metrics"]["average_time_per_recipe_seconds"] * 1000
    avg_throughput = results["performance_metrics"]["throughput_recipes_per_second"]
    plt.axvline(avg_latency, color='r', linestyle='--', label=f'Avg Latency: {avg_latency:.2f} ms')
    plt.axhline(avg_throughput, color='g', linestyle='--', label=f'Avg Throughput: {avg_throughput:.2f} rps')
    plt.legend()
    
    viz_path = output_dir / "sota_performance_benchmark.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Saved SOTA benchmark visualization to {viz_path}")
    plt.show()

    # Save visualization
    viz_path = output_dir / "performance_test_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Saved visualization to {viz_path}")
    plt.close()
    
    # Create summary report
    create_summary_report(results, output_dir)

def create_summary_report(results: Dict[str, Any], output_dir: Path):
    """Creates a human-readable text file summarizing the test results.

    Detailed Description:
        - This function generates a `.txt` file that provides a clean, text-based summary
          of the most important findings from the performance test.
        - It includes sections for overall performance metrics, prediction distribution,
          and timing statistics, making the results easy to read and share.

    Parameters:
        - results (Dict[str, Any]): The dictionary of test results.
        - output_dir (Path): The directory where the report file will be saved.
    """
    
    report_path = output_dir / "performance_test_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RECIPE CLASSIFICATION PERFORMANCE TEST REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Test Date: {results['test_parameters']['test_date']}\n")
        f.write(f"Samples Tested: {results['test_parameters']['actual_samples']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*30 + "\n")
        perf = results["performance_metrics"]
        f.write(f"Total Execution Time: {perf['total_execution_time_seconds']:.2f} seconds\n")
        f.write(f"Average Time per Recipe: {perf['average_time_per_recipe_seconds']*1000:.2f} ms\n")
        f.write(f"Throughput: {perf['throughput_recipes_per_second']:.1f} recipes/second\n")
        f.write(f"Success Rate: {perf['success_rate']:.1f}%\n")
        f.write(f"Errors: {perf['errors']}\n\n")
        
        f.write("PREDICTION DISTRIBUTION:\n")
        f.write("-"*30 + "\n")
        dist = results["prediction_distribution"]
        f.write(f"Vegan Recipes: {dist['total_vegan']} ({dist['vegan_percentage']:.1f}%)\n")
        f.write(f"Keto Recipes: {dist['total_keto']} ({dist['keto_percentage']:.1f}%)\n")
        f.write(f"Both Vegan & Keto: {dist['both_vegan_and_keto']} ({dist['both_percentage']:.1f}%)\n\n")
        
        f.write("TIMING STATISTICS:\n")
        f.write("-"*30 + "\n")
        timing = results["timing_statistics"]
        f.write(f"Min Recipe Time: {timing['min_recipe_time']*1000:.2f} ms\n")
        f.write(f"Max Recipe Time: {timing['max_recipe_time']*1000:.2f} ms\n")
        f.write(f"Median Recipe Time: {timing['median_recipe_time']*1000:.2f} ms\n")
        f.write(f"Standard Deviation: {timing['std_recipe_time']*1000:.2f} ms\n")
        
    logger.info(f"📝 Saved summary report to {report_path}")

# Main execution function for easy calling
def run_performance_test_suite(n_samples: int = 5000) -> Dict[str, Any]:
    """The main entry-point function to run the complete performance test suite.

    Detailed Description:
        - This function serves as the primary wrapper for executing the performance test.
        - It calls `run_performance_test` and includes top-level error handling (a try...except block)
          to catch any exceptions that might occur during the test run, ensuring that the script
          exits gracefully.

    Parameters:
        - n_samples (int): The number of random samples to test.

    Returns:
        - Dict[str, Any]: The complete performance test results.

    Raises:
        - Exception: Catches and re-raises any exception that occurs during the test.
    """
    logger.info("🎯 Starting Recipe Classification Performance Test Suite")
    
    try:
        results = run_performance_test(n_samples=n_samples, save_results=True)
        logger.success("✅ Performance test completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        raise

if __name__ == "__main__":
    # Run performance test with default parameters
    results = run_performance_test_suite(5000) 