# --- File: generate_ground_truth.py (Refactored for Production) ---

"""
Ground Truth Generation Pipeline with MLflow.

This script uses a 'teacher' LLM (mocked for this implementation) to generate
a high-quality, nuanced ground truth dataset. The process is tracked using
MLflow for full reproducibility.
"""

import json
import time
from pathlib import Path
from typing import List, Dict

import mlflow
import pandas as pd
from loguru import logger

try:
    from nb.src.ingredient_processor import get_ingredient_context
except ImportError:
    logger.critical("Failed to import get_ingredient_context. Ensure ingredient_processor.py exists.")
    def get_ingredient_context(ing): return {"error": "module not found"}

# --- Configuration ---
RAW_DATA_PATH = Path(__file__).resolve().parent / "raw_data" / "recipes.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "ground_truth_sample.csv"
MLFLOW_EXPERIMENT_NAME = "Ground Truth Generation"
TEACHER_MODEL = "Mocked Gemini 1.5 Pro"
SAMPLE_SIZE = 500 # Use a smaller sample for faster execution

# --- Mock Teacher LLM ---
def get_teacher_classification(contexts: List[Dict]) -> Dict:
    """
    Mocks a call to a powerful 'teacher' LLM like Gemini.
    Generates a classification persona based on the processed contexts.
    """
    total_carbs = sum(c.get('normalized_nutrients', {}).get('total_carbs_g', 100) or 100 for c in contexts)
    is_clearly_not_vegan = any("non_vegan" in c.get('retrieved_data', {}).get('name', '') for c in contexts)
    
    persona = "strict_vegan"
    if total_carbs > 50:
        persona = "not_keto"
    elif 20 < total_carbs <= 50:
        persona = "borderline_keto"
    else:
        persona = "strict_keto"
        
    if is_clearly_not_vegan:
        if persona == "strict_vegan": persona = "not_vegan" # Should not happen, but for safety
    
    return {
        "classification": persona,
        "reasoning": f"Mocked classification based on total carbs ({total_carbs:.1f}g) and vegan checks.",
        "keto": "keto" in persona,
        "vegan": "vegan" in persona
    }

# --- Main Execution ---
def main():
    """Orchestrates the ground truth generation process."""
    logger.info(f"===== Starting Ground Truth Generation (Sample Size: {SAMPLE_SIZE}) =====")
    
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}. Aborting.")
        return

    df = pd.read_csv(RAW_DATA_PATH).sample(n=SAMPLE_SIZE, random_state=42)
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run Started: {run.info.run_name}")
        mlflow.log_params({"teacher_model": TEACHER_MODEL, "sample_size": SAMPLE_SIZE})

        results = []
        start_time = time.time()
        
        for index, row in df.iterrows():
            ingredients = eval(row['ingredients']) if isinstance(row['ingredients'], str) else []
            if not ingredients:
                continue
            
            contexts = [get_ingredient_context(ing) for ing in ingredients]
            classification = get_teacher_classification(contexts)
            
            results.append({
                "recipe_id": row.get('id', index),
                "ingredients": str(ingredients),
                **classification
            })

        end_time = time.time()
        
        # Save and log results
        results_df = pd.DataFrame(results)
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        results_df.to_csv(OUTPUT_PATH, index=False)
        mlflow.log_artifact(str(OUTPUT_PATH))
        
        # Log metrics
        total_time = end_time - start_time
        mlflow.log_metrics({
            "recipes_processed": len(results_df),
            "total_time_seconds": total_time,
            "avg_time_per_recipe": total_time / len(results_df) if results_df.shape[0] > 0 else 0
        })

        logger.success(f"Ground truth generation complete. Results saved to {OUTPUT_PATH}")
        logger.info(f"Processed {len(results_df)} recipes in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()