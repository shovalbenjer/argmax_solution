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
    """Mocks a 'teacher' LLM call to generate a classification persona.

    Detailed Description:
        - This function simulates a call to a powerful 'teacher' language model (like Gemini 1.5 Pro)
          to generate a ground truth classification for a recipe.
        - It uses a simplified heuristic: it calculates the total carbohydrates from the provided contexts
          and checks for any non-vegan ingredients to assign a dietary persona (e.g., 'strict_keto', 'not_vegan').
        - This mock is used for generating a sample ground truth dataset without incurring API costs.

    Parameters:
        - contexts (List[Dict]): A list of dictionaries, where each dictionary contains the
          processed context for a single ingredient (e.g., nutritional information).

    Returns:
        - Dict: A dictionary containing the classification persona, the reasoning behind it,
          and boolean flags for 'keto' and 'vegan'.

    Examples:
        >>> get_teacher_classification([{'normalized_nutrients': {'total_carbs_g': 5}}, {'normalized_nutrients': {'total_carbs_g': 2}}])
        {'classification': 'strict_keto', 'reasoning': 'Mocked classification based on total carbs (7.0g) and vegan checks.', 'keto': True, 'vegan': False}
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
    """Orchestrates the ground truth generation process using a mocked teacher LLM.

    Detailed Description:
        - This function manages the end-to-end process of creating a ground truth dataset.
        - It loads a sample of recipes from a raw CSV file.
        - It initializes an MLflow experiment to track the generation process, logging parameters
          like the model name and sample size.
        - It iterates through each recipe, processes its ingredients to get their nutritional contexts,
          and uses the mocked `get_teacher_classification` function to generate a classification.
        - Finally, it saves the results to a CSV file and logs the output file and performance
          metrics as artifacts in MLflow.

    Libraries Used:
        - pandas: For reading the raw recipe data and creating the final results DataFrame. Its performance
          and ease of use with tabular data make it a better choice than standard Python lists of dictionaries.
        - mlflow: To log the experiment, including parameters, metrics, and the output artifact. This ensures
          reproducibility and provides a clear record of the generation process.
        - loguru: For structured and colorful logging, which improves the readability of the console output
          compared to the standard `logging` module.
        - pathlib: For robust and cross-platform path management, which is safer and more readable than
          using string concatenation for file paths.
    """
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