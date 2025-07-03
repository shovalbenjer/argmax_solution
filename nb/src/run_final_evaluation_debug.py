"""
Debug version of the final evaluation script with enhanced error handling.
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import polars as pl
from loguru import logger
import os
import ast
import asyncio
import json

# Import nest_asyncio for Jupyter compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    logger.warning("nest_asyncio not available - may cause issues in Jupyter environments")

from sklearn.metrics import classification_report

try:
    from context_aware_classifier import ContextAwareDietClassifier
    from config import app_config
    from llm_client import llm_client
except ImportError as e:
    logger.critical(f"Could not import required modules for evaluation: {e}")
    sys.exit(1)

def load_gold_label_personas(eval_data_dir: Path) -> pl.DataFrame:
    """Loads and concatenates all gold label persona tables for evaluation."""
    persona_files = [
        eval_data_dir / "borderline_keto.csv",
        eval_data_dir / "strict_keto.csv",
        eval_data_dir / "strict_vegan.csv",
        eval_data_dir / "ground_truth_sample.csv"
    ]
    
    final_expected_schema = {
        'recipe_id': pl.Utf8,
        'persona': pl.Utf8,
        'reasoning': pl.Utf8,
        'is_vegan': pl.Boolean,
        'is_keto': pl.Boolean,
        'confidence': pl.Float64, 
        'title': pl.Utf8,
        'ingredients': pl.Utf8,
        'description': pl.Utf8,
        'instructions': pl.Utf8,
        'photo_url': pl.Utf8,
    }

    dfs = []
    for f in persona_files:
        if not f.exists():
            logger.error(f"Gold label persona file not found: {f}")
            continue
        try:
            df = pl.read_csv(f)
            
            # Rename columns if necessary
            if "vegan" in df.columns:
                df = df.rename({"vegan": "is_vegan"})
            if "keto" in df.columns:
                df = df.rename({"keto": "is_keto"})
            
            # Add missing columns with correct types
            for col, dtype in final_expected_schema.items():
                if col not in df.columns:
                    if dtype == pl.Utf8:
                        df = df.with_columns(pl.lit("").cast(dtype).alias(col))
                    else:
                        df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
            
            df = df.select(list(final_expected_schema.keys()))
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Failed to load and standardize {f}: {e}")
    
    if not dfs:
        logger.error("No gold label persona tables could be loaded. Exiting.")
        sys.exit(1)
    
    df_all = pl.concat(dfs)
    logger.info(f"Loaded {len(df_all)} total records from gold label persona tables.")
    
    def safe_eval_ingredients(val):
        if isinstance(val, str) and val.startswith('['):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return val if isinstance(val, list) else []
    
    df_all = df_all.with_columns(
        pl.col("ingredients").map_elements(safe_eval_ingredients, return_dtype=pl.List(pl.Utf8)).alias("ingredients_list")
    )
    return df_all

async def run_classification_for_evaluation(df: pl.DataFrame) -> pl.DataFrame:
    """Runs the Qwen-based context-aware classification for evaluation."""
    logger.info("Initializing ContextAwareDietClassifier...")
    
    try:
        classifier = ContextAwareDietClassifier()
        logger.info("✅ ContextAwareDietClassifier initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize classifier: {e}")
        raise
    
    predictions = []
    df_pandas = df.to_pandas()
    
    logger.info(f"Starting classification of {len(df_pandas)} recipes...")
    
    for index, row in df_pandas.iterrows():
        recipe_id = row['recipe_id']
        ingredients = row['ingredients_list']
        title = row['title']
        
        logger.info(f"Processing recipe {index + 1}/{len(df_pandas)}: {recipe_id}")
        
        # Check ingredients
        if ingredients is None:
            has_ingredients = False
        elif hasattr(ingredients, '__len__'):
            has_ingredients = len(ingredients) > 0
        else:
            has_ingredients = bool(ingredients)
        
        if not has_ingredients:
            logger.warning(f"Skipping recipe {recipe_id} - no ingredients found")
            predictions.append({
                'recipe_id': recipe_id,
                'title': title,
                'ingredients': row['ingredients'],
                'true_is_keto': row['is_keto'],
                'true_is_vegan': row['is_vegan'],
                'pred_is_keto': False,
                'pred_is_vegan': False,
                'reasoning': "No ingredients provided for classification"
            })
            continue
        
        try:
            logger.info(f"Classifying recipe {recipe_id} with {len(ingredients)} ingredients")
            
            # Use timeout protection
            result = await asyncio.wait_for(
                classifier.classify_recipe(ingredients),
                timeout=30.0  # Reduced timeout for debugging
            )
            
            predictions.append({
                'recipe_id': recipe_id,
                'title': title,
                'ingredients': row['ingredients'],
                'true_is_keto': row['is_keto'],
                'true_is_vegan': row['is_vegan'],
                'pred_is_keto': result.get('recipe_is_keto', False),
                'pred_is_vegan': result.get('recipe_is_vegan', False),
                'reasoning': result.get('reasoning', 'No reasoning provided.')
            })
            
            logger.info(f"✅ Successfully classified recipe {recipe_id}")
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout classifying recipe {recipe_id} - took longer than 30 seconds")
            predictions.append({
                'recipe_id': recipe_id,
                'title': title,
                'ingredients': row['ingredients'],
                'true_is_keto': row['is_keto'],
                'true_is_vegan': row['is_vegan'],
                'pred_is_keto': False,
                'pred_is_vegan': False,
                'reasoning': "Classification timed out after 30 seconds"
            })
        except Exception as e:
            logger.error(f"❌ Error classifying recipe {recipe_id}: {e}")
            predictions.append({
                'recipe_id': recipe_id,
                'title': title,
                'ingredients': row['ingredients'],
                'true_is_keto': row['is_keto'],
                'true_is_vegan': row['is_vegan'],
                'pred_is_keto': False,
                'pred_is_vegan': False,
                'reasoning': f"Classification failed due to error: {e}"
            })
            
    return pl.DataFrame(predictions)

def calculate_and_report_metrics(eval_df: pl.DataFrame):
    """Calculates and prints comprehensive classification metrics for evaluation."""
    eval_pandas = eval_df.to_pandas()
    
    print("\n" + "="*20 + " Keto Classification Evaluation (Qwen vs Gemini) " + "="*20)
    print(classification_report(
        eval_pandas['true_is_keto'], eval_pandas['pred_is_keto'], 
        target_names=['Not Keto', 'Keto'], zero_division=0
    ))
    
    print("\n" + "="*20 + " Vegan Classification Evaluation (Qwen vs Gemini) " + "="*20)
    print(classification_report(
        eval_pandas['true_is_vegan'], eval_pandas['pred_is_vegan'], 
        target_names=['Not Vegan', 'Vegan'], zero_division=0
    ))

    # Save detailed predictions for further analysis
    output_dir = Path("nb/src/data/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_df.write_csv(output_dir / "predictions_debug.csv")
    logger.success(f"Detailed predictions saved to {output_dir / 'predictions_debug.csv'}")

def main():
    """Main execution function for the debug evaluation pipeline."""
    logger.info("🚀 Starting Debug Evaluation of Qwen Classifier against Gold Label Personas...")
    
    try:
        # Check system health first
        logger.info("Checking system health...")
        
        # Check Ollama
        models = llm_client.list_models()
        if not models:
            logger.error("❌ No models found in Ollama. Please ensure Ollama is running.")
            return -1
        
        logger.info(f"✅ Found {len(models)} models in Ollama")
        
        # Load data
        eval_data_dir = Path(app_config.PROJECT_ROOT) / "eval_data"
        ground_truth_df = load_gold_label_personas(eval_data_dir)
        if ground_truth_df.is_empty():
            logger.error("No ground truth data available for evaluation. Exiting.")
            return -1

        # Run classification
        start_time = time.time()
        try:
            evaluation_predictions_df = asyncio.run(run_classification_for_evaluation(ground_truth_df))
        except RuntimeError as e:
            if "asyncio.run()" in str(e) and "running event loop" in str(e):
                loop = asyncio.get_event_loop()
                evaluation_predictions_df = loop.run_until_complete(run_classification_for_evaluation(ground_truth_df))
            else:
                raise
        end_time = time.time()

        logger.info("--- Evaluation Results ---")
        calculate_and_report_metrics(evaluation_predictions_df)
        
        total_time = end_time - start_time
        avg_time = total_time / len(evaluation_predictions_df) if len(evaluation_predictions_df) > 0 else 0
        logger.success(f"Evaluation finished in {total_time:.2f}s ({avg_time * 1000:.2f} ms/recipe).")
        
        return 0
        
    except Exception as e:
        logger.error(f"Critical error during evaluation: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return -1

if __name__ == "__main__":
    sys.exit(main()) 