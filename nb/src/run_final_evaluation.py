"""Final Evaluation Script.

This script evaluates the performance of the Qwen-based, context-aware dietary
classification system against the Gemini-generated persona ground truth.
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from argparse import ArgumentParser
import pandas as pd
from loguru import logger
import os
import ast
import asyncio
import json

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

try:
    from context_aware_classifier import ContextAwareDietClassifier
    from config import app_config
except ImportError as e:
    logger.critical(f"Could not import required modules for evaluation: {e}")
    sys.exit(1)

def load_ground_truth_data(ground_truth_path: Path) -> pd.DataFrame:
    """Loads and prepares the Gemini-generated ground truth data."""
    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found at: {ground_truth_path}.")
        logger.info("Please run `nb/src/ground_truth/generate.py` first to create the ground truth.")
        sys.exit(1)
    
    df = pd.read_csv(ground_truth_path)
    logger.info(f"Loaded {len(df)} records from {ground_truth_path.name}")

    # Ensure 'ingredients' column is a list of strings
    def safe_eval_ingredients(val):
        if isinstance(val, str) and val.startswith('['):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return val if isinstance(val, list) else []
        
    df['ingredients_list'] = df['ingredients'].apply(safe_eval_ingredients)
    
    return df

async def run_classification_for_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the Qwen-based context-aware classification for evaluation."""
    classifier = ContextAwareDietClassifier()
    predictions = []
    
    logger.info("Starting Qwen-based context-aware classification for evaluation...")
    
    for index, row in df.iterrows():
        recipe_id = row['recipe_id']
        ingredients = row['ingredients_list']
        title = row['title']
        
        try:
            # Get predictions from our Qwen-based classifier
            llm_pred = await classifier.classify_with_context(ingredients, title)
            
            predictions.append({
                'recipe_id': recipe_id,
                'title': title,
                'ingredients': row['ingredients'], # Keep raw for context
                'true_is_keto': row['is_keto'],
                'true_is_vegan': row['is_vegan'],
                'pred_is_keto': llm_pred.get('is_keto', False),
                'pred_is_vegan': llm_pred.get('is_vegan', False),
                'reasoning': llm_pred.get('reasoning', 'No reasoning provided.')
            })
        except Exception as e:
            logger.error(f"Error classifying recipe {recipe_id}: {e}. Defaulting to False predictions.")
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
            
    return pd.DataFrame(predictions)

def calculate_and_report_metrics(eval_df: pd.DataFrame):
    """Calculates and prints classification reports for Keto and Vegan."""
    print("\n" + "="*20 + " Keto Classification Evaluation (Qwen vs Gemini) " + "="*20)
    print(classification_report(
        eval_df['true_is_keto'], eval_df['pred_is_keto'], 
        target_names=['Not Keto', 'Keto'], zero_division=0
    ))
    
    print("\n" + "="*20 + " Vegan Classification Evaluation (Qwen vs Gemini) " + "="*20)
    print(classification_report(
        eval_df['true_is_vegan'], eval_df['pred_is_vegan'], 
        target_names=['Not Vegan', 'Vegan'], zero_division=0
    ))

    # Save detailed predictions for further analysis
    output_dir = Path("nb/src/data/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.success(f"Detailed predictions saved to {output_dir / 'predictions.csv'}")

def main():
    """Main execution block for the final evaluation pipeline."""
    logger.info("🚀 Starting Final Evaluation of Qwen Classifier against Gemini Ground Truth...")
    
    ground_truth_path = Path(app_config.PROJECT_ROOT) / "data" / "personas_ground_truth.csv"
    
    # 1. Load Gemini-generated ground truth
    ground_truth_df = load_ground_truth_data(ground_truth_path)
    if ground_truth_df.empty:
        logger.error("No ground truth data available for evaluation. Exiting.")
        return -1

    # 2. Run Qwen-based context-aware classification
    start_time = time.time()
    evaluation_predictions_df = asyncio.run(run_classification_for_evaluation(ground_truth_df))
    end_time = time.time()

    logger.info("--- Evaluation Results ---")
    calculate_and_report_metrics(evaluation_predictions_df)
    
    total_time = end_time - start_time
    avg_time = total_time / len(evaluation_predictions_df) if len(evaluation_predictions_df) > 0 else 0
    logger.success(f"Evaluation finished in {total_time:.2f}s ({avg_time * 1000:.2f} ms/recipe).")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 