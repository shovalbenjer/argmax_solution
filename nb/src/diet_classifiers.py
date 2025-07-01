# --- File: diet_classifiers.py (Refined for Production) ---

"""
Fully Integrated, Production-Grade Dietary Classifier.

This single, robust script contains the complete, end-to-end system for
dietary classification, designed for reliability and performance. It integrates:
1.  An Elasticsearch database handler.
2.  The Hybrid Retrieval Cascade and Normalization Engine.
3.  A `transformers`-based LLM Handler for the Qwen-0.6B model.
"""
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Any
import os
import sys
from argparse import ArgumentParser
import ast

import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from ingredient_parser import parse_ingredient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    from sklearn.metrics import classification_report
except ImportError:
    logger.warning("scikit-learn not installed. Evaluation metrics will be limited.")
    classification_report = lambda y_true, y_pred, target_names: "scikit-learn not available."

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from shared.diet_classifiers import is_keto, is_vegan, diet_classifier
    from shared.database import db_manager
    from shared.config import app_config
    logger.info("Successfully imported shared modules")
except ImportError as e:
    logger.critical(f"Could not import shared modules: {e}")
    sys.exit(1)

# Use centralized configuration
KETO_CARBS_THRESHOLD = app_config.KETO_CARBS_THRESHOLD

# Use shared database manager instead of duplicated code
# DatabaseHandler is replaced by shared.database.db_manager

# Use shared LLM client instead of mock implementation
from shared.llm_client import llm_client

# Use shared instances instead of local ones

# Use shared ingredient processor instead of duplicated code
def get_ingredient_context(raw_ingredient: str) -> Dict:
    """Legacy wrapper for shared ingredient processor."""
    return diet_classifier.processor.get_ingredient_context(raw_ingredient)

@lru_cache(maxsize=512)
def classify_ingredients(ingredients: tuple) -> Dict:
    """Legacy wrapper for shared LLM classification."""
    import asyncio
    try:
        # Try to run async classification
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in async context, create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, diet_classifier.classify_with_llm(list(ingredients)))
                return future.result()
        else:
            return asyncio.run(diet_classifier.classify_with_llm(list(ingredients)))
    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        # Fallback to rule-based
        return {
            "is_keto": diet_classifier.is_keto(list(ingredients)),
            "is_vegan": diet_classifier.is_vegan(list(ingredients))
        }

# These functions are now imported from shared.diet_classifiers
# Keeping here for any legacy code that might import from this module

# --- Final Evaluation ---

def run_evaluation():
    """Runs the final evaluation against the ground truth dataset."""
    logger.info("===== Running Final Evaluation =====")
    gt_path = Path(__file__).resolve().parent / "data" / "ground_truth_sample.csv"
    if not gt_path.exists():
        logger.error(f"Ground truth file not found at {gt_path}. Run generate_ground_truth.py first.")
        return

    ground_truth = pd.read_csv(gt_path)
    
    # Safely evaluate string-formatted lists
    ground_truth['ingredients'] = ground_truth['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])

    start_time = time.time()
    ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
    ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)
    end_time = time.time()

    logger.info("--- Keto Classification Report ---")
    print(classification_report(ground_truth['keto'], ground_truth['keto_pred']))
    
    logger.info("--- Vegan Classification Report ---")
    print(classification_report(ground_truth['vegan'], ground_truth['vegan_pred']))

    total_time = end_time - start_time
    avg_time = total_time / len(ground_truth) if len(ground_truth) > 0 else 0
    logger.success(f"Evaluation finished in {total_time:.2f}s ({avg_time:.3f}s/recipe).")

# Use shared classifier methods
def is_ingredient_keto(ingredient: str) -> bool:
    """Checks if a single ingredient is keto-friendly."""
    return diet_classifier.is_ingredient_keto(ingredient)

def is_ingredient_vegan(ingredient: str) -> bool:
    """Checks if a single ingredient is vegan-friendly."""
    return diet_classifier.is_ingredient_vegan(ingredient)

def main(args):
    """Runs the evaluation against the provided ground truth file."""
    logger.info("--- Starting Evaluation ---")
    try:
        ground_truth = pd.read_csv(args.ground_truth, index_col=None)
        ground_truth['ingredients'] = ground_truth['ingredients'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    except FileNotFoundError:
        logger.error(f"Ground truth file not found at: {args.ground_truth}")
        return -1

    start_time = time.time()
    ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
    ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)
    end_time = time.time()

    print("\n" + "="*20 + " Keto Classification " + "="*20)
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred'], target_names=['Not Keto', 'Keto'], zero_division=0
    ))
    
    print("\n" + "="*20 + " Vegan Classification " + "="*20)
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred'], target_names=['Not Vegan', 'Vegan'], zero_division=0
    ))
    
    logger.success(f"== Evaluation finished in {end_time - start_time:.2f} seconds ==")
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str, default="data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))