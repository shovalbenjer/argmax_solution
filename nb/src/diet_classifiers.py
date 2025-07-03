# --- File: diet_classifiers.py (Refined for Production) ---

"""
Diet Classifiers - Submission Implementation

Implements the complete pipeline: Input Parsing → Arctic → Knowledge DB → Qwen
This matches the original submission format while using our sophisticated backend.

Pipeline: String/List Input → Ingredient Parser → Arctic Text2SQL → Knowledge DB → Qwen → Classification
"""
import json
import sys
import asyncio
from argparse import ArgumentParser
from typing import List, Union
from time import time
import pandas as pd
from loguru import logger

try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")

try:
    from ingredient_parser import parse_ingredient
except ImportError:
    logger.warning("ingredient-parser-nlp not available, using fallback parsing")
    parse_ingredient = None

# Local imports (no shared dependency)
from context_aware_classifier import ContextAwareDietClassifier
from config import app_config
from ingredient_processor.processor import get_context_with_rapidfuzz_fallback

# Initialize the RAG pipeline
context_classifier = ContextAwareDietClassifier()

def parse_ingredient_name(raw_ingredient: str) -> str:
    """
    Extract clean ingredient name from raw ingredient string.
    
    Uses ingredient-parser-nlp (97.8% accuracy) with fallback to simple parsing.
    Examples: "3 pounds pork shoulder, cut into chunks" → "pork shoulder"
    """
    if parse_ingredient:
        try:
            parsed = parse_ingredient(raw_ingredient, discard_isolated_stop_words=True)
            if parsed.name and len(parsed.name) > 0:
                name = parsed.name[0].text
                logger.debug(f"Parsed '{raw_ingredient}' → '{name}' (confidence: {parsed.name[0].confidence:.3f})")
                return name.lower().strip()
        except Exception as e:
            logger.warning(f"Ingredient parser failed for '{raw_ingredient}': {e}")
    
    # Fallback: simple cleaning
    # Remove quantities, units, and preparation instructions
    clean_name = raw_ingredient.lower().strip()
    # Remove common quantity patterns
    import re
    clean_name = re.sub(r'^\d+[\s\w]*\s+', '', clean_name)  # Remove leading numbers/units
    clean_name = re.sub(r',.*$', '', clean_name)  # Remove everything after comma
    clean_name = clean_name.strip()
    logger.debug(f"Fallback parsed '{raw_ingredient}' → '{clean_name}'")
    return clean_name

def is_ingredient_keto(ingredient: str) -> bool:
    """
    Complete pipeline: Ingredient Parsing → Arctic → Knowledge DB → Qwen
    
    Pipeline steps:
    1. Parse ingredient name from raw string (handles "1 cup flour" → "flour")
    2. Arctic Text2SQL generates SQL query for parsed ingredient
    3. Query executes against knowledge_graph.db 
    4. Retrieved context passed to Qwen3-0.6B for final judgment
    """
    try:
        # Step 1: Parse ingredient name
        clean_ingredient = parse_ingredient_name(ingredient)
        
        # Step 2-4: Arctic → Knowledge DB → Qwen pipeline
        result = asyncio.run(context_classifier.classify_single_ingredient(clean_ingredient))
        is_keto_result = result.get('is_keto', False)
        
        logger.debug(f"Keto classification: '{ingredient}' → '{clean_ingredient}' → {is_keto_result}")
        return is_keto_result
    except Exception as e:
        logger.error(f"Keto classification failed for '{ingredient}': {e}")
        return False

def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Complete pipeline: Ingredient Parsing → Arctic → Knowledge DB → Qwen
    
    Pipeline steps:
    1. Parse ingredient name from raw string (handles "2 tbsp butter" → "butter")
    2. Arctic Text2SQL generates SQL query for parsed ingredient
    3. Query executes against knowledge_graph.db
    4. Retrieved context passed to Qwen3-0.6B for final judgment
    """
    try:
        # Step 1: Parse ingredient name
        clean_ingredient = parse_ingredient_name(ingredient)
        
        # Step 2-4: Arctic → Knowledge DB → Qwen pipeline
        result = asyncio.run(context_classifier.classify_single_ingredient(clean_ingredient))
        is_vegan_result = result.get('is_vegan', False)
        
        logger.debug(f"Vegan classification: '{ingredient}' → '{clean_ingredient}' → {is_vegan_result}")
        return is_vegan_result
    except Exception as e:
        logger.error(f"Vegan classification failed for '{ingredient}': {e}")
        return False

def parse_ingredients_input(ingredients: Union[str, List[str]]) -> List[str]:
    """
    Handle both string and list inputs to match original submission format.
    
    Expected inputs:
    - String: '["chicken breast", "spinach", "olive oil"]' (JSON format)
    - String: "chicken breast, spinach, olive oil" (comma-separated)
    - List: ["chicken breast", "spinach", "olive oil"] (direct list)
    """
    if isinstance(ingredients, str):
        try:
            # Try parsing as JSON first
            ingredients_list = json.loads(ingredients)
            logger.debug(f"Parsed JSON input: {ingredients_list}")
        except json.JSONDecodeError:
            # Fallback: split by comma
            ingredients_list = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
            logger.debug(f"Parsed comma-separated input: {ingredients_list}")
    else:
        ingredients_list = ingredients
        logger.debug(f"Using direct list input: {ingredients_list}")
    
    return ingredients_list

def is_keto(ingredients: Union[str, List[str]]) -> bool:
    """
    Check if recipe is keto-friendly (all ingredients must be keto).
    
    Handles both string and list inputs to match original submission format.
    """
    try:
        ingredients_list = parse_ingredients_input(ingredients)
        result = all(map(is_ingredient_keto, ingredients_list))
        logger.info(f"Keto recipe classification: {len(ingredients_list)} ingredients → {result}")
        return result
    except Exception as e:
        logger.error(f"Keto recipe classification failed: {e}")
        return False

def is_vegan(ingredients: Union[str, List[str]]) -> bool:
    """
    Check if recipe is vegan-friendly (all ingredients must be vegan).
    
    Handles both string and list inputs to match original submission format.
    """
    try:
        ingredients_list = parse_ingredients_input(ingredients)
        result = all(map(is_ingredient_vegan, ingredients_list))
        logger.info(f"Vegan recipe classification: {len(ingredients_list)} ingredients → {result}")
        return result
    except Exception as e:
        logger.error(f"Vegan recipe classification failed: {e}")
        return False

def main(args):
    """Main function for evaluating diet classifiers against ground truth."""
    logger.info("Starting diet classifier evaluation")
    
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    logger.info(f"Loaded {len(ground_truth)} ground truth records")
    
    try:
        start_time = time()
        
        # Apply keto classification
        logger.info("Running keto classification")
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        
        # Apply vegan classification  
        logger.info("Running vegan classification")
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)

        end_time = time()
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    
    logger.success("Diet classifier evaluation completed")
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))