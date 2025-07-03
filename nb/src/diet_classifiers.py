"""
Diet Classifiers - Production-Ready Implementation

This module implements the complete diet classification pipeline that processes
ingredient inputs and determines whether recipes are keto-friendly or vegan.
The system uses a sophisticated multi-stage approach combining ingredient parsing,
database queries, and LLM-based classification.

Pipeline Architecture:
1. Input Parsing: Handles various input formats (JSON, comma-separated, lists)
2. Ingredient Extraction: Uses NLP parsing to extract clean ingredient names
3. Arctic Text2SQL: Generates database queries for nutritional context
4. Knowledge Database: Retrieves factual nutritional and vegan ontology data
5. Qwen Classification: LLM-based final classification with context awareness

Key Features:
- Multi-format input support (JSON, CSV, direct lists)
- High-accuracy ingredient parsing (97.8% accuracy with fallback)
- Context-aware classification using retrieved knowledge
- Async-safe operations with dedicated event loop management
- Comprehensive error handling and logging
- Production-ready performance optimization

Example:
    >>> from diet_classifiers import is_keto, is_vegan
    >>> ingredients = ["chicken breast", "spinach", "olive oil"]
    >>> keto_result = is_keto(ingredients)
    >>> vegan_result = is_vegan(ingredients)
    >>> print(f"Keto: {keto_result}, Vegan: {vegan_result}")
"""
import json
import sys
import asyncio
from argparse import ArgumentParser
from typing import List, Union
import time
import polars as pd
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor, Future

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

class AsyncRunner:
    """
    Manages a dedicated thread with persistent event loop for async operations.
    
    This class provides a thread-safe way to execute async operations from
    synchronous contexts. It maintains a dedicated thread with its own event
    loop, allowing the diet classification pipeline to work seamlessly in
    both sync and async environments.
    
    The AsyncRunner is essential for the diet classification system because
    the underlying LLM operations are asynchronous, but the public API
    needs to remain synchronous for compatibility with existing code.
    
    Attributes:
        loop: Dedicated asyncio event loop
        thread: Background thread running the event loop
        
    Example:
        >>> runner = AsyncRunner()
        >>> result = runner.run_coroutine(some_async_function())
        >>> runner.close()  # Clean up resources
    """
    
    def __init__(self):
        """
        Initialize the async runner with a dedicated event loop thread.
        
        Creates a background thread that runs its own asyncio event loop.
        This allows async operations to be executed from synchronous contexts
        without blocking the main thread.
        """
        self.loop = None
        self.thread = None
        self._start_loop()
    
    def _start_loop(self):
        """
        Start the event loop in a dedicated thread.
        
        Creates a new thread that runs an asyncio event loop continuously.
        The method waits for the loop to be ready before returning to ensure
        the runner is immediately usable.
        """
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
    
    def run_coroutine(self, coro):
        """
        Run a coroutine in the dedicated thread's event loop.
        
        Executes an async coroutine in the background event loop and returns
        the result. This method provides a bridge between sync and async code.
        
        Args:
            coro: Async coroutine to execute
            
        Returns:
            The result of the coroutine execution
            
        Raises:
            TimeoutError: If the coroutine takes longer than 5 minutes
            Exception: Any exception raised by the coroutine
            
        Example:
            >>> async def my_async_func():
            ...     return "Hello from async"
            >>> result = runner.run_coroutine(my_async_func())
            >>> print(result)
            'Hello from async'
        """
        if self.loop is None or self.loop.is_closed():
            self._start_loop()
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=300)  # 5 minute timeout
    
    def close(self):
        """
        Close the event loop and stop the thread.
        
        Properly shuts down the background thread and event loop to prevent
        resource leaks. This method should be called when the AsyncRunner
        is no longer needed.
        """
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)

# Global async runner instance
_async_runner = AsyncRunner()

def parse_ingredient_name(raw_ingredient: str) -> str:
    """
    Extract clean ingredient name from raw ingredient string.
    
    Uses the ingredient-parser-nlp library (97.8% accuracy) with fallback
    to simple regex-based parsing. Handles common ingredient formats including
    quantities, units, and preparation instructions.
    
    Args:
        raw_ingredient: Raw ingredient string (e.g., "3 pounds pork shoulder, cut into chunks")
        
    Returns:
        Clean ingredient name (e.g., "pork shoulder")
        
    Example:
        >>> parse_ingredient_name("2 tbsp olive oil")
        'olive oil'
        >>> parse_ingredient_name("1 cup all-purpose flour, sifted")
        'all-purpose flour'
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
    
    Determines if a single ingredient is keto-friendly by following the complete
    classification pipeline. This function handles the full process from raw
    ingredient string to final classification.
    
    Pipeline steps:
    1. Parse ingredient name from raw string (handles "1 cup flour" → "flour")
    2. Arctic Text2SQL generates SQL query for parsed ingredient
    3. Query executes against knowledge_graph.db 
    4. Retrieved context passed to Qwen3-0.6B for final judgment
    
    Args:
        ingredient: Raw ingredient string to classify
        
    Returns:
        bool: True if ingredient is keto-friendly, False otherwise
        
    Example:
        >>> is_ingredient_keto("chicken breast")
        True
        >>> is_ingredient_keto("white rice")
        False
    """
    try:
        # Step 1: Parse ingredient name
        clean_ingredient = parse_ingredient_name(ingredient)
        
        # Step 2-4: Arctic → Knowledge DB → Qwen pipeline (event loop safe)
        result = _async_runner.run_coroutine(context_classifier.classify_single_ingredient(clean_ingredient))
        is_keto_result = result.get('is_keto', False)
        
        logger.debug(f"Keto classification: '{ingredient}' → '{clean_ingredient}' → {is_keto_result}")
        return is_keto_result
    except Exception as e:
        logger.error(f"Keto classification failed for '{ingredient}': {e}")
        return False

def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Complete pipeline: Ingredient Parsing → Arctic → Knowledge DB → Qwen
    
    Determines if a single ingredient is vegan-friendly by following the complete
    classification pipeline. This function handles the full process from raw
    ingredient string to final classification.
    
    Pipeline steps:
    1. Parse ingredient name from raw string (handles "2 tbsp butter" → "butter")
    2. Arctic Text2SQL generates SQL query for parsed ingredient
    3. Query executes against knowledge_graph.db
    4. Retrieved context passed to Qwen3-0.6B for final judgment
    
    Args:
        ingredient: Raw ingredient string to classify
        
    Returns:
        bool: True if ingredient is vegan-friendly, False otherwise
        
    Example:
        >>> is_ingredient_vegan("spinach")
        True
        >>> is_ingredient_vegan("milk")
        False
    """
    try:
        # Step 1: Parse ingredient name
        clean_ingredient = parse_ingredient_name(ingredient)
        
        # Step 2-4: Arctic → Knowledge DB → Qwen pipeline (event loop safe)
        result = _async_runner.run_coroutine(context_classifier.classify_single_ingredient(clean_ingredient))
        is_vegan_result = result.get('is_vegan', False)
        
        logger.debug(f"Vegan classification: '{ingredient}' → '{clean_ingredient}' → {is_vegan_result}")
        return is_vegan_result
    except Exception as e:
        logger.error(f"Vegan classification failed for '{ingredient}': {e}")
        return False

def parse_ingredients_input(ingredients: Union[str, List[str]]) -> List[str]:
    """
    Handle both string and list inputs to match original submission format.
    
    This function provides flexible input parsing to support various formats
    that users might provide. It handles JSON strings, comma-separated strings,
    and direct lists, making the API more user-friendly.
    
    Expected inputs:
    - String: '["chicken breast", "spinach", "olive oil"]' (JSON format)
    - String: "chicken breast, spinach, olive oil" (comma-separated)
    - List: ["chicken breast", "spinach", "olive oil"] (direct list)
    
    Args:
        ingredients: Input ingredients in any supported format
        
    Returns:
        List of clean ingredient strings
        
    Example:
        >>> parse_ingredients_input('["chicken", "spinach"]')
        ['chicken', 'spinach']
        >>> parse_ingredients_input("chicken, spinach, olive oil")
        ['chicken', 'spinach', 'olive oil']
        >>> parse_ingredients_input(["chicken", "spinach"])
        ['chicken', 'spinach']
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
    
    Evaluates whether a complete recipe is keto-friendly by checking each
    individual ingredient. A recipe is considered keto if ALL ingredients
    are keto-friendly.
    
    Handles both string and list inputs to match original submission format.
    
    Args:
        ingredients: Recipe ingredients in any supported format
        
    Returns:
        bool: True if all ingredients are keto-friendly, False otherwise
        
    Example:
        >>> is_keto(["chicken breast", "spinach", "olive oil"])
        True
        >>> is_keto(["chicken breast", "white rice", "olive oil"])
        False
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
    
    Evaluates whether a complete recipe is vegan-friendly by checking each
    individual ingredient. A recipe is considered vegan if ALL ingredients
    are vegan-friendly.
    
    Handles both string and list inputs to match original submission format.
    
    Args:
        ingredients: Recipe ingredients in any supported format
        
    Returns:
        bool: True if all ingredients are vegan-friendly, False otherwise
        
    Example:
        >>> is_vegan(["spinach", "olive oil", "quinoa"])
        True
        >>> is_vegan(["spinach", "milk", "quinoa"])
        False
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
    """
    Main function for evaluating diet classifiers against ground truth.
    
    This function provides a command-line interface for testing the diet
    classification system against a ground truth dataset. It evaluates
    both keto and vegan classification accuracy and provides detailed
    performance metrics.
    
    Args:
        args: Command line arguments containing ground truth file path
        
    Returns:
        int: Exit code (0 for success, -1 for failure)
        
    Example:
        >>> python diet_classifiers.py --ground_truth data/ground_truth.csv
    """
    logger.info("Starting diet classifier evaluation")
    
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    logger.info(f"Loaded {len(ground_truth)} ground truth records")
    
    try:
        start_time = time.time()
        
        # Apply keto classification
        logger.info("Running keto classification")
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        
        # Apply vegan classification  
        logger.info("Running vegan classification")
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)

        end_time = time.time()
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        print(f"Error: {e}")
        return -1
    finally:
        # Clean up async runner
        _async_runner.close()

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