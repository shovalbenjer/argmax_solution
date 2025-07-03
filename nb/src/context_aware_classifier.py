"""
Context-Aware Diet Classifier - Cache-First Function-Calling Architecture

Integrates structured query results with LLM reasoning for diet classification.
Uses a modern cache-first approach with Qwen Agent -> JSON -> SQL pipeline.
Performance optimized with Redis caching for instant results on known ingredients.
"""
import sqlite3
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import polars as pl
import hashlib
import time

from config import app_config
from llm_client import LLMClient
from database import db_manager
from function_calling_handler import FunctionCallingHandler
from query_engine import translate_json_to_sql
from utils.cache_manager import get_cache_manager

# Configure professional logging
logger = logging.getLogger(__name__)

def execute_sql_query(sql: str, params: List[Any]) -> str:
    """Executes a SQL query with parameters and returns the formatted results."""
    try:
        with db_manager.get_sqlite_connection() as conn:
            df = pl.read_database(query=sql, connection=conn, execute_options={"parameters": params})
            
            if df.is_empty():
                return "No matching data found in the database."
            
            # Convert to a list of dictionaries for clean JSON output
            return json.dumps(df.to_dicts(), indent=2)
            
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return f"Database query error: {e}"

class ContextAwareDietClassifier:
    """
    Enhanced cache-first diet classifier using function-calling agent to query
    the knowledge base and a separate LLM for final classification.
    
    Performance Features:
    - Redis cache check before expensive LLM operations
    - Ingredient-level caching for reusable components
    - Recipe-level caching for complete results
    - Graceful fallback when cache is unavailable
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.function_handler = FunctionCallingHandler()
        self.cache_manager = get_cache_manager()
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
    def _generate_ingredient_cache_key(self, ingredient: str) -> str:
        """Generate a consistent cache key for an ingredient."""
        # Normalize ingredient name for consistent caching
        normalized = ingredient.lower().strip()
        # Remove common quantifiers that don't affect classification
        normalized = normalized.replace("100g", "").replace("1 cup", "").replace("1 tbsp", "").strip()
        return f"ingredient_classification:{normalized}"
    
    def _generate_recipe_cache_key(self, ingredients: List[str]) -> str:
        """Generate a consistent cache key for a recipe."""
        # Sort ingredients for consistent key regardless of order
        sorted_ingredients = sorted([ing.lower().strip() for ing in ingredients])
        ingredients_str = "|".join(sorted_ingredients)
        # Use hash for consistent short keys
        hash_obj = hashlib.md5(ingredients_str.encode())
        return f"recipe_classification:{hash_obj.hexdigest()}"
    
    async def _get_cached_ingredient_classification(self, ingredient: str) -> Optional[Dict[str, Any]]:
        """Check cache for existing ingredient classification."""
        if not self.cache_manager.is_available():
            return None
        
        cache_key = self._generate_ingredient_cache_key(ingredient)
        cached_result = self.cache_manager.get_ingredient_context(ingredient)
        
        if cached_result:
            self.cache_hits += 1
            logger.debug(f"Cache hit for ingredient: {ingredient}")
            return cached_result.get("context") if isinstance(cached_result, dict) else cached_result
        
        self.cache_misses += 1
        return None
    
    async def _cache_ingredient_classification(self, ingredient: str, result: Dict[str, Any]):
        """Cache ingredient classification result."""
        if not self.cache_manager.is_available():
            return
        
        try:
            # Add metadata for cache management
            cache_data = {
                "classification": result,
                "ingredient": ingredient,
                "cached_at": time.time(),
                "cache_version": "2.0"
            }
            
            self.cache_manager.set_ingredient_context(
                ingredient, 
                cache_data,
                ttl=604800  # 1 week for ingredients
            )
            logger.debug(f"Cached classification for ingredient: {ingredient}")
            
        except Exception as e:
            logger.warning(f"Failed to cache ingredient classification: {e}")
    
    async def _get_cached_recipe_classification(self, ingredients: List[str]) -> Optional[Dict[str, Any]]:
        """Check cache for existing recipe classification."""
        if not self.cache_manager.is_available():
            return None
        
        cache_key = self._generate_recipe_cache_key(ingredients)
        recipe_id = hashlib.md5("|".join(sorted(ingredients)).encode()).hexdigest()
        
        cached_result = self.cache_manager.get_classification_result(recipe_id)
        
        if cached_result:
            self.cache_hits += 1
            logger.debug(f"Cache hit for recipe with {len(ingredients)} ingredients")
            return cached_result.get("classification") if isinstance(cached_result, dict) else cached_result
        
        self.cache_misses += 1
        return None
    
    async def _cache_recipe_classification(self, ingredients: List[str], result: Dict[str, Any]):
        """Cache recipe classification result."""
        if not self.cache_manager.is_available():
            return
        
        try:
            recipe_id = hashlib.md5("|".join(sorted(ingredients)).encode()).hexdigest()
            
            # Add metadata for cache management
            cache_data = {
                "recipe_classification": result,
                "ingredients": ingredients,
                "cached_at": time.time(),
                "cache_version": "2.0"
            }
            
            self.cache_manager.set_classification_result(
                recipe_id,
                cache_data,
                ttl=86400  # 1 day for recipes
            )
            logger.debug(f"Cached classification for recipe with {len(ingredients)} ingredients")
            
        except Exception as e:
            logger.warning(f"Failed to cache recipe classification: {e}")

    async def _get_context_for_ingredient(self, ingredient: str) -> str:
        """
        Generates a JSON query, translates it to SQL, and executes it
        to retrieve context for a single ingredient.
        """
        question = f"What are the nutritional values and vegan status for the ingredient: {ingredient}?"
        
        # Step 1: LLM generates a structured JSON query
        json_query = await self.function_handler.generate_json_query(question)
        if "error" in json_query:
            error_msg = f"Failed to generate JSON query for '{ingredient}': {json_query['error']}"
            logger.error(error_msg)
            return error_msg

        # Step 2: Translate JSON to a safe SQL query
        try:
            sql, params = translate_json_to_sql(json_query)
            logger.debug(f"Translated SQL for '{ingredient}': {sql} with params {params}")
        except ValueError as e:
            error_msg = f"Failed to translate JSON to SQL for '{ingredient}': {e}"
            logger.error(error_msg)
            return error_msg
        
        # Step 3: Execute the SQL query
        return execute_sql_query(sql, params)

    async def classify_single_ingredient(self, ingredient: str) -> Dict[str, Any]:
        """
        Classifies a single ingredient using cache-first approach.
        Checks cache before expensive LLM operations.
        """
        self.total_requests += 1
        logger.debug(f"Starting classification for ingredient: {ingredient}")
        
        # Step 1: Check cache first
        cached_result = await self._get_cached_ingredient_classification(ingredient)
        if cached_result:
            logger.debug(f"Returning cached result for ingredient: {ingredient}")
            
            # Extract classification if nested in cache structure
            if "classification" in cached_result:
                return cached_result["classification"]
            elif "is_keto" in cached_result and "is_vegan" in cached_result:
                return cached_result
            else:
                logger.warning(f"Invalid cached data structure for {ingredient}, proceeding with LLM")
        
        # Step 2: Cache miss - perform LLM classification
        logger.debug(f"Cache miss - performing LLM classification for: {ingredient}")
        
        retrieved_context = await self._get_context_for_ingredient(ingredient)

        # Step 3: Use LLM for final classification based on the retrieved context
        prompt = f"""You are a nutrition expert judging an ingredient based on factual data.

        **Ingredient:**
        {ingredient}

        **Data from Knowledge Base:**
        {retrieved_context}

        **Your Task:**
        Analyze the data to classify the ingredient's compliance with keto and vegan diets.
        - **Keto:** Net carbs (carbohydrate_g - fiber_g) should be low (e.g., <= {app_config.KETO_CARBS_THRESHOLD}g per 100g).
        - **Vegan:** Must not be an animal product. Check `is_explicitly_non_vegan` flags and look for indicators like `cholesterol_mg > 0`.

        Provide your final judgment in a valid JSON format.

        **JSON Output Format:**
        {{
          "is_keto": boolean,
          "is_vegan": boolean,
          "reasoning": "A brief, evidence-based explanation for your decision.",
          "confidence": "high | medium | low"
        }}
        """
        
        model_name = "qwen/qwen3-0.6b-gguf:q8_0"
        if not self.llm_client.is_model_available(model_name):
            logger.error(f"Classification model '{model_name}' is not available.")
            return {"error": "Classification model not available"}

        result = await self.llm_client.query_async(model_name, prompt, as_json=True)
        
        # Step 4: Cache the result for future use
        if "error" not in result:
            await self._cache_ingredient_classification(ingredient, result)
        
        return result

    async def classify_recipe(self, ingredients: List[str]) -> Dict[str, Any]:
        """
        Classifies an entire recipe using cache-first approach.
        Checks for cached recipe results before processing individual ingredients.
        """
        self.total_requests += 1
        logger.debug(f"Starting recipe classification for {len(ingredients)} ingredients")
        
        # Step 1: Check for cached recipe result
        cached_recipe = await self._get_cached_recipe_classification(ingredients)
        if cached_recipe:
            logger.debug(f"Returning cached recipe result for {len(ingredients)} ingredients")
            
            # Extract recipe classification if nested
            if "recipe_classification" in cached_recipe:
                return cached_recipe["recipe_classification"]
            elif "recipe_is_keto" in cached_recipe and "recipe_is_vegan" in cached_recipe:
                return cached_recipe
            else:
                logger.warning(f"Invalid cached recipe data structure, proceeding with classification")
        
        # Step 2: Cache miss - classify individual ingredients
        logger.debug(f"Cache miss - performing ingredient-by-ingredient classification")
        
        classifications = await asyncio.gather(
            *(self.classify_single_ingredient(ing) for ing in ingredients)
        )
        
        recipe_is_keto = True
        recipe_is_vegan = True
        full_reasoning = []
        
        for i, result in enumerate(classifications):
            ingredient = ingredients[i]
            if "error" in result:
                # If any ingredient fails, the whole recipe is conservatively marked as non-compliant
                recipe_is_keto = False
                recipe_is_vegan = False
                full_reasoning.append(f"{ingredient}: Classification failed - {result['error']}")
                continue

            if not result.get("is_keto", False):
                recipe_is_keto = False
            if not result.get("is_vegan", False):
                recipe_is_vegan = False
            
            full_reasoning.append(f"{ingredient}: {result.get('reasoning', 'No reasoning provided.')}")
        
        # Step 3: Compile final recipe result
        recipe_result = {
            "recipe_is_keto": recipe_is_keto,
            "recipe_is_vegan": recipe_is_vegan,
            "ingredient_analysis": classifications,
            "summary_reasoning": " ".join(full_reasoning),
            "cache_performance": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0
            }
        }
        
        # Step 4: Cache the recipe result
        await self._cache_recipe_classification(ingredients, recipe_result)
        
        return recipe_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            "cache_available": self.cache_manager.is_available()
        }

if __name__ == '__main__':
    async def test_cache_aware_classifier():
        classifier = ContextAwareDietClassifier()

        # Test 1: Single ingredient (should cache)
        print("\n--- Testing Single Ingredient: Chicken Breast (First Call) ---")
        start_time = time.time()
        result1 = await classifier.classify_single_ingredient("100g chicken breast")
        first_call_time = time.time() - start_time
        print(json.dumps(result1, indent=2))
        print(f"First call time: {first_call_time:.2f}s")

        # Test 2: Same ingredient (should hit cache)
        print("\n--- Testing Same Ingredient: Chicken Breast (Second Call) ---")
        start_time = time.time()
        result2 = await classifier.classify_single_ingredient("100g chicken breast")
        second_call_time = time.time() - start_time
        print(json.dumps(result2, indent=2))
        print(f"Second call time: {second_call_time:.2f}s")
        print(f"Speed improvement: {first_call_time / second_call_time:.1f}x faster")
        
        # Test 3: Full Recipe
        recipe = ["100g chicken breast", "50g spinach", "1 tbsp olive oil"]
        print(f"\n--- Testing Recipe: {recipe} ---")
        recipe_result = await classifier.classify_recipe(recipe)
        print(json.dumps(recipe_result, indent=2))
        
        # Test 4: Performance stats
        print("\n--- Cache Performance Stats ---")
        stats = classifier.get_performance_stats()
        print(json.dumps(stats, indent=2))

    asyncio.run(test_cache_aware_classifier()) 