"""
Context-Aware Diet Classifier - SOTA Semantic Architecture

PIPELINE: Recipe Ingredient → ingredient-parser → Arctic Semantic SQL → Database → Fuzzy Fallback → Qwen

Key Features:
1. ingredient-parser: Extracts clean ingredient names from recipe strings
2. Arctic Text2SQL: Generates semantic LIKE queries for flexible matching  
3. Fuzzy fallback: RapidFuzz for when semantic queries fail
4. Dual-database: nutrition_facts (keto) + vegan_ontology (vegan)
5. Edge case handling: Compound foods, synonyms, processing variations
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
import re

from config import app_config
from llm_client import LLMClient
from database import db_manager
from function_calling_handler import FunctionCallingHandler
from query_engine import translate_json_to_sql
from utils.cache_manager import get_cache_manager
from ingredient_processor.processor import processor

# Import ingredient-parser for professional ingredient extraction
try:
    from ingredient_parser import parse_ingredient
    INGREDIENT_PARSER_AVAILABLE = True
except ImportError:
    INGREDIENT_PARSER_AVAILABLE = False
    logger.warning("ingredient-parser-nlp not available - using fallback parsing")

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

class SOTASemanticClassifier:
    """
    State-of-the-Art semantic diet classifier using the complete pipeline:
    
    Recipe → ingredient-parser → Arctic semantic SQL → Database → Fuzzy fallback → Qwen
    
    This solves the exact matching problem while maintaining high precision.
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.function_handler = FunctionCallingHandler()
        self.cache_manager = get_cache_manager()
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
    def extract_ingredient_name(self, raw_ingredient: str) -> str:
        """
        Extract clean ingredient name using ingredient-parser (97.8% accuracy).
        
        Examples:
        - "3 pounds pork shoulder, cut into chunks" → "pork shoulder"
        - "2 tbsp extra virgin olive oil" → "olive oil" 
        - "1 cup diced tomatoes" → "tomatoes"
        """
        if INGREDIENT_PARSER_AVAILABLE:
            try:
                parsed = parse_ingredient(raw_ingredient, discard_isolated_stop_words=True)
                if parsed.name and len(parsed.name) > 0:
                    clean_name = parsed.name[0].text.lower().strip()
                    logger.debug(f"Parsed '{raw_ingredient}' → '{clean_name}' (confidence: {parsed.name[0].confidence:.3f})")
                    return clean_name
            except Exception as e:
                logger.warning(f"Ingredient parser failed for '{raw_ingredient}': {e}")
        
        # Fallback: Basic cleaning for when ingredient-parser unavailable
        clean_name = raw_ingredient.lower().strip()
        # Remove quantities and units
        clean_name = re.sub(r'^\d+[\s\w]*\s+', '', clean_name)  
        # Remove preparation instructions
        clean_name = re.sub(r',.*$', '', clean_name)  
        clean_name = clean_name.strip()
        
        logger.debug(f"Fallback parsed '{raw_ingredient}' → '{clean_name}'")
        return clean_name

    async def _get_semantic_keto_context(self, ingredient: str) -> str:
        """
        Arctic-powered semantic search for keto classification.
        
        Uses LIKE queries to find nutritionally similar ingredients,
        prioritizing raw/unprocessed forms.
        """
        clean_ingredient = self.extract_ingredient_name(ingredient)
        
        # Enhanced Arctic prompt for semantic similarity
        question = f"""Find nutritional information for ingredients semantically similar to '{clean_ingredient}' in the nutrition_facts table.

SEMANTIC SEARCH STRATEGY:
1. Use LIKE operators with wildcards for flexible matching (e.g., '%{clean_ingredient}%')
2. Prioritize raw, unprocessed forms (look for 'raw', 'fresh', 'uncooked')  
3. Avoid compound/processed foods (souffle, casserole, prepared dishes)
4. Include variant forms and preparations that maintain base nutrition

TARGET INGREDIENT: {clean_ingredient}

Return nutritional data focusing on: carbohydrate_g, fiber_g, protein_g, total_fat_g for keto assessment."""

        # Generate Arctic semantic query
        json_query = await self.function_handler.generate_json_query(question)
        if "error" in json_query:
            logger.error(f"Failed to generate semantic keto query for '{ingredient}': {json_query['error']}")
            return await self._get_fuzzy_fallback_context(ingredient, "nutrition")

        try:
            sql, params = translate_json_to_sql(json_query)
            logger.debug(f"Semantic keto SQL for '{ingredient}': {sql} with params {params}")
            return execute_sql_query(sql, params)
        except ValueError as e:
            logger.error(f"Failed to translate semantic keto query: {e}")
            return await self._get_fuzzy_fallback_context(ingredient, "nutrition")

    async def _get_semantic_vegan_context(self, ingredient: str) -> str:
        """
        Arctic-powered semantic search for vegan classification.
        
        Searches vegan ontology for animal product detection using semantic matching.
        """
        clean_ingredient = self.extract_ingredient_name(ingredient)
        
        # Enhanced Arctic prompt for vegan semantic search
        question = f"""Search the vegan_ontology table for animal product information about '{clean_ingredient}'.

VEGAN SEMANTIC SEARCH:
1. Check 'term' column with LIKE '%{clean_ingredient}%' for flexible matching
2. Search 'aliases' column for alternative names (synonyms, regional names)
3. Look for parent categories (e.g., 'milk' matches 'whole milk', 'skim milk')
4. Return 'is_explicitly_non_vegan' flag and 'description' for animal origin info

TARGET INGREDIENT: {clean_ingredient}

Find any matches that indicate animal product status."""

        # Generate Arctic semantic query  
        json_query = await self.function_handler.generate_json_query(question)
        if "error" in json_query:
            logger.error(f"Failed to generate semantic vegan query for '{ingredient}': {json_query['error']}")
            return await self._get_fuzzy_fallback_context(ingredient, "vegan")

        try:
            sql, params = translate_json_to_sql(json_query)
            logger.debug(f"Semantic vegan SQL for '{ingredient}': {sql} with params {params}")
            return execute_sql_query(sql, params)
        except ValueError as e:
            logger.error(f"Failed to translate semantic vegan query: {e}")
            return await self._get_fuzzy_fallback_context(ingredient, "vegan")

    async def _get_fuzzy_fallback_context(self, ingredient: str, context_type: str) -> str:
        """
        Fuzzy matching fallback when Arctic semantic search fails.
        
        Uses the existing processor's RapidFuzz implementation for backup.
        """
        logger.info(f"Using fuzzy fallback for {context_type} context: {ingredient}")
        
        try:
            # Use existing processor for comprehensive fuzzy matching
            context = processor.process_ingredient_comprehensive(ingredient)
            
            if context_type == "nutrition" and context.get("nutrition_data"):
                return json.dumps([context["nutrition_data"]], indent=2)
            elif context_type == "vegan" and context.get("vegan_info"): 
                return json.dumps([context["vegan_info"]], indent=2)
            else:
                # Try fuzzy matches if direct lookup failed
                if context.get("fuzzy_matches"):
                    best_match = context["fuzzy_matches"][0][0]
                    logger.info(f"Fuzzy fallback found: '{best_match}' for '{ingredient}'")
                    
                    if context_type == "nutrition":
                        fallback_data = processor.get_nutrition_data(best_match)
                        return json.dumps([fallback_data] if fallback_data else [], indent=2)
        else:
                        fallback_data = processor.get_vegan_info(best_match)
                        return json.dumps([fallback_data] if fallback_data else [], indent=2)
                        
                return "No matching data found in fuzzy fallback."
                
        except Exception as e:
            logger.error(f"Fuzzy fallback failed for '{ingredient}': {e}")
            return f"Fuzzy fallback error: {e}"

    async def _get_cached_ingredient_classification(self, ingredient: str) -> Optional[Dict[str, Any]]:
        """Check cache for existing ingredient classification."""
        if not self.cache_manager.is_available():
            return None
        
        cached_result = self.cache_manager.get_ingredient_context(ingredient)
        
        if cached_result:
            self.cache_hits += 1
            logger.debug(f"Cache hit for ingredient: {ingredient}")
            return cached_result.get("classification") if isinstance(cached_result, dict) else cached_result
        
        self.cache_misses += 1
        return None

    async def _cache_ingredient_classification(self, ingredient: str, result: Dict[str, Any]):
        """Cache ingredient classification result."""
        if not self.cache_manager.is_available():
            return
        
        try:
            cache_data = {
                "classification": result,
                "ingredient": ingredient,
                "cached_at": time.time(),
                "cache_version": "4.0"  # SOTA semantic version
            }
            
            self.cache_manager.set_ingredient_context(
                ingredient, 
                cache_data,
                ttl=604800  # 1 week for ingredients
            )
            logger.debug(f"Cached SOTA classification for ingredient: {ingredient}")
            
        except Exception as e:
            logger.warning(f"Failed to cache ingredient classification: {e}")

    async def classify_single_ingredient(self, ingredient: str) -> Dict[str, Any]:
        """
        SOTA semantic ingredient classification pipeline.
        
        FULL PIPELINE:
        1. Cache check (performance)
        2. ingredient-parser: Extract clean name from recipe text
        3. Arctic Text2SQL: Generate semantic LIKE queries
        4. Database: Flexible matching with wildcards  
        5. Fuzzy fallback: RapidFuzz if semantic fails
        6. Qwen reasoning: Final classification with edge case handling
        7. Cache result: Future performance
        """
        self.total_requests += 1
        logger.debug(f"Starting SOTA semantic classification for: {ingredient}")
        
        # Step 1: Check cache first
        cached_result = await self._get_cached_ingredient_classification(ingredient)
        if cached_result:
            logger.debug(f"Returning cached result for ingredient: {ingredient}")
            return cached_result
        
        # Step 2: Extract clean ingredient name
        clean_ingredient = self.extract_ingredient_name(ingredient)
        logger.debug(f"Extracted ingredient name: '{ingredient}' → '{clean_ingredient}'")
        
        # Step 3: Parallel semantic context retrieval 
        logger.debug(f"Cache miss - performing SOTA semantic lookup for: {clean_ingredient}")
        
        keto_context, vegan_context = await asyncio.gather(
            self._get_semantic_keto_context(ingredient),
            self._get_semantic_vegan_context(ingredient)
        )

        # Step 4: Enhanced Qwen reasoning with SOTA prompting
        prompt = f"""You are an expert nutrition classifier using SOTA semantic search results.

**ORIGINAL INGREDIENT:** {ingredient}
**EXTRACTED NAME:** {clean_ingredient}

**KETO CONTEXT (nutrition_facts semantic search):**
{keto_context}

**VEGAN CONTEXT (vegan_ontology semantic search):**
{vegan_context}

**SOTA CLASSIFICATION PROTOCOL:**

**KETO RULES:**
- Carbohydrates ≤ {app_config.KETO_CARBS_THRESHOLD}g per 100g = KETO-FRIENDLY  
- Use semantic matches prioritizing raw/unprocessed forms
- If multiple matches, select the most representative base ingredient
- Handle preparation variants (e.g., "diced tomatoes" = "tomatoes")

**VEGAN RULES:**
- If vegan_ontology shows is_explicitly_non_vegan=1 → NON-VEGAN
- Unknown plant ingredients → DEFAULT VEGAN (conservative)
- Animal products: meat, dairy, eggs, honey, gelatin, animal fat
- Plant foods: vegetables, fruits, grains, nuts, legumes → VEGAN

**SEMANTIC MATCHING PRIORITY:**
1. Raw/fresh forms over processed
2. Basic ingredients over compound foods  
3. Higher semantic similarity scores
4. Nutritional representativeness

**EDGE CASE HANDLING:**
- Compound ingredients: Extract base component
- Preparation variants: Focus on core ingredient
- Regional synonyms: Use semantic matching
- Missing data: Conservative classification

**OUTPUT FORMAT:**
{{
  "is_keto": boolean,
  "is_vegan": boolean,
  "reasoning": "Detailed evidence citing semantic matches and database findings",
  "confidence": "high | medium | low",
  "semantic_match_quality": "excellent | good | poor | fallback",
  "extracted_ingredient": "{clean_ingredient}"
}}"""
        
        model_name = "qwen/qwen3-0.6b-gguf:q8_0"
        if not self.llm_client.is_model_available(model_name):
            logger.error(f"Classification model '{model_name}' is not available.")
            return {"error": "Classification model not available"}

        result = await self.llm_client.query_async(model_name, prompt, as_json=True)
        
        # Step 5: Cache the result for future use
        if "error" not in result:
            await self._cache_ingredient_classification(ingredient, result)
        
        return result

    async def classify_recipe(self, ingredients: List[str]) -> Dict[str, Any]:
        """
        SOTA recipe classification with comprehensive semantic analysis.
        """
        self.total_requests += 1
        logger.debug(f"Starting SOTA recipe classification for {len(ingredients)} ingredients")
        
        # Parallel classification of all ingredients
        classifications = await asyncio.gather(
            *(self.classify_single_ingredient(ing) for ing in ingredients)
        )
        
        recipe_is_keto = True
        recipe_is_vegan = True
        ingredient_details = []
        semantic_quality_stats = {"excellent": 0, "good": 0, "poor": 0, "fallback": 0}
        
        for i, result in enumerate(classifications):
            ingredient = ingredients[i]
            
            if "error" in result:
                recipe_is_keto = False
                recipe_is_vegan = False
                ingredient_details.append({
                    "ingredient": ingredient,
                    "error": result.get("error", "Unknown error"),
                    "is_keto": False,
                    "is_vegan": False
                })
                continue

            # Extract classification results
            is_keto = result.get("is_keto", False)
            is_vegan = result.get("is_vegan", False)
            semantic_quality = result.get("semantic_match_quality", "unknown")
            
            if semantic_quality in semantic_quality_stats:
                semantic_quality_stats[semantic_quality] += 1
            
            ingredient_details.append({
                "ingredient": ingredient,
                "extracted_name": result.get("extracted_ingredient", ""),
                "is_keto": is_keto,
                "is_vegan": is_vegan,
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", "medium"),
                "semantic_quality": semantic_quality
            })
            
            # Recipe fails if any ingredient fails
            if not is_keto:
                recipe_is_keto = False
            if not is_vegan:
                recipe_is_vegan = False
        
        # Compile comprehensive result
        recipe_result = {
            "recipe_is_keto": recipe_is_keto,
            "recipe_is_vegan": recipe_is_vegan,
            "ingredient_count": len(ingredients),
            "successful_classifications": len([d for d in ingredient_details if "error" not in d]),
            "semantic_quality_distribution": semantic_quality_stats,
            "ingredient_analysis": ingredient_details,
            "cache_performance": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0
            }
        }
        
        return recipe_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            "cache_available": self.cache_manager.is_available(),
            "ingredient_parser_available": INGREDIENT_PARSER_AVAILABLE
        }

# For backward compatibility
ContextAwareDietClassifier = SOTASemanticClassifier

if __name__ == '__main__':
    async def test_sota_semantic_classifier():
        classifier = SOTASemanticClassifier()

        # Test realistic recipe ingredients that need semantic matching
        test_ingredients = [
            "3 pounds pork shoulder, cut into chunks",  # Should find "pork" semantically
            "2 tbsp extra virgin olive oil",            # Should find "olive oil"  
            "1 cup diced tomatoes",                     # Should find "tomatoes"
            "100g fresh spinach leaves",                # Should find "spinach, raw"
            "2 cups whole milk",                        # Should find milk products
            "1 lb ground beef, 80/20"                   # Should find beef variants
        ]

        print("=== TESTING SOTA SEMANTIC CLASSIFIER ===")
        
        for ingredient in test_ingredients:
            print(f"\n--- Testing: {ingredient} ---")
            start_time = time.time()
            result = await classifier.classify_single_ingredient(ingredient)
            end_time = time.time()
            
            print(json.dumps(result, indent=2))
            print(f"Classification time: {end_time - start_time:.2f}s")
        
        # Test complete recipe
        print(f"\n--- Testing Complete Recipe ---")
        recipe = [
            "3 pounds pork shoulder, cut into chunks", 
            "2 tbsp extra virgin olive oil",
            "1 cup diced tomatoes",
            "100g fresh spinach leaves"
        ]
        recipe_result = await classifier.classify_recipe(recipe)
        print(json.dumps(recipe_result, indent=2))
        
        # Performance stats
        print("\n--- SOTA Performance Stats ---")
        stats = classifier.get_performance_stats()
        print(json.dumps(stats, indent=2))

    asyncio.run(test_sota_semantic_classifier()) 