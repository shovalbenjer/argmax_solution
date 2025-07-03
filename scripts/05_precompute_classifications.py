#!/usr/bin/env python3
"""
Offline Classification Pre-computation Script

This script pre-computes diet classifications for all known ingredients and
recipes, storing results in Redis cache for instant lookup during runtime.

This offline processing strategy dramatically improves online performance by
eliminating expensive LLM calls for previously seen ingredients.

Usage:
    python scripts/05_precompute_classifications.py [--ingredients-only] [--recipes-only] [--batch-size 100]
"""
import sys
import argparse
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3

# Add nb/src to path for accessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import app_config
from utils.cache_manager import get_cache_manager
from diet_classifiers import is_keto, is_vegan
from database import db_manager

class ClassificationPrecomputer:
    """
    Handles offline pre-computation of ingredient and recipe classifications.
    
    This class processes ingredients from the knowledge database and optionally
    recipes from OpenSearch, storing results in Redis for fast runtime access.
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.cache_manager = get_cache_manager()
        self.processed_count = 0
        self.cached_count = 0
        self.error_count = 0
        
    def get_all_ingredients(self) -> List[str]:
        """Get all unique ingredient names from the nutrition database."""
        logger.info("Loading ingredient names from knowledge database...")
        
        try:
            with sqlite3.connect(app_config.DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT name FROM nutrition_facts WHERE name IS NOT NULL AND name != ''")
                ingredients = [row[0] for row in cursor.fetchall()]
                
            logger.info(f"Found {len(ingredients)} unique ingredients in database")
            return ingredients
            
        except Exception as e:
            logger.error(f"Failed to load ingredients from database: {e}")
            return []
    
    def get_sample_recipes(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get sample recipes from OpenSearch for pre-computation."""
        logger.info(f"Loading sample recipes from OpenSearch (limit: {limit})...")
        
        try:
            client = db_manager.get_opensearch_client()
            if not client:
                logger.warning("OpenSearch not available - skipping recipe pre-computation")
                return []
            
            # Search for sample recipes
            response = client.search(
                index="recipes",
                body={
                    "query": {"match_all": {}},
                    "size": limit,
                    "_source": ["title", "ingredients"]
                }
            )
            
            recipes = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                if 'ingredients' in source and source['ingredients']:
                    recipes.append({
                        'id': hit['_id'],
                        'title': source.get('title', 'Unknown Recipe'),
                        'ingredients': source['ingredients']
                    })
            
            logger.info(f"Found {len(recipes)} recipes with ingredient data")
            return recipes
            
        except Exception as e:
            logger.error(f"Failed to load recipes from OpenSearch: {e}")
            return []
    
    async def classify_ingredient_async(self, ingredient: str) -> Optional[Dict[str, Any]]:
        """Classify a single ingredient asynchronously."""
        try:
            # Create ingredient list for classification
            ingredient_list = [ingredient]
            
            # Get classifications
            start_time = time.time()
            is_keto_result = is_keto(ingredient_list)
            is_vegan_result = is_vegan(ingredient_list)
            processing_time = time.time() - start_time
            
            result = {
                "ingredient": ingredient,
                "is_keto": is_keto_result,
                "is_vegan": is_vegan_result,
                "processing_time_ms": processing_time * 1000,
                "timestamp": time.time(),
                "cache_version": "1.0"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to classify ingredient '{ingredient}': {e}")
            self.error_count += 1
            return None
    
    async def classify_recipe_async(self, recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Classify a single recipe asynchronously."""
        try:
            ingredients = recipe['ingredients']
            
            # Handle different ingredient formats
            if isinstance(ingredients, str):
                try:
                    ingredients = json.loads(ingredients)
                except json.JSONDecodeError:
                    # Assume comma-separated
                    ingredients = [ing.strip() for ing in ingredients.split(',')]
            
            if not isinstance(ingredients, list):
                logger.warning(f"Invalid ingredients format for recipe {recipe['id']}")
                return None
            
            # Get classifications
            start_time = time.time()
            is_keto_result = is_keto(ingredients)
            is_vegan_result = is_vegan(ingredients)
            processing_time = time.time() - start_time
            
            result = {
                "recipe_id": recipe['id'],
                "title": recipe['title'],
                "ingredients": ingredients,
                "is_keto": is_keto_result,
                "is_vegan": is_vegan_result,
                "processing_time_ms": processing_time * 1000,
                "timestamp": time.time(),
                "cache_version": "1.0"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to classify recipe '{recipe['id']}': {e}")
            self.error_count += 1
            return None
    
    def precompute_ingredients(self, ingredients: List[str]) -> int:
        """Pre-compute classifications for a list of ingredients."""
        logger.info(f"Pre-computing classifications for {len(ingredients)} ingredients...")
        
        if not self.cache_manager.is_available():
            logger.warning("Redis cache not available - classifications will not be cached")
            return 0
        
        cached_count = 0
        
        # Process ingredients in batches
        for i in range(0, len(ingredients), self.batch_size):
            batch = ingredients[i:i + self.batch_size]
            batch_start = time.time()
            
            logger.info(f"Processing batch {i//self.batch_size + 1}: ingredients {i+1}-{min(i+len(batch), len(ingredients))}")
            
            # Process batch concurrently
            async def process_batch():
                tasks = [self.classify_ingredient_async(ingredient) for ingredient in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            try:
                results = asyncio.run(process_batch())
                
                # Cache successful results
                for result in results:
                    if result and not isinstance(result, Exception):
                        # Cache with ingredient-specific key
                        cache_key = f"ingredient:{result['ingredient'].lower().strip()}"
                        self.cache_manager.set_ingredient_context(
                            result['ingredient'], 
                            result, 
                            ttl=604800  # 1 week
                        )
                        cached_count += 1
                        self.processed_count += 1
                
                batch_time = time.time() - batch_start
                logger.info(f"  Batch completed in {batch_time:.2f}s - {len(batch)} ingredients processed")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                self.error_count += len(batch)
        
        self.cached_count += cached_count
        return cached_count
    
    def precompute_recipes(self, recipes: List[Dict[str, Any]]) -> int:
        """Pre-compute classifications for a list of recipes."""
        logger.info(f"Pre-computing classifications for {len(recipes)} recipes...")
        
        if not self.cache_manager.is_available():
            logger.warning("Redis cache not available - classifications will not be cached")
            return 0
        
        cached_count = 0
        
        # Process recipes in smaller batches (they're more complex)
        recipe_batch_size = max(1, self.batch_size // 5)
        
        for i in range(0, len(recipes), recipe_batch_size):
            batch = recipes[i:i + recipe_batch_size]
            batch_start = time.time()
            
            logger.info(f"Processing recipe batch {i//recipe_batch_size + 1}: recipes {i+1}-{min(i+len(batch), len(recipes))}")
            
            # Process batch sequentially for recipes (they're already complex)
            for recipe in batch:
                result = asyncio.run(self.classify_recipe_async(recipe))
                
                if result:
                    # Cache with recipe-specific key
                    self.cache_manager.set_classification_result(
                        result['recipe_id'],
                        result,
                        ttl=86400  # 1 day
                    )
                    cached_count += 1
                    self.processed_count += 1
            
            batch_time = time.time() - batch_start
            logger.info(f"  Recipe batch completed in {batch_time:.2f}s - {len(batch)} recipes processed")
        
        self.cached_count += cached_count
        return cached_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        if self.cache_manager.is_available():
            return self.cache_manager.get_stats()
        else:
            return {"status": "unavailable"}

def main():
    """Main execution function with command-line argument support."""
    parser = argparse.ArgumentParser(description="Pre-compute diet classifications for caching")
    parser.add_argument("--ingredients-only", action="store_true", 
                       help="Only pre-compute ingredient classifications")
    parser.add_argument("--recipes-only", action="store_true",
                       help="Only pre-compute recipe classifications")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing (default: 50)")
    parser.add_argument("--recipe-limit", type=int, default=500,
                       help="Maximum number of recipes to process (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually doing it")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("OFFLINE CLASSIFICATION PRE-COMPUTATION")
    logger.info("="*60)
    
    # Validate cache availability
    cache_manager = get_cache_manager()
    if not cache_manager.is_available() and not args.dry_run:
        logger.error("Redis cache is not available. Please start Redis server.")
        logger.info("To start Redis with Docker: docker run -d -p 6379:6379 redis:alpine")
        return 1
    
    precomputer = ClassificationPrecomputer(batch_size=args.batch_size)
    
    start_time = time.time()
    
    try:
        if not args.recipes_only:
            # Pre-compute ingredient classifications
            logger.info("Phase 1: Ingredient Pre-computation")
            logger.info("-" * 40)
            
            ingredients = precomputer.get_all_ingredients()
            if not ingredients:
                logger.warning("No ingredients found in database")
            elif args.dry_run:
                logger.info(f"Would process {len(ingredients)} ingredients")
            else:
                ingredient_cached = precomputer.precompute_ingredients(ingredients)
                logger.info(f"Cached {ingredient_cached} ingredient classifications")
        
        if not args.ingredients_only:
            # Pre-compute recipe classifications
            logger.info("\nPhase 2: Recipe Pre-computation")
            logger.info("-" * 40)
            
            recipes = precomputer.get_sample_recipes(limit=args.recipe_limit)
            if not recipes:
                logger.warning("No recipes found in OpenSearch")
            elif args.dry_run:
                logger.info(f"Would process {len(recipes)} recipes")
            else:
                recipe_cached = precomputer.precompute_recipes(recipes)
                logger.info(f"Cached {recipe_cached} recipe classifications")
        
        # Final statistics
        total_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info("PRE-COMPUTATION SUMMARY")
        logger.info("="*60)
        
        if not args.dry_run:
            cache_stats = precomputer.get_cache_statistics()
            
            logger.info(f"Total processed: {precomputer.processed_count}")
            logger.info(f"Total cached: {precomputer.cached_count}")
            logger.info(f"Errors: {precomputer.error_count}")
            logger.info(f"Processing time: {total_time:.2f} seconds")
            
            if cache_stats.get("status") == "connected":
                logger.info(f"Cache keys total: {cache_stats.get('total_keys', 0)}")
                logger.info(f"Cache memory usage: {cache_stats.get('memory_usage', 'unknown')}")
            
            success_rate = (precomputer.cached_count / precomputer.processed_count * 100) if precomputer.processed_count > 0 else 0
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            if success_rate >= 90:
                logger.info("SUCCESS: Pre-computation completed successfully")
                return 0
            else:
                logger.warning("PARTIAL: Pre-computation completed with some failures")
                return 0
        else:
            logger.info("Dry run completed - no actual processing performed")
            return 0
            
    except Exception as e:
        logger.error(f"FAILED: Pre-computation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 