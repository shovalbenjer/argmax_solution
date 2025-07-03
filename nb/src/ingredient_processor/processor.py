import sys
from pathlib import Path
import pandas as pd
import json
from ingredient_parser import parse_ingredient
from loguru import logger
from rapidfuzz import process, fuzz
from typing import List, Dict, Optional, Tuple
import re

# Add nb/src to path for database access
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import db_manager
from ingredient_normalizer import normalizer

# --- Enhanced Ingredient Processing with Normalization ---

class EnhancedIngredientProcessor:
    """
    Enhanced ingredient processor that combines normalization, fuzzy matching,
    and database access for optimal caching and lookup performance.
    """
    
    def __init__(self):
        self._ingredient_names_cache = None
        self._nutrition_cache = {}
        self._vegan_cache = {}
        
    def get_all_ingredient_names(self) -> List[str]:
        """Fetches all unique ingredient names from the knowledge graph for fuzzy matching."""
        try:
            with db_manager.get_sqlite_connection() as conn:
                return pd.read_sql("SELECT DISTINCT name FROM nutrition_facts", conn)['name'].tolist()
        except Exception as e:
            logger.warning(f"Could not load ingredient names: {e}")
            return []

    def get_ingredient_names_cached(self) -> List[str]:
        """Get cached ingredient names, loading them if needed."""
        if self._ingredient_names_cache is None:
            self._ingredient_names_cache = self.get_all_ingredient_names()
            logger.info(f"Loaded {len(self._ingredient_names_cache)} ingredient names for fuzzy matching")
        return self._ingredient_names_cache

    def normalize_ingredient(self, raw_ingredient: str) -> str:
        """
        Normalize ingredient text for consistent lookup and caching.
        Uses the shared ingredient normalizer.
        """
        return normalizer.normalize_ingredient(raw_ingredient)
    
    def get_cache_key(self, raw_ingredient: str) -> str:
        """
        Generate a cache key for the ingredient.
        Uses the shared ingredient normalizer.
        """
        return normalizer.get_cache_key(raw_ingredient)
    
    def parse_ingredient_safely(self, raw_ingredient: str) -> Tuple[str, Optional[dict]]:
        """
        Safely parse ingredient text and extract the base ingredient name.
        Returns (normalized_name, parsed_data) tuple.
        """
        try:
            parsed = parse_ingredient(raw_ingredient)
            if parsed and hasattr(parsed, 'name') and parsed.name:
                base_name = parsed.name.text if hasattr(parsed.name, 'text') else str(parsed.name)
                normalized = self.normalize_ingredient(base_name)
                
                # Extract additional parsed data
                parsed_data = {
                    'quantity': getattr(parsed, 'quantity', None),
                    'unit': getattr(parsed, 'unit', None),
                    'preparation': getattr(parsed, 'preparation', None),
                    'original_name': base_name
                }
                
                return normalized, parsed_data
            else:
                # Fallback to direct normalization
                normalized = self.normalize_ingredient(raw_ingredient)
                return normalized, None
                
        except Exception as e:
            logger.debug(f"Ingredient parsing failed for '{raw_ingredient}': {e}")
            normalized = self.normalize_ingredient(raw_ingredient)
            return normalized, None

    def get_nutrition_data_batch(self, ingredient_names: List[str]) -> Dict[str, Dict]:
        """Retrieves nutritional data for a batch of ingredients with caching."""
        # Check cache first
        cached_results = {}
        uncached_names = []
        
        for name in ingredient_names:
            if name in self._nutrition_cache:
                cached_results[name] = self._nutrition_cache[name]
            else:
                uncached_names.append(name)
        
        # Fetch uncached items from database
        if uncached_names:
            try:
                with db_manager.get_sqlite_connection() as conn:
                    placeholders = ','.join(['?'] * len(uncached_names))
                    query = f"SELECT * FROM nutrition_facts WHERE name IN ({placeholders})"
                    df = pd.read_sql(query, conn, params=uncached_names)
                    
                    # Cache the results
                    for row in df.to_dict('records'):
                        self._nutrition_cache[row['name']] = row
                        cached_results[row['name']] = row
                        
            except Exception as e:
                logger.error(f"Failed to fetch nutrition data for batch: {e}")
        
        return cached_results

    def get_nutrition_data(self, ingredient_name: str) -> Optional[Dict]:
        """Retrieves comprehensive nutritional data for an ingredient with caching."""
        if ingredient_name in self._nutrition_cache:
            return self._nutrition_cache[ingredient_name]
        
        result = db_manager.query_nutrition_data(ingredient_name)
        if result:
            self._nutrition_cache[ingredient_name] = result
        
        return result

    def get_vegan_info(self, ingredient_name: str) -> Optional[Dict]:
        """Checks the vegan status of an ingredient with caching."""
        if ingredient_name in self._vegan_cache:
            return self._vegan_cache[ingredient_name]
        
        result = db_manager.query_vegan_ontology(ingredient_name)
        if result:
            self._vegan_cache[ingredient_name] = result
        
        return result

    def find_best_matches(self, normalized_name: str, limit: int = 3) -> List[Tuple[str, float]]:
        """
        Find the best fuzzy matches for a normalized ingredient name.
        Returns list of (ingredient_name, match_score) tuples.
        """
        all_names = self.get_ingredient_names_cached()
        if not all_names:
            return []
        
        # Use rapidfuzz for fuzzy matching
        matches = process.extract(normalized_name, all_names, scorer=fuzz.WRatio, limit=limit)
        return [(match[0], match[1]) for match in matches if match[1] > 60]  # Only return matches above 60% similarity

    def process_ingredient_comprehensive(self, raw_ingredient: str) -> Dict:
        """
        Comprehensive ingredient processing with normalization, exact matching,
        and fuzzy fallback. Optimized for caching and performance.
        
        Returns a structured context dictionary with all relevant information.
        """
        context = {
            "original": raw_ingredient,
            "normalized": "",
            "cache_key": "",
            "parsed_data": None,
            "match_type": "none",
            "nutrition_data": None,
            "vegan_info": None,
            "fuzzy_matches": [],
            "confidence": 0.0
        }
        
        # Step 1: Parse and normalize the ingredient
        normalized_name, parsed_data = self.parse_ingredient_safely(raw_ingredient)
        context["normalized"] = normalized_name
        context["cache_key"] = self.get_cache_key(raw_ingredient)
        context["parsed_data"] = parsed_data
        
        if not normalized_name:
            logger.warning(f"Could not normalize ingredient: '{raw_ingredient}'")
            return context
        
        # Step 2: Try exact match first
        nutrition_data = self.get_nutrition_data(normalized_name)
        if nutrition_data:
            context["match_type"] = "exact"
            context["nutrition_data"] = nutrition_data
            context["vegan_info"] = self.get_vegan_info(normalized_name)
            context["confidence"] = 1.0
            return context
        
        # Step 3: Try fuzzy matching as fallback
        fuzzy_matches = self.find_best_matches(normalized_name, limit=3)
        if fuzzy_matches:
            context["match_type"] = "fuzzy"
            context["fuzzy_matches"] = fuzzy_matches
            
            # Get nutrition data for the best match
            best_match_name = fuzzy_matches[0][0]
            best_match_score = fuzzy_matches[0][1]
            
            context["nutrition_data"] = self.get_nutrition_data(best_match_name)
            context["vegan_info"] = self.get_vegan_info(best_match_name)
            context["confidence"] = best_match_score / 100.0
            
            logger.info(f"Fuzzy match for '{raw_ingredient}': '{best_match_name}' (score: {best_match_score})")
        
        return context

# Global processor instance
processor = EnhancedIngredientProcessor()

# --- Legacy API Compatibility ---
def get_all_ingredient_names() -> List[str]:
    """Legacy compatibility function."""
    return processor.get_all_ingredient_names()

def get_ingredient_names_cached() -> List[str]:
    """Legacy compatibility function."""
    return processor.get_ingredient_names_cached()

def get_nutrition_data_batch(ingredient_names: List[str]) -> Dict[str, Dict]:
    """Legacy compatibility function."""
    return processor.get_nutrition_data_batch(ingredient_names)

def get_nutrition_data(ingredient_name: str) -> Optional[Dict]:
    """Legacy compatibility function."""
    return processor.get_nutrition_data(ingredient_name)

def get_vegan_info(ingredient_name: str) -> Optional[Dict]:
    """Legacy compatibility function."""
    return processor.get_vegan_info(ingredient_name)

def get_context_with_rapidfuzz_fallback(raw_ingredient: str) -> Dict:
    """
    Legacy compatibility function that maintains the original API.
    Now powered by the enhanced processor.
    """
    context = processor.process_ingredient_comprehensive(raw_ingredient)
    
    # Transform to legacy format
    legacy_context = {
        "original": context["original"],
        "match_type": context["match_type"],
        "results": []
    }
    
    if context["nutrition_data"]:
        result = context["nutrition_data"].copy()
        if context["match_type"] == "fuzzy" and context["fuzzy_matches"]:
            result["match_score"] = context["fuzzy_matches"][0][1]
        legacy_context["results"] = [result]
    
    return legacy_context 