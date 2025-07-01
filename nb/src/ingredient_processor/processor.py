import sys
from pathlib import Path
import pandas as pd
import json
from ingredient_parser import parse_ingredient
from loguru import logger
from rapidfuzz import process, fuzz
from typing import List, Dict

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.database import db_manager

# --- Database Access ---
def get_all_ingredient_names() -> List[str]:
    """Fetches all unique ingredient names from the knowledge graph for fuzzy matching."""
    try:
        with db_manager.get_sqlite_connection() as conn:
            return pd.read_sql("SELECT DISTINCT name FROM nutrition_facts", conn)['name'].tolist()
    except Exception as e:
        logger.warning(f"Could not load ingredient names: {e}")
        return []

# Lazy-load names for rapidfuzz to avoid module initialization errors
_ALL_INGREDIENT_NAMES = None

def get_ingredient_names_cached() -> List[str]:
    """Get cached ingredient names, loading them if needed."""
    global _ALL_INGREDIENT_NAMES
    if _ALL_INGREDIENT_NAMES is None:
        _ALL_INGREDIENT_NAMES = get_all_ingredient_names()
    return _ALL_INGREDIENT_NAMES

def get_nutrition_data_batch(ingredient_names: List[str]) -> Dict[str, Dict]:
    """Retrieves nutritional data for a batch of ingredients."""
    with db_manager.get_sqlite_connection() as conn:
        query = f"SELECT * FROM nutrition_facts WHERE name IN ({','.join(['?']*len(ingredient_names))})"
        df = pd.read_sql(query, conn, params=ingredient_names)
        return {row['name']: row for row in df.to_dict('records')}

def get_nutrition_data(ingredient_name: str) -> dict | None:
    """Retrieves comprehensive nutritional data for an ingredient from the enhanced knowledge graph.

    Uses shared database manager for consistent access.
    """
    return db_manager.query_nutrition_data(ingredient_name)

def get_unit_conversion(ingredient_name: str, from_unit: str) -> float | None:
    """Retrieves a unit conversion multiplier for converting to grams.

    Uses shared database manager for consistent access.
    """
    with db_manager.get_sqlite_connection() as conn:
        query = """
            SELECT factor FROM unit_conversions
            WHERE unit = :unit AND standard_unit = 'g'
        """
        df = pd.read_sql(query, conn, params={"unit": from_unit})
        if not df.empty:
            return df.iloc[0]['factor']
    return None

def get_vegan_info(ingredient_name: str) -> dict | None:
    """Checks the vegan status of an ingredient against the comprehensive vegan ontology.

    Uses shared database manager for consistent access.
    """
    return db_manager.query_vegan_ontology(ingredient_name)

# --- Main Processor with Fallback ---
def get_context_with_rapidfuzz_fallback(raw_ingredient: str) -> Dict:
    """
    Analyzes a raw ingredient, trying an exact match first, then falling back
    to rapidfuzz to find the top 3 potential matches.
    """
    context = {"original": raw_ingredient, "match_type": "none", "results": []}
    
    # 1. Parse the ingredient
    try:
        parsed = parse_ingredient(raw_ingredient)
        name = (parsed.name[0] if isinstance(parsed.name, list) else parsed.name).text
    except Exception:
        name = raw_ingredient.lower().strip()

    # 2. Try for an exact match
    exact_match = get_nutrition_data_batch([name])
    if exact_match:
        context['match_type'] = 'exact'
        context['results'] = list(exact_match.values())
        return context

    # 3. Fallback to rapidfuzz if no exact match
    context['match_type'] = 'fallback'
    all_names = get_ingredient_names_cached()
    if not all_names:
        return context  # No ingredient names available yet
    fuzzy_matches = process.extract(name, all_names, scorer=fuzz.WRatio, limit=3)
    
    if not fuzzy_matches:
        return context # No fallback matches found

    # Get nutrition data for the fuzzy matches
    match_names = [match[0] for match in fuzzy_matches]
    fuzzy_nutrition = get_nutrition_data_batch(match_names)
    
    # Structure the results
    for match_name, score, _ in fuzzy_matches:
        if match_name in fuzzy_nutrition:
            result = fuzzy_nutrition[match_name]
            result['match_score'] = score
            context['results'].append(result)
            
    return context 