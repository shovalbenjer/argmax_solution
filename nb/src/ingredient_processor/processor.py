import pandas as pd
from ingredient_parser import parse_ingredient
from loguru import logger
from rapidfuzz import process, fuzz
from typing import List, Dict

# Use the centralized db connection from our new utility
from utils.db import get_db_connection

# --- Database Access ---
def get_all_ingredient_names() -> List[str]:
    """Fetches all unique ingredient names from the knowledge graph for fuzzy matching."""
    with get_db_connection() as conn:
        return pd.read_sql("SELECT DISTINCT name FROM nutrition_facts", conn)['name'].tolist()

# Pre-load names for rapidfuzz
ALL_INGREDIENT_NAMES = get_all_ingredient_names()

def get_nutrition_data_batch(ingredient_names: List[str]) -> Dict[str, Dict]:
    """Retrieves nutritional data for a batch of ingredients."""
    with get_db_connection() as conn:
        query = f"SELECT * FROM nutrition_facts WHERE name IN ({','.join(['?']*len(ingredient_names))})"
        df = pd.read_sql(query, conn, params=ingredient_names)
        return {row['name']: row for row in df.to_dict('records')}

def get_nutrition_data(ingredient_name: str) -> dict | None:
    """Retrieves nutritional data for an ingredient from the knowledge graph.

    Detailed Description:
        - This function connects to the SQLite database using a centralized connection handler.
        - It executes a SQL query to select all columns from the `nutrition_facts` table
          for a given ingredient name.
        - It uses pandas to execute the query and fetch the results for convenience.

    Parameters:
        - ingredient_name (str): The name of the ingredient to look up.

    Returns:
        - dict | None: A dictionary containing the nutritional data if the ingredient is found,
          otherwise None.

    Libraries Used:
        - pandas: Used with `read_sql` to execute the SQL query and return the results
          in a structured DataFrame, which is easy to handle.
        - utils.db: A custom utility to get a managed database connection.
    """
    with get_db_connection() as conn:
        query = "SELECT * FROM nutrition_facts WHERE name = :name"
        df = pd.read_sql(query, conn, params={"name": ingredient_name})
        if not df.empty:
            return df.iloc[0].to_dict()
    return None

def get_unit_conversion(ingredient_name: str, from_unit: str) -> float | None:
    """Retrieves a unit conversion multiplier for converting to grams.

    Detailed Description:
        - This function queries the `unit_conversions` table in the knowledge graph.
        - It specifically looks for a multiplier to convert a given volumetric or non-standard
          unit (`from_unit`) into grams ('g') for a specific ingredient.

    Parameters:
        - ingredient_name (str): The name of the ingredient for the conversion.
        - from_unit (str): The original unit (e.g., "cup", "tbsp").

    Returns:
        - float | None: The multiplication factor to convert to grams, or None if not found.
    """
    with get_db_connection() as conn:
        query = """
            SELECT multiplier FROM unit_conversions
            WHERE ingredient_name = :name AND from_unit = :unit AND to_unit = 'g'
        """
        df = pd.read_sql(query, conn, params={"name": ingredient_name, "unit": from_unit})
        if not df.empty:
            return df.iloc[0]['multiplier']
    return None

def get_vegan_info(ingredient_name: str) -> dict | None:
    """Checks the vegan status of an ingredient against the vegan ontology.

    Detailed Description:
        - This function queries the `vegan_ontology` table in the knowledge graph.
        - It checks for the presence of the ingredient term and returns its `is_vegan` status.

    Parameters:
        - ingredient_name (str): The ingredient term to check.

    Returns:
        - dict | None: A dictionary containing the boolean vegan status if the term is found,
          otherwise None.
    """
    with get_db_connection() as conn:
        query = "SELECT is_vegan FROM vegan_ontology WHERE term = :term"
        df = pd.read_sql(query, conn, params={"term": ingredient_name})
        if not df.empty:
            return {"is_vegan_term": bool(df.iloc[0]['is_vegan'])}
    return None

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
    fuzzy_matches = process.extract(name, ALL_INGREDIENT_NAMES, scorer=fuzz.WRatio, limit=3)
    
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