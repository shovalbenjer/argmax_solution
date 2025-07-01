"""
Core Ingredient Analysis and Normalization Engine.

Implements the Hybrid Retrieval Cascade:
1. Parse ingredient string.
2. Symbolic/Syntactic Search against Elasticsearch.
3. Normalization Engine to calculate nutritional values.
"""
from typing import Dict, Any, Optional
from loguru import logger
from ingredient_parser import parse_ingredient
from rapidfuzz import process, fuzz
from .database_handler import DatabaseHandler

db_handler = DatabaseHandler()
FUZZY_MATCH_THRESHOLD = 90

def get_ingredient_context(raw_ingredient: str) -> Dict:
    """Processes a raw ingredient string to generate a detailed context.

    Detailed Description:
        - This function implements a Hybrid Retrieval Cascade to analyze and enrich a single ingredient string.
        - **Stage 1: Parsing:** It uses the `ingredient-parser` library to extract the name, quantity, and unit from the raw string.
        - **Stage 2: Retrieval:** It performs a search in the database (handled by `DatabaseHandler`) for the parsed ingredient name to find its base nutritional data.
        - **Stage 3: Normalization:** It calculates the total nutritional values (currently just carbohydrates) based on the retrieved data and the parsed quantity. A full implementation would handle unit conversions.
        - **Stage 4: Assembly:** It combines the original string, parsed data, retrieved database record, and normalized values into a single dictionary.

    Parameters:
        - raw_ingredient (str): The raw ingredient string to process (e.g., "2 cups of all-purpose flour").

    Returns:
        - Dict: A dictionary containing the 'original', 'parsed', 'retrieved', and 'normalized' information for the ingredient.

    Libraries Used:
        - loguru: For enhanced logging with better formatting.
        - ingredient_parser: A specialized library for parsing ingredient strings. It is superior to regex or simple string splitting because it understands the common structure of recipe ingredients.
        - rapidfuzz: Although not directly used in this function's final version, it's often used for fuzzy string matching to handle variations in ingredient names.
        - database_handler: A custom module for interacting with the ingredient database.
    """
    # 1. Ingredient Parsing
    try:
        parsed = parse_ingredient(raw_ingredient, discard_isolated_stop_words=True)
        name = ' '.join([item.text for item in parsed.name]) if isinstance(parsed.name, list) else parsed.name.text
        parsed_info = {"name": name, "quantity": float(parsed.amount[0].quantity) if parsed.amount else 1.0, "unit": parsed.amount[0].unit if parsed.amount else "unit"}
    except Exception as e:
        logger.warning(f"Parsing failed for '{raw_ingredient}': {e}. Using raw string.")
        parsed_info = {"name": raw_ingredient, "quantity": 1.0, "unit": "unit"}

    # 2. Symbolic/Syntactic Retrieval
    retrieved_data = db_handler.search_ingredient(parsed_info["name"])
    
    # 3. Normalization Engine
    normalized = {}
    if retrieved_data and "carbohydrates" in retrieved_data:
        try:
            # Note: Unit conversion is a placeholder. A full implementation would query
            # a unit conversion table or use a library.
            conversion_factor = 1.0 
            total_carbs = float(retrieved_data["carbohydrates"]) * parsed_info["quantity"] * conversion_factor
            normalized["total_carbs_g"] = total_carbs
        except (ValueError, TypeError) as e:
            logger.error(f"Normalization calculation error: {e}")
            normalized["total_carbs_g"] = None
            
    # 4. Assemble Final Context
    return {
        "original": raw_ingredient,
        "parsed": parsed_info,
        "retrieved": retrieved_data,
        "normalized": normalized
    } 