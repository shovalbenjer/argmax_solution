import pandas as pd
from ingredient_parser.core import parse_ingredient
from loguru import logger

# Use the centralized db connection from our new utility
from utils.db import get_db_connection

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


def get_ingredient_context(raw_ingredient: str) -> dict:
    """Analyzes a raw ingredient string and returns a structured, enriched context.

    Detailed Description:
        - This function orchestrates the full analysis of a single ingredient string.
        - **1. Parsing:** It uses the `ingredient-parser` library to break down the raw string
          into its constituent parts (name, quantity, unit). It includes error handling for
          malformed strings.
        - **2. Retrieval:** It calls the various getter functions (`get_nutrition_data`, `get_vegan_info`)
          to retrieve all available information about the parsed ingredient name from the knowledge graph.
        - **3. Normalization:** It performs a deterministic calculation of the total carbohydrates.
          If a unit conversion is required (e.g., from "cups" to "grams"), it retrieves the
          necessary multiplier and applies it.
        - **4. Assembly:** It aggregates all the information—original, parsed, retrieved, and
          normalized—into a single context dictionary, including any errors encountered.

    Parameters:
        - raw_ingredient (str): The raw ingredient string (e.g., "2 cups of flour").

    Returns:
        - dict: A dictionary containing the full context and analysis of the ingredient.

    Libraries Used:
        - ingredient_parser: For robustly parsing structured data from the ingredient string.
        - loguru: For logging warnings and errors during the process.
    """
    context = {
        "original": raw_ingredient,
        "parsed": {},
        "retrieved_nutrition": {},
        "normalized": {},
        "vegan_info": {},
        "errors": []
    }

    # 1. Ingredient Parsing
    try:
        parsed = parse_ingredient(raw_ingredient)
        # ingredient-parser sometimes returns a list, sometimes a single object
        name_obj = parsed.name[0] if isinstance(parsed.name, list) else parsed.name
        name = name_obj.text

        unit_text = parsed.amount[0].unit if parsed.amount and parsed.amount[0].unit else "unit"
        quantity = float(parsed.amount[0].quantity) if parsed.amount else 1.0

        context['parsed'] = {"name": name, "quantity": quantity, "unit": unit_text}
    except Exception as e:
        logger.warning(f"Parsing failed for '{raw_ingredient}': {e}. Using raw string.")
        context['errors'].append(f"Parsing failed: {e}")
        context['parsed'] = {"name": raw_ingredient.lower().strip(), "quantity": 1.0, "unit": "unit"}
        name = context['parsed']['name']

    # 2. Symbolic Retrieval (from knowledge_graph.db)
    retrieved_nutrition = get_nutrition_data(name)
    if retrieved_nutrition:
        context['retrieved_nutrition'] = retrieved_nutrition

    # 3. Deterministic Normalization Engine
    if retrieved_nutrition and context['parsed']['unit'] != 'unit':
        base_carbs = retrieved_nutrition.get('carbohydrates_g', 0) or 0
        
        multiplier = 1.0
        # If the unit is not a weight, we need to convert it to grams
        if context['parsed']['unit'].lower() not in ['g', 'gram', 'grams', 'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds']:
             multiplier = get_unit_conversion(name, context['parsed']['unit'])

        if multiplier:
            # Placeholder for actual weight conversion logic if needed
            grams_quantity = context['parsed']['quantity'] * multiplier
            # Nutrition facts are per 100g
            calculated_carbs = (base_carbs / 100) * grams_quantity
            context['normalized'] = {"calculated_carbs_g": calculated_carbs}
        else:
            context['errors'].append(f"No unit conversion found for '{name}' from '{context['parsed']['unit']}' to grams.")
            
    # 4. Vegan Check
    vegan_info = get_vegan_info(name)
    if vegan_info:
        context['vegan_info'] = vegan_info

    return context 