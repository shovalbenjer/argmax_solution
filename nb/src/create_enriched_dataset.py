"""
Creates an enriched recipe dataset for LLM-based classification.

This script orchestrates a pipeline to prepare data for the ground truth
generation process. It performs the following steps:
1.  Fetches a random sample of recipes from the OpenSearch 'recipes' index.
2.  For each recipe, it processes the list of ingredients.
3.  It calls the `ingredient_processor` to analyze each ingredient, matching
    it against a knowledge graph (SQLite DB) to retrieve nutritional information.
4.  It compiles this nutritional data into a "Retrieval-Augmented Generation"
    (RAG) summary.
5.  The original recipe data, along with the new RAG summary, is saved to a
    CSV file, which serves as the input for the main classification script.

This pre-enrichment step is crucial for providing the LLM with the necessary
context to make accurate dietary classifications (e.g., for keto).
"""
import os
import sqlite3
import polars as pl
import json
import logging
from pathlib import Path
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

# Attempt to import the ingredient processor
try:
    from ingredient_processor import analyze_ingredient
    INGREDIENT_PROCESSOR_AVAILABLE = True
except (ImportError, sqlite3.OperationalError) as e:
    logging.warning(f"Could not import or use ingredient_processor: {e}")
    INGREDIENT_PROCESSOR_AVAILABLE = False

# --- Configuration ---
CONFIG = {
    "OUTPUT_DIR": Path("data"),
    "OUTPUT_FILE": Path("data/enriched_recipes_for_llm.csv"),
    "OPENSEARCH_SAMPLE_SIZE": 500,  # Match the sample size for ground truth
    "LOG_LEVEL": logging.INFO
}

logging.basicConfig(level=CONFIG["LOG_LEVEL"], format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def fetch_recipes_from_opensearch(sample_size: int) -> Optional[pl.DataFrame]:
    """Fetches a random sample of recipes from the OpenSearch database.

    Connects to the OpenSearch service and retrieves a specified number of recipes
    using a query with a random score to ensure a diverse sample.

    Args:
        sample_size (int): The number of random recipes to fetch.

    Returns:
        Optional[pl.DataFrame]: A Polars DataFrame containing the fetched recipes
        (with columns '_id', 'title', 'ingredients'), or None if the operation fails.
    """
    opensearch_url = 'http://os:9200'  # Use the Docker service name
    logging.info(f"Connecting to OpenSearch at {opensearch_url}...")

    try:
        client = OpenSearch(hosts=[opensearch_url])
        if not client.ping():
            raise ConnectionError("Could not connect to OpenSearch.")

        logging.info(f"Fetching {sample_size} random recipes from OpenSearch...")
        query = {"size": sample_size, "query": {"function_score": {"query": {"match_all": {}}, "random_score": {}}}, "_source": ["title", "ingredients"]}
        scanner = helpers.scan(client, index="recipes", query=query, scroll='5m', size=1000)

        recipe_data = []
        for i, hit in enumerate(scanner):
            if i >= sample_size:
                break
            source = hit.get('_source', {})
            ingredients_list = source.get('ingredients', [])
            recipe_data.append({
                "_id": hit.get('_id'),
                "title": source.get('title'),
                "ingredients": json.dumps(ingredients_list if isinstance(ingredients_list, list) else [])
            })

        if not recipe_data:
            logging.warning("No recipes were fetched from OpenSearch.")
            return None

        logging.info(f"✅ Successfully fetched {len(recipe_data)} recipes.")
        return pl.DataFrame(recipe_data)

    except Exception as e:
        logging.error(f"❌ Failed to fetch recipes from OpenSearch: {e}", exc_info=True)
        return None


def generate_rag_summary(ingredients: str) -> str:
    """Generates a Retrieval-Augmented Generation (RAG) summary for a list of ingredients.

    For a given JSON string of ingredients, this function parses the list, analyzes
    each ingredient using the `ingredient_processor`, and compiles the retrieved
    nutritional data into a formatted string summary.

    Args:
        ingredients (str): A JSON string representing a list of ingredients.

    Returns:
        str: A formatted string containing the nutritional summary for the
             ingredients, or an error/status message if analysis is not possible.
    """
    if not INGREDIENT_PROCESSOR_AVAILABLE:
        return "Ingredient processor not available. Analysis skipped."
    try:
        ingredient_list = json.loads(ingredients)
        summaries = []
        for ing_string in ingredient_list:
            analysis = analyze_ingredient(ing_string)
            if analysis and analysis.get('db_data'):
                db_data = analysis['db_data']
                summary = (
                    f"- Ingredient: '{analysis.get('parsed_name', 'N/A')}'. "
                    f"Match: {analysis.get('match_method')} ({analysis.get('match_score', 0):.0f}). "
                    f"DB: Carbs={db_data.get('carbohydrates', 'N/A')}g, "
                    f"Protein={db_data.get('protein', 'N/A')}g, "
                    f"Fat={db_data.get('fat', 'N/A')}g."
                )
                summaries.append(summary)
        return "\\n".join(summaries) if summaries else "No nutritional data found for any ingredient."
    except Exception as e:
        logging.warning(f"Could not generate RAG summary for ingredients. Error: {e}")
        return "Error during analysis."


def main():
    """Main function to orchestrate the recipe enrichment pipeline."""
    logging.info("🚀 Starting Recipe Enrichment Pipeline...")
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

    df = fetch_recipes_from_opensearch(CONFIG["OPENSEARCH_SAMPLE_SIZE"])
    if df is None or df.is_empty():
        logging.error("❌ Aborting pipeline: Failed to fetch recipes.")
        return

    logging.info("Enriching recipes with nutritional data from knowledge graph...")
    rag_summaries = [generate_rag_summary(row['ingredients']) for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Analyzing Ingredients")]
    df = df.with_columns(pl.Series(name="rag_summary", values=rag_summaries))

    output_file = CONFIG["OUTPUT_FILE"]
    df.write_csv(output_file)
    logging.info(f"💾 Successfully saved {len(df)} enriched recipes to {output_file}")
    logging.info("🎉 Enrichment pipeline completed successfully!")


if __name__ == "__main__":
    main() 