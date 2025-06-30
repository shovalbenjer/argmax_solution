"""Populates an OpenSearch database with recipe and ingredient data.

This script manages the end-to-end process of indexing recipe data for the
search application. It connects to an OpenSearch service, creates the necessary
indices with appropriate mappings, and then reads recipe data from a Parquet
file.

The script performs two main indexing tasks:
1.  **Recipe Indexing**: Each recipe is indexed as a document in the 'recipes'
    index.
2.  **Ingredient Indexing**: A unique, normalized list of all ingredients across
    all recipes is created and indexed into the 'ingredients' index.

The script is designed to be run from the command line and includes checks to
prevent re-indexing if data already exists.

Note:
    This script is a duplicate of `nb/src/index_data.py`. This is a known
    architectural issue that should be resolved by creating a shared package.
"""
import sys
import string
import json
import re
import logging
from opensearchpy import OpenSearch
from time import sleep
import polars as pl
from pathlib import Path
from typing import List, Dict
from argparse import ArgumentParser
from tqdm import tqdm

# Configure logging
logging.getLogger('opensearch').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_ingredient(ingredient: str) -> str:
    """Cleans and standardizes a raw ingredient string.

    This function applies a series of transformations to an ingredient string
    to create a normalized representation. This helps in creating a canonical
    list of unique ingredients.

    Args:
        ingredient (str): The raw ingredient string.

    Returns:
        str: The normalized ingredient string.
    """
    if type(ingredient) != str:
        return str(ingredient)
    ingredient = ingredient.lower().strip()
    ingredient = ingredient.rsplit(',', 1)[0]
    ingredient = re.sub(r"\([^()]+\)", "", ingredient)
    ingredient = re.sub(r'\d+\s*\d*/\d*', '', ingredient)
    ingredient = ingredient.translate(
        {ord(c): ' ' for c in string.punctuation})
    measurements = [
        'cup', 'cups', 'can', 'cans', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon', 'teaspoons', 'tsp',
        'ounce', 'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs', 'gram', 'grams', 'g',
        'kilogram', 'kilograms', 'kg', 'milliliter', 'milliliters', 'ml', 'liter', 'liters', 'l',
        'pinch', 'pinches', 'dash', 'dashes', 'piece', 'pieces', 'slice', 'slices', 'small', 'medium', 'large',
        'cube', 'cubes', 'inch', 'inches', 'cm', 'mm', 'quart', 'quarts', 'qt', 'jar', 'scoop', 'scoops',
        'gallon', 'gallons', 'gal', 'pint', 'pints', 'pt', 'fluid ounce', 'fluid ounces', 'fl oz', 'package', 'packages', 'pkg', 'pack', 'packs'
    ]
    pattern = r'\b(' + '|'.join(measurements) + r')\b'
    ingredient = re.sub(pattern, '', ingredient)
    ingredient = re.sub(r'[^a-z\s]', '', ingredient)
    if ingredient.endswith('ies'):
        ingredient = ingredient[:-3]+'y'
    elif ingredient.endswith('es'):
        if ingredient[-3] in {'s', 'x', 'z'}:
            ingredient = ingredient[:-2]
        else:
            ingredient = ingredient[:-1]
    elif ingredient.endswith('s'):
        ingredient = ingredient[:-1]
    ingredient = ' '.join(ingredient.split())
    return ingredient


def wait_for_opensearch(client: OpenSearch, max_retries: int = 30, retry_interval: int = 2) -> bool:
    """Polls the OpenSearch client to wait for the service to be ready.

    This function attempts to connect to OpenSearch by repeatedly calling the
    client's `ping()` method. It is used to ensure that the script does not
    proceed before its database backend is available.

    Args:
        client (OpenSearch): The OpenSearch client instance.
        max_retries (int): The maximum number of connection attempts.
        retry_interval (int): The delay in seconds between retries.

    Returns:
        bool: True if the connection succeeds, False otherwise.
    """
    for i in range(max_retries):
        try:
            if client.ping():
                return True
        except:
            pass
        print(
            f"Waiting for OpenSearch to be ready... (attempt {i+1}/{max_retries})")
        sleep(retry_interval)
    return False


def check_data_exists(client: OpenSearch) -> bool:
    """Checks if data already exists in the OpenSearch indices.

    This function queries the 'recipes' and 'ingredients' indices to see if
    they contain any documents.

    Args:
        client (OpenSearch): An initialized OpenSearch client.

    Returns:
        bool: True if both indices exist and contain at least one document,
            False otherwise.
    """
    try:
        # Check if indices exist and have data
        recipes_count = client.count(index="recipes")["count"]
        ingredients_count = client.count(index="ingredients")["count"]

        if recipes_count > 0 and ingredients_count > 0:
            logger.info(
                f"Found existing data: {recipes_count} recipes and {ingredients_count} ingredients")
            return True

        return False
    except Exception as e:
        logger.error(f"Error checking data existence: {e}")
        return False


def delete_existing_data(client: OpenSearch):
    """Deletes the 'recipes' and 'ingredients' indices from OpenSearch.

    Args:
        client (OpenSearch): An initialized OpenSearch client.
    """
    try:
        if client.indices.exists(index="recipes"):
            client.indices.delete(index="recipes")
            logger.info("Deleted existing recipes index")
        if client.indices.exists(index="ingredients"):
            client.indices.delete(index="ingredients")
            logger.info("Deleted existing ingredients index")
    except Exception as e:
        logger.error(f"Error deleting existing data: {e}")


def create_index(client: OpenSearch):
    """Creates the 'recipes' and 'ingredients' indices with specific mappings.

    This function defines and applies the schema for the search indices. If the
    indices already exist, this function does nothing.

    Args:
        client (OpenSearch): An initialized OpenSearch client.
    """
    if not client.indices.exists(index="recipes"):
        mappings = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "ingredients": {"type": "text"},
                    "instructions": {"type": "text"},
                    "photo_url": {"type": "keyword"}
                }
            }
        }
        client.indices.create(index="recipes", body=mappings)
        logger.info(f"Created index: recipes")

    if not client.indices.exists(index="ingredients"):
        mappings = {
            "mappings": {
                "properties": {
                    "ingredients": {"type": "text"}
                }
            }
        }
        client.indices.create(index="ingredients", body=mappings)
        logger.info(f"Created index: ingredients")


def batch_index_recipes(client: OpenSearch, recipes: List[Dict], batch_size: int = 10240):
    """Indexes recipes and ingredients into OpenSearch in batches.

    This function uses the OpenSearch bulk API for efficient indexing. It
    processes a list of recipe dictionaries, sending them to the 'recipes'
    index. It also extracts, normalizes, and collects a unique set of all
    ingredients, which are then bulk-indexed into the 'ingredients' index.

    Args:
        client (OpenSearch): An initialized OpenSearch client.
        recipes (List[Dict]): A list of recipe dictionaries to index.
        batch_size (int): The number of documents to include in each bulk request.
    """
    actions = []
    ingredients = set()
    for recipe in recipes:
        actions.append({"index": {"_index": "recipes"}})
        actions.append(recipe)
        ingredients |= {normalize_ingredient(
            ing) for ing in recipe["ingredients"]}
        if len(actions) >= batch_size * 2:
            client.bulk(body=actions)
            # logger.info(f"Indexed {len(actions)//2} recipes")
            actions = []

    # Index any remaining recipes
    if actions:
        client.bulk(body=actions)
        # logger.info(f"Indexed {len(actions)//2} recipes")

    actions = []
    for ing in ingredients:
        actions.append({"index": {"_index": "ingredients"}})
        actions.append({"ingredients": ing})
    client.bulk(body=actions)
    # logger.info(f"Indexed {len(actions)//2} ingredients")


def main(args):
    """Main function to orchestrate the OpenSearch indexing pipeline.

    - Initializes and waits for the OpenSearch client.
    - Checks if data already exists to prevent re-indexing.
    - Creates indices if they don't exist.
    - Reads recipe data from a Parquet file.
    - Indexes the recipes and ingredients in batches.

    Args:
        args: An object containing command-line arguments, as returned by
            `ArgumentParser.parse_args()`. It should include 'opensearch_url',
            'data_file', and 'batch_size'.
    """
    # Initialize OpenSearch client
    client = OpenSearch(
        hosts=[args.opensearch_url],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Wait for OpenSearch to be ready
    if not wait_for_opensearch(client):
        logger.error("Failed to connect to OpenSearch")
        sys.exit(1)

    # Check if data already exists
    if check_data_exists(client):
        logger.info("Data already exists in OpenSearch, skipping indexing")
        return

    # Read data from Parquet using Polars
    try:
        df = pl.read_parquet(args.data_file)
        logger.info(f"Loaded {len(df)} recipes from {args.data_file}")
    except Exception as e:
        logger.error(f"Error reading data from {args.data_file}: {e}")
        sys.exit(1)

    # Convert DataFrame to a list of dictionaries for indexing
    recipes_to_index = df.to_dicts()
    
    # Create indices if they don't exist
    create_index(client)

    # Index recipes in batches
    logger.info("Starting bulk indexing of recipes...")
    batch_index_recipes(client, recipes_to_index, args.batch_size)
    logger.info("Indexing complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Index recipe data into OpenSearch.")
    parser.add_argument("--opensearch_url", type=str, default="http://localhost:9200", help="OpenSearch URL.")
    parser.add_argument("--data_file", type=str, default="data/recipes.parquet", help="Path to the Parquet data file.")
    parser.add_argument("--batch_size", type=int, default=10240, help="Batch size for bulk indexing.")
    sys.exit(main(parser.parse_args()))
