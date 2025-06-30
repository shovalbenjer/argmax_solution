"""Integration tests for the data ingestion pipeline.

This module contains integration tests for the data ingestion process located
in `src.data_ingestion.main`. These tests are designed to be run against the
actual data sources and database, providing a full-stack validation of the
ingestion pipeline.

The tests cover:
    - Creation of the database and tables.
    - Row counts in each table.
    - Absence of null values in critical columns.
    - Pydantic model validation.

Fixtures:
    db_connection: A module-scoped fixture that runs the ingestion pipeline
        once and provides a connection to the test database.

Note:
    These tests are intended to be run in a controlled environment (like a CI/CD
    pipeline) and will create and destroy a test database file.
"""
import pytest
import sqlite3
import polars as pl
from pathlib import Path
import os
import sys
from pydantic import ValidationError

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion.main import main as run_ingestion
from src.data_ingestion.models import NutritionRow, UnitConversionRow, VeganOntologyRow, RecipeRow

# --- Test Configuration ---
# Point to the actual database that the script will create
TEST_DB_PATH = Path(__file__).resolve().parent.parent / "src" / "data" / "knowledge_graph.db"
TABLE_NAMES = ["nutrition_facts", "unit_conversions", "vegan_ontology", "recipes"]

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def db_connection():
    """Pytest fixture to set up and tear down the test database.

    This module-scoped fixture executes the main data ingestion script once,
    populating a test SQLite database. It then yields a connection to this
    database for use in the test cases. After all tests in the module have
    run, the fixture cleans up by deleting the test database file.

    Yields:
        sqlite3.Connection: A connection object to the temporary test database.
    """
    # We need to temporarily override the config to use our test DB
    from src.data_ingestion import main as ingestion_main
    original_db_path = ingestion_main.CONFIG.db_path
    ingestion_main.CONFIG.db_path = TEST_DB_PATH
    
    # Run the main ingestion functions
    ingestion_main.ingest_base_data(TEST_DB_PATH, ingestion_main.CONFIG.data_files)
    ingestion_main.ingest_recipes(TEST_DB_PATH, ingestion_main.CONFIG.recipe_file_path)
    
    # Yield a connection to the newly created test database
    conn = sqlite3.connect(TEST_DB_PATH)
    yield conn
    
    # Teardown: restore original config and clean up the test database
    ingestion_main.CONFIG.db_path = original_db_path
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

# --- Test Cases ---

def test_database_and_tables_created(db_connection):
    """Tests the creation of the database and all required tables.

    Args:
        db_connection (sqlite3.Connection): The fixture providing a connection
            to the test database.
    """
    assert TEST_DB_PATH.exists()
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    for table_name in TABLE_NAMES:
        assert table_name in tables

def test_data_ingestion_counts(db_connection):
    """Tests that a reasonable number of rows were ingested into each table.

    This acts as a sanity check to ensure that the data sources were read
    correctly and that the ingestion process did not fail silently.

    Args:
        db_connection (sqlite3.Connection): The fixture providing a connection
            to the test database.
    """
    # These counts can be made more precise if the source data is static.
    # For now, we'll check for a reasonable number of rows.
    counts = {}
    for table_name in TABLE_NAMES:
        df = pl.read_database(f"SELECT * FROM {table_name}", db_connection)
        counts[table_name] = len(df)

    assert counts["nutrition_facts"] > 3000
    assert counts["unit_conversions"] > 10
    assert counts["vegan_ontology"] > 200
    assert counts["recipes"] > 50

def test_no_nulls_in_critical_columns(db_connection):
    """Tests for the absence of NULL values in critical columns.

    This test verifies that primary keys and other essential columns across
    multiple tables do not contain any NULL values, which would indicate
    data quality issues.

    Args:
        db_connection (sqlite3.Connection): The fixture providing a connection
            to the test database.
    """
    # Nutrition table
    df_nutrition = pl.read_database("SELECT * FROM nutrition_facts", db_connection)
    assert df_nutrition.filter(pl.col("name").is_null()).is_empty()

    # Unit conversions table
    df_units = pl.read_database("SELECT * FROM unit_conversions", db_connection)
    assert df_units.filter(pl.col("from_unit").is_null()).is_empty()
    assert df_units.filter(pl.col("to_unit").is_null()).is_empty()
    assert df_units.filter(pl.col("multiplier").is_null()).is_empty()

    # Vegan ontology table
    df_vegan = pl.read_database("SELECT * FROM vegan_ontology", db_connection)
    assert df_vegan.filter(pl.col("term").is_null()).is_empty()

    # Recipes table
    df_recipes = pl.read_database("SELECT * FROM recipes", db_connection)
    assert df_recipes.filter(pl.col("id").is_null()).is_empty()
    assert df_recipes.filter(pl.col("ingredients").is_null()).is_empty()

def test_pydantic_model_validation():
    """Tests the Pydantic models' validation logic.

    This test is a unit test that does not depend on the database. It ensures
    that the Pydantic models correctly validate data, raising `ValidationError`
    for invalid inputs and successfully creating models for valid inputs.
    """
    # Example of a valid row
    valid_nutrition = {"name": "Test Food", "serving_size": "100g", "calories": 100.0, "protein_g": 10.0, "fat_g": 5.0, "carbohydrates_g": 20.0}
    assert NutritionRow(**valid_nutrition)

    # Example of an invalid row
    invalid_nutrition = {"name": "Bad Food", "serving_size": "100g", "calories": "not-a-number"}
    with pytest.raises(ValidationError):
        NutritionRow(**invalid_nutrition)

@pytest.mark.skip(reason="This test is a placeholder and is not needed.")
def test_placeholder_data_exists(db_connection):
    """Placeholder test to demonstrate checking for data existence.

    Note:
        This test is currently skipped as it is a placeholder.

    Args:
        db_connection (sqlite3.Connection): The fixture providing a connection
            to the test database.
    """
    # This test is a placeholder. A real test would insert known data
    # and then select it back to ensure it was written correctly.
    cursor = db_connection.cursor()
    # We can't insert if the table is populated with real data,
    # so we'll just check if the table is not empty.
    df = pl.read_database("SELECT * FROM vegan_ontology LIMIT 1", db_connection)
    assert len(df) > 0

# --- Ingestion Logic Tests ---

# In a full test suite, we would import functions from main.py and test them.
# This requires refactoring main.py to make its functions more modular and testable.
# For example, the parsing logic inside ingest_recipes could be its own function.

# from src.data_ingestion.main import parse_ingredient_string # (if it existed)
#
# def test_ingredient_parser():
#     """Tests the ingredient parsing logic."""
#     test_string = "2 1/2 cups all-purpose flour"
#     parsed = parse_ingredient_string(test_string) # Hypothetical function
#     assert parsed.name == "all-purpose flour"
#     assert parsed.amount.quantity == 2.5
#     assert parsed.amount.unit == "cup"

# This demonstrates the structure. Since the main script is not yet refactored
# for testability, these tests are foundational placeholders. 