"""
Unit tests for the experimental context engine.

This module contains tests for the advanced, but currently incomplete, context
engine. The tests use `pytest` for the test framework, `monkeypatch` for
isolating components, and an in-memory `duckdb` instance to simulate the
knowledge graph database.
"""
import pytest
import duckdb
from unittest.mock import MagicMock

# Important: This assumes the .pyx has been compiled.
from src.context_engine.fast_processor import process_recipe_batch
from src.context_engine.main import get_context_for_raw_ingredient
from src.context_engine.models import RSLSQLResponse, NormalizedContext

@pytest.fixture
def mock_db_connection():
    """Creates a mock, in-memory DuckDB database for testing.

    This pytest fixture sets up an in-memory database and populates it with
    sample data for `nutrition_facts` and `unit_conversions`. This allows
    tests to run against a predictable database state without needing the
    actual SQLite file.

    Returns:
        A connection object to the in-memory DuckDB instance.
    """
    """Create a mock in-memory DB and populate it with test data."""
    conn = duckdb.connect(':memory:')
    conn.execute("""
        CREATE TABLE nutrition_facts (
            name VARCHAR, 
            carbohydrates_g_per_100g DOUBLE,
            protein_g_per_100g DOUBLE,
            fat_g_per_100g DOUBLE
        );
    """)
    conn.execute("""
        INSERT INTO nutrition_facts VALUES 
        ('all-purpose flour', 76.3, 10.3, 1.0);
    """)
    conn.execute("""
        CREATE TABLE unit_conversions (
            ingredient_name VARCHAR, 
            from_unit VARCHAR,
            to_unit VARCHAR,
            multiplier DOUBLE
        );
    """)
    conn.execute("""
        INSERT INTO unit_conversions VALUES 
        ('all-purpose flour', 'cup', 'gram', 125.0);
    """)
    return conn

def test_single_ingredient_processing(monkeypatch, mock_db_connection):
    """Tests the end-to-end orchestration for a single ingredient.

    This test validates the `get_context_for_raw_ingredient` function.
    It uses `monkeypatch` to:
    1.  Replace the (mocked) RSL-SQL handler with a MagicMock that returns a
        predictable SQL query.
    2.  Replace the database connection function with one that returns the
        in-memory `mock_db_connection`.
    
    This effectively tests the orchestration logic while isolating it from the
    actual text-to-SQL model and the real database.
    """
    """Tests the full pipeline for a single ingredient, mocking RSL-SQL."""

    # 1. Mock the RSL-SQL response
    mock_rsl_response = RSLSQLResponse(
        generated_sql="SELECT * FROM nutrition_facts WHERE name = 'all-purpose flour'",
        vegan_candidates=[],
        confidence=0.99
    )
    # Patch the handler to return our mock response
    monkeypatch.setattr(
        "src.context_engine.main.mock_rsl_sql_handler",
        MagicMock(return_value=mock_rsl_response)
    )
    # Patch the db connection function to return our mock db
    monkeypatch.setattr(
        "src.context_engine.main.get_db_connection",
        MagicMock(return_value=mock_db_connection)
    )
    
    # 2. Run the main processing function
    result = get_context_for_raw_ingredient("2 cups all-purpose flour")

    # 3. Assertions
    assert isinstance(result, NormalizedContext)
    assert result.parsed_name == "all purpose flour" # Note: parser normalizes
    # We didn't mock the unit conversion SQL, so this part won't be fully tested here
    # A more complex test could mock the SQL execution itself.
    assert result.audit_trail["rsl_confidence"] == 0.99

def test_cython_batch_processing(monkeypatch):
    """Tests the high-level Cython batch processing function.

    This test validates that the `process_recipe_batch` Cython function correctly
    iterates through a list of ingredients and calls the underlying Python
    processing function (`get_context_for_raw_ingredient`) for each one.

    It uses `monkeypatch` to replace the Python function with a mock, ensuring
    that this test focuses only on the batching logic of the Cython code, not
    the underlying processing.
    """
    """Tests the Cython batch processor."""
    
    # 1. Mock the main Python function that the Cython code calls
    mock_context = NormalizedContext(
        raw_ingredient="test",
        parsed_name="test",
        normalized_nutrition={},
        vegan_context=[],
        audit_trail={}
    )
    monkeypatch.setattr(
        "src.context_engine.main.get_context_for_raw_ingredient",
        MagicMock(return_value=mock_context)
    )
    
    # 2. Run the Cython batch function
    ingredients = ["2 cups flour", "1 tsp salt"]
    results = process_recipe_batch(ingredients)
    
    # 3. Assertions
    assert len(results) == 2
    assert results[0] == mock_context
    assert results[1] == mock_context 