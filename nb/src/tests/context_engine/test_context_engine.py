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
    """Creates an in-memory DuckDB database fixture for isolated testing.

    Detailed Description:
        - This pytest fixture sets up a clean, in-memory database for each test.
        - It creates the required table schema and populates it with known test data.
        - This approach ensures tests are deterministic and don't depend on external
          database state or files.

    Returns:
        - duckdb.Connection: A connection to the in-memory test database.

    Libraries Used:
        - duckdb: An in-memory analytical database that's faster than SQLite for test scenarios.
          It provides SQL compatibility while avoiding file I/O overhead.
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
    """Tests the complete ingredient processing pipeline with mocked dependencies.

    Detailed Description:
        - This integration test validates the `get_context_for_raw_ingredient` function.
        - It uses pytest's `monkeypatch` to replace external dependencies:
          1. The RSL-SQL handler is mocked to return predictable SQL queries.
          2. The database connection is replaced with the in-memory test database.
        - This approach tests the orchestration logic while isolating it from
          external services and unpredictable data.

    Parameters:
        - monkeypatch (pytest.MonkeyPatch): Pytest's dependency injection mechanism.
        - mock_db_connection (duckdb.Connection): The test database fixture.

    Libraries Used:
        - pytest: The testing framework, chosen for its powerful fixture system and
          monkeypatching capabilities over unittest's more verbose mocking approach.
        - unittest.mock.MagicMock: For creating mock objects that simulate external dependencies.
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