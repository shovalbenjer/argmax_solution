"""
Unit and integration tests for the production data ingestion pipeline.

This module contains a suite of tests for the main data ingestion script
located at `src.data_ingestion.main`. It uses `pytest` for the testing
framework, an in-memory `duckdb` database to avoid file system dependencies,
and `monkeypatch` to redirect the ingestion script to use temporary test data.
"""

from pathlib import Path

import duckdb
import polars as pl
import pytest
from src.data_ingestion.main import ingest_data
from src.data_ingestion.models import NutritionRow


# Use a fixture to set up a temporary directory for test data
@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A pytest fixture to create a temporary directory for raw data files.

    This isolates the tests from the actual project data files, ensuring that
    tests are repeatable and do not have side effects.

    Args:
        tmp_path: The pytest-provided temporary path object.

    Returns:
        A Path object pointing to the created temporary data directory.
    """
    """Create a temporary directory for raw data files."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def db_connection():
    """A pytest fixture to provide an in-memory DuckDB database connection.

    This allows tests to run database operations without creating or modifying
    any files on disk, ensuring tests are fast and clean.

    Returns:
        A connection object to the in-memory DuckDB instance.
    """
    """Create an in-memory DuckDB connection for testing."""
    return duckdb.connect(":memory:")


def test_pydantic_validation_good(tmp_data_dir):
    """Tests that valid data correctly passes Pydantic model validation."""
    csv_path = tmp_data_dir / "good_nutrition.csv"
    good_data = pl.DataFrame(
        {
            "name": ["test_food"],
            "calories": [100.0],
            "protein_g_per_100g": [10.0],
            "carbohydrates_g_per_100g": [20.0],
            "fat_g_per_100g": [5.0],
        }
    )
    good_data.write_csv(csv_path)

    df = pl.read_csv(csv_path)
    # This should not raise an error
    validated_data = NutritionRow(**df.to_dicts()[0])
    assert validated_data.name == "test_food"


def test_pydantic_validation_bad(tmp_data_dir):
    """Tests that invalid data correctly raises a validation error."""
    csv_path = tmp_data_dir / "bad_nutrition.csv"
    bad_data = pl.DataFrame(
        {
            "name": ["bad_food"],
            "calories": ["not_a_float"],  # Invalid type
            "protein_g_per_100g": [10.0],
            # Missing carbohydrates_g
            "fat_g_per_100g": [5.0],
        }
    )
    bad_data.write_csv(csv_path)

    df = pl.read_csv(csv_path)
    with pytest.raises(Exception):  # Catches Pydantic's ValidationError
        NutritionRow(**df.to_dicts()[0])


def test_ingestion_pipeline(db_connection, tmp_data_dir, monkeypatch):
    """Performs an end-to-end test of the data ingestion pipeline.

    This test verifies the entire `ingest_data` function. It works by:
    1.  Creating mock CSV files with known data in a temporary directory.
    2.  Using `monkeypatch` to make the ingestion script use the temporary
        directory as its data source.
    3.  Running the main ingestion function with the in-memory database.
    4.  Asserting that the database table, data, and index were created
        correctly.
    """
    """Tests the full data ingestion pipeline using mock data and an in-memory DB."""
    # Create mock CSV files
    nutrition_csv = tmp_data_dir / "nutrition.csv"
    pl.DataFrame(
        {
            "name": ["test_food_1", "test_food_2"],
            "calories": [100.0, 200.0],
            "protein_g_per_100g": [10.0, 20.0],
            "carbohydrates_g_per_100g": [5.0, 15.0],
            "fat_g_per_100g": [2.0, 4.0],
        }
    ).write_csv(nutrition_csv)

    # Create dummy files for other tables to prevent "file not found" errors
    (tmp_data_dir / "unit_conversion.csv").touch()
    (tmp_data_dir / "vegan_ontology.csv").touch()

    # Use monkeypatch to override the RAW_DATA_DIR constant in the main script
    # This redirects the script to our temporary test directory
    monkeypatch.setattr("src.data_ingestion.main.RAW_DATA_DIR", tmp_data_dir)

    # Run the ingestion process
    ingest_data(db_connection)

    # --- Verification ---
    # 1. Check if the table was created
    tables = db_connection.execute("SHOW TABLES").fetchall()
    assert ("nutrition_facts",) in tables

    # 2. Check if the data was inserted correctly
    result = db_connection.execute("SELECT COUNT(*) FROM nutrition_facts").fetchone()
    assert result[0] == 2

    result_df = db_connection.execute("SELECT * FROM nutrition_facts").pl()
    assert "test_food_1" in result_df["name"].to_list()

    # 3. Check if the index was created
    indices = db_connection.execute("PRAGMA index_list('nutrition_facts');").fetchall()
    # Note: DuckDB index naming might differ, so we check if an index exists at all
    assert len(indices) > 0
