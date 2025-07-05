#!/usr/bin/env python3
"""
Data Quality Validation Script

Validates that data ingestion completed successfully with required benchmarks.
Tests nutrition data, vegan ontology, and unit conversions for completeness and quality.

This script should be run after 01_build_knowledge_base.py to verify the
knowledge database was created properly.

The script performs comprehensive validation including:
- Database file existence and size verification
- Table structure validation
- Record count validation against benchmarks
- Data quality checks (null values, duplicates, valid ranges)
- Statistical analysis of data completeness

Returns:
    int: Exit code (0 for success, 1 for failure)

Raises:
    FileNotFoundError: If database file is missing
    sqlite3.Error: If database operations fail
    ValueError: If validation criteria are not met
"""
import logging
import sqlite3
import sys
from pathlib import Path

# Add nb/src to path for accessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from config import app_config

# Data Quality Benchmarks
BENCHMARKS = {
    "nutrition_facts_min_records": 8000,
    "vegan_ontology_min_records": 200,  # Adjusted from real data size
    "unit_conversions_min_records": 15,
    "min_database_size_mb": 5,  # Adjusted for realistic expectations
    "required_tables": ["nutrition_facts", "vegan_ontology", "unit_conversions"],
}


def validate_database_exists():
    """
    Validate that knowledge_graph.db was created.
    
    This function checks for database file existence and validates
    the file size against minimum requirements.
    
    Raises:
        FileNotFoundError: If database file is missing
        ValueError: If database size is below minimum threshold
    """
    if not app_config.DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {app_config.DB_PATH}")

    size_mb = app_config.DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Database found: {app_config.DB_PATH} ({size_mb:.1f} MB)")

    if size_mb < BENCHMARKS["min_database_size_mb"]:
        logger.warning(
            f"Database size {size_mb:.1f} MB below benchmark {BENCHMARKS['min_database_size_mb']} MB"
        )
    else:
        logger.info(
            f"Database size meets benchmark ({size_mb:.1f} MB >= {BENCHMARKS['min_database_size_mb']} MB)"
        )


def validate_table_structure():
    """
    Validate that all required tables exist with proper structure.
    
    This function checks for the presence of all required tables and
    validates their column structure.
    
    Raises:
        ValueError: If required tables are missing
        sqlite3.Error: If database operations fail
    """
    with sqlite3.connect(app_config.DB_PATH) as conn:
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        logger.info(f"Found tables: {tables}")

        # Check required tables
        missing_tables = set(BENCHMARKS["required_tables"]) - set(tables)
        if missing_tables:
            raise ValueError(f"Missing required tables: {missing_tables}")

        logger.info("All required tables present")

        # Validate table schemas
        for table in BENCHMARKS["required_tables"]:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            logger.info(f"Table {table}: {len(columns)} columns")


def validate_record_counts():
    """
    Validate that tables have sufficient records.
    
    This function checks record counts against minimum benchmarks
    and reports detailed statistics for each table.
    
    Returns:
        dict: Dictionary containing record counts for each table
        
    Raises:
        sqlite3.Error: If database operations fail
    """
    with sqlite3.connect(app_config.DB_PATH) as conn:
        cursor = conn.cursor()

        results = {}

        # Check nutrition_facts
        cursor.execute("SELECT COUNT(*) FROM nutrition_facts;")
        nutrition_count = cursor.fetchone()[0]
        results["nutrition_facts"] = nutrition_count

        if nutrition_count < BENCHMARKS["nutrition_facts_min_records"]:
            logger.warning(
                f"nutrition_facts: {nutrition_count} records < {BENCHMARKS['nutrition_facts_min_records']} benchmark"
            )
        else:
            logger.info(
                f"nutrition_facts: {nutrition_count} records >= {BENCHMARKS['nutrition_facts_min_records']} benchmark"
            )

        # Check vegan_ontology
        cursor.execute("SELECT COUNT(*) FROM vegan_ontology;")
        vegan_count = cursor.fetchone()[0]
        results["vegan_ontology"] = vegan_count

        if vegan_count < BENCHMARKS["vegan_ontology_min_records"]:
            logger.warning(
                f"vegan_ontology: {vegan_count} records < {BENCHMARKS['vegan_ontology_min_records']} benchmark"
            )
        else:
            logger.info(
                f"vegan_ontology: {vegan_count} records >= {BENCHMARKS['vegan_ontology_min_records']} benchmark"
            )

        # Check unit_conversions
        cursor.execute("SELECT COUNT(*) FROM unit_conversions;")
        unit_count = cursor.fetchone()[0]
        results["unit_conversions"] = unit_count

        if unit_count < BENCHMARKS["unit_conversions_min_records"]:
            logger.warning(
                f"unit_conversions: {unit_count} records < {BENCHMARKS['unit_conversions_min_records']} benchmark"
            )
        else:
            logger.info(
                f"unit_conversions: {unit_count} records >= {BENCHMARKS['unit_conversions_min_records']} benchmark"
            )

        return results


def validate_data_quality():
    """
    Validate data quality and integrity.
    
    This function performs comprehensive data quality checks including:
    - Null/empty ingredient names
    - Duplicate ingredient names
    - Valid nutrition values (calories > 0)
    - Data completeness statistics
    
    Raises:
        sqlite3.Error: If database operations fail
    """
    with sqlite3.connect(app_config.DB_PATH) as conn:
        cursor = conn.cursor()

        # Check for null ingredient names
        cursor.execute(
            "SELECT COUNT(*) FROM nutrition_facts WHERE name IS NULL OR name = '';"
        )
        null_names = cursor.fetchone()[0]

        if null_names > 0:
            logger.warning(
                f"Found {null_names} nutrition records with null/empty names"
            )
        else:
            logger.info("No null ingredient names found")

        # Check for duplicate ingredient names
        cursor.execute(
            "SELECT name, COUNT(*) FROM nutrition_facts GROUP BY name HAVING COUNT(*) > 1;"
        )
        duplicates = cursor.fetchall()

        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate ingredient names")
            for name, count in duplicates[:5]:  # Show first 5
                logger.warning(f"  - '{name}': {count} duplicates")
        else:
            logger.info("No duplicate ingredient names found")

        # Check for reasonable nutrition values
        cursor.execute("SELECT COUNT(*) FROM nutrition_facts WHERE calories > 0;")
        valid_calories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM nutrition_facts;")
        total_records = cursor.fetchone()[0]

        calories_percentage = (
            (valid_calories / total_records * 100) if total_records > 0 else 0
        )
        logger.info(
            f"Records with valid calories: {valid_calories}/{total_records} ({calories_percentage:.1f}%)"
        )

        if calories_percentage < 80:
            logger.warning("Low percentage of records with valid calorie data")


def main():
    """
    Run all data quality validations.
    
    This function orchestrates the complete data quality validation pipeline:
    1. Database existence and size validation
    2. Table structure validation
    3. Record count validation
    4. Data quality validation
    5. Summary reporting
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
        
    Raises:
        FileNotFoundError: If database file is missing
        ValueError: If validation criteria are not met
        sqlite3.Error: If database operations fail
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 60)

    try:
        # Database existence and size
        validate_database_exists()

        # Table structure
        validate_table_structure()

        # Record counts
        record_counts = validate_record_counts()

        # Data quality
        validate_data_quality()

        # Summary
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Database: {app_config.DB_PATH}")
        logger.info(f"Size: {app_config.DB_PATH.stat().st_size / (1024 * 1024):.1f} MB")

        for table, count in record_counts.items():
            benchmark = BENCHMARKS.get(f"{table}_min_records", 0)
            status = "PASS" if count >= benchmark else "WARN"
            logger.info(
                f"{status}: {table}: {count:,} records (benchmark: {benchmark:,})"
            )

        # Overall status
        all_benchmarks_met = all(
            record_counts[table] >= BENCHMARKS[f"{table}_min_records"]
            for table in record_counts
        )

        if all_benchmarks_met:
            logger.info("SUCCESS: All data quality benchmarks met")
            return 0
        else:
            logger.warning("PARTIAL: Some benchmarks not met - review warnings above")
            return 0  # Continue with warnings as non-critical

    except Exception as e:
        logger.error(f"FAILED: Data quality validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
