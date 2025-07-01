"""
Database Handler for accessing the project's Knowledge Graph.

This module provides a simplified, high-level interface for querying
the SQLite database (`knowledge_graph.db`). It abstracts away the direct
SQL queries and connection management, offering clean functions to retrieve
data for EDA, testing, or other analyses.

It is built on top of the centralized db utility to ensure consistent
database connection handling across the application.
"""
import pandas as pd
from loguru import logger

# Import the engine from our new centralized utility
from utils.db import get_engine

class DatabaseHandler:
    def __init__(self):
        """Initializes the handler with the shared database engine."""
        self.engine = get_engine()
        logger.info(f"DatabaseHandler initialized with engine: {self.engine.url}")

    def get_all_nutrition_facts(self) -> pd.DataFrame:
        """Returns all records from the nutrition_facts table."""
        try:
            return pd.read_sql("SELECT * FROM nutrition_facts", self.engine)
        except Exception as e:
            logger.error(f"Failed to get all nutrition facts: {e}")
            return pd.DataFrame()

    def get_all_unit_conversions(self) -> pd.DataFrame:
        """Returns all records from the unit_conversions table."""
        try:
            return pd.read_sql("SELECT * FROM unit_conversions", self.engine)
        except Exception as e:
            logger.error(f"Failed to get all unit conversions: {e}")
            return pd.DataFrame()

    def get_all_vegan_ontology(self) -> pd.DataFrame:
        """Returns all records from the vegan_ontology table."""
        try:
            return pd.read_sql("SELECT * FROM vegan_ontology", self.engine)
        except Exception as e:
            logger.error(f"Failed to get all vegan ontology: {e}")
            return pd.DataFrame()

# A global instance for easy access from other scripts, mimicking the original design.
db_handler = DatabaseHandler() 