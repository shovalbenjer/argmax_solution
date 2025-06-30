"""
Data Manager for the Dietary Classification System.

This module is responsible for loading all necessary data from the CSV files
into memory for fast and efficient lookups by the classifier. It uses a
singleton pattern to ensure data is loaded only once.
"""
from pathlib import Path
import pandas as pd
from loguru import logger

class DataManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Loads all required datasets into memory."""
        self.base_dir = Path(__file__).resolve().parent
        self.raw_data_dir = self.base_dir / "raw_data"
        
        self.nutrition_df = self._load_nutrition_data()
        self.vegan_ontology_set = self._load_vegan_ontology()
        
    def _load_nutrition_data(self) -> pd.DataFrame:
        """Loads and processes the nutrition data."""
        path = self.raw_data_dir / "nutrition.csv"
        logger.info(f"Loading nutrition data from {path}...")
        try:
            # The CSV has no header, so we assign column names manually.
            # We only need the name (col 1) and carbohydrates (col 21).
            col_names = [f"col_{i}" for i in range(22)]
            df = pd.read_csv(path, header=None, names=col_names)
            
            # Select and rename the columns we need
            df = df[['col_1', 'col_21']].rename(columns={
                'col_1': 'name',
                'col_21': 'carbohydrates_g'
            })

            # Clean and set the index
            df['name_normalized'] = df['name'].str.lower().str.strip()
            df = df.set_index('name_normalized')
            df['carbohydrates_g'] = pd.to_numeric(df['carbohydrates_g'], errors='coerce').fillna(100)
            
            logger.success("Nutrition data loaded successfully.")
            return df
        except Exception as e:
            logger.critical(f"Failed to load nutrition data: {e}")
            return pd.DataFrame() # Return empty dataframe on failure

    def _load_vegan_ontology(self) -> set:
        """Loads the vegan ontology into a set for fast lookups."""
        path = self.raw_data_dir / "vegan_ontology.csv"
        logger.info(f"Loading vegan ontology from {path}...")
        try:
            df = pd.read_csv(path)
            # We are interested in terms that are explicitly NOT vegan
            non_vegan_terms = df[df['is_vegan'] == False]['term'].str.lower().str.strip()
            logger.success("Vegan ontology loaded successfully.")
            return set(non_vegan_terms)
        except Exception as e:
            logger.critical(f"Failed to load vegan ontology: {e}")
            return set() # Return empty set on failure

# Create a single, globally accessible instance
data_manager = DataManager() 