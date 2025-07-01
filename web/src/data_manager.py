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
    """A singleton data manager for loading and caching nutritional and dietary data.

    Detailed Description:
        - This class implements the singleton pattern to ensure that expensive data loading
          operations are performed only once during the application lifecycle.
        - It loads nutrition facts and vegan ontology data from CSV files into optimized
          in-memory data structures (pandas DataFrame and Python set) for fast lookups.
        - The singleton pattern is crucial here because data loading can take several seconds
          and should not be repeated for each classifier call.

    Libraries Used:
        - pandas: For loading and processing the nutrition CSV data. Its vectorized operations
          and indexing capabilities make it superior to native Python for tabular data manipulation.
        - pathlib: For robust, cross-platform file path handling.
        - loguru: For structured logging with better formatting than the standard logging module.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes the data manager by loading all required datasets.

        Detailed Description:
            - This method is called once when the singleton instance is first created.
            - It sets up the file paths and calls the individual data loading methods.
            - The initialization is separated from `__new__` to keep the singleton logic clean.
        """
        self.base_dir = Path(__file__).resolve().parent
        self.raw_data_dir = self.base_dir / "raw_data"
        
        self.nutrition_df = self._load_nutrition_data()
        self.vegan_ontology_set = self._load_vegan_ontology()
        
    def _load_nutrition_data(self) -> pd.DataFrame:
        """Loads and preprocesses nutrition data for fast ingredient lookups.

        Detailed Description:
            - This method loads the raw nutrition CSV file, which contains no headers.
            - It selects only the required columns (ingredient name and carbohydrates).
            - It normalizes ingredient names to lowercase and strips whitespace for consistent matching.
            - It sets the normalized name as the DataFrame index for O(1) lookup performance.
            - It handles missing carbohydrate values by filling them with a conservative default (100g).

        Returns:
            - pd.DataFrame: A DataFrame indexed by normalized ingredient names, containing
              carbohydrate information, or an empty DataFrame if loading fails.

        Libraries Used:
            - pandas: For CSV loading, data cleaning, and indexing operations.
        """
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
        """Loads the vegan ontology into a set for O(1) ingredient classification.

        Detailed Description:
            - This method loads the vegan ontology CSV file containing ingredient classifications.
            - It filters for ingredients that are explicitly marked as non-vegan (is_vegan == False).
            - It normalizes the terms to lowercase for consistent matching.
            - It returns a Python set for O(1) membership testing during classification.

        Returns:
            - set: A set of normalized non-vegan ingredient terms, or an empty set if loading fails.

        Libraries Used:
            - pandas: For CSV loading and filtering operations.
            - Python set: For O(1) membership testing, which is more efficient than list lookup.
        """
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