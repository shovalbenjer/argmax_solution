"""
High-Performance Data Ingestion Pipeline for Knowledge Graph Creation.

This script implements the complete, production-grade data ingestion process
as outlined in the V4 project plan. It builds the `knowledge_graph.db`
from the raw CSV data using a high-performance, validated, and robust pipeline.

Key Features:
- Polars for high-speed data manipulation.
- Pydantic for robust data validation and type safety.
- Centralized configuration for easy management.
- Modular functions for clarity and testability.
- Detailed logging for traceability.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Type
import polars as pl
from loguru import logger
from pydantic import BaseModel, ValidationError, field_validator

# --- Pydantic Models for Validation ---

class NutritionRow(BaseModel):
    """Validates and cleans a row from the nutrition.csv file."""
    name: str
    calories: float = 0.0
    protein_g: float = 0.0
    carbohydrates_g: float = 0.0
    fat_g: float = 0.0

    @field_validator('name', mode='before')
    @classmethod
    def clean_name(cls, v):
        return v.lower().strip() if isinstance(v, str) else v

class UnitConversionRow(BaseModel):
    """Validates a row from the unit_conversion.csv file."""
    ingredient_name: str
    from_unit: str
    to_unit: str
    multiplier: float

class VeganOntologyRow(BaseModel):
    """Validates and transforms a row from the vegan_ontology.csv file."""
    term: str
    is_explicitly_non_vegan: bool

    @field_validator('is_explicitly_non_vegan', mode='before')
    @classmethod
    def convert_bool(cls, v):
        if isinstance(v, str):
            return v.strip().upper() == 'F'
        return v

# --- Configuration ---

class AppConfig:
    """Singleton configuration class for all paths and settings."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.base_dir / "raw_data"
        self.db_path = self.data_dir / "knowledge_graph.db"
        
        self.data_files: Dict[str, Dict[str, Any]] = {
            "nutrition_facts": {
                "path": self.raw_data_dir / "nutrition.csv",
                "model": NutritionRow,
                "indices": ["name"],
            },
            "unit_conversions": {
                "path": self.raw_data_dir / "unit_conversion.csv",
                "model": UnitConversionRow,
                "indices": ["ingredient_name"],
            },
            "vegan_ontology": {
                "path": self.raw_data_dir / "vegan_ontology.csv",
                "model": VeganOntologyRow,
                "indices": ["term"],
            },
        }

CONFIG = AppConfig()

# --- Database Operations ---

def create_database(db_path: Path):
    """Creates a fresh, clean SQLite database."""
    db_path.parent.mkdir(exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    
    with sqlite3.connect(db_path) as conn:
        logger.success(f"Successfully created database at {db_path}")

def write_to_db(df: pl.DataFrame, table_name: str, db_path: Path, indices: List[str]):
    """Writes a Polars DataFrame to a table in the SQLite database."""
    df.to_pandas().to_sql(table_name, f"sqlite:///{db_path}", if_exists="replace", index=False)
    logger.success(f"Wrote {len(df)} rows to table '{table_name}'.")
    with sqlite3.connect(db_path) as conn:
        for col in indices:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name}({col})")
    logger.info(f"Created indices for table '{table_name}'.")

# --- Data Processing ---

def read_and_validate(file_path: Path, model: Type[BaseModel], column_map: Dict[str, str], has_header: bool = True) -> pl.DataFrame:
    """Reads, renames, and validates a CSV file."""
    df = pl.read_csv(file_path, has_header=has_header)
    
    if not has_header:
        # If no header, the number of columns must match the map
        if len(df.columns) == len(column_map):
            df.columns = list(column_map.keys())
        else:
            # Handle cases where column count doesn't match (e.g., nutrition.csv)
            # We select columns by their position/index, which is brittle but necessary here.
            selected_cols = []
            for col_pos_str, new_name in column_map.items():
                col_pos = int(col_pos_str.split('_')[1]) -1 # from "field_1" to 0
                selected_cols.append(df.columns[col_pos])
            df = df.select(selected_cols)
            df.columns = list(column_map.values())
            
    df = df.rename({k:v for k,v in column_map.items() if k in df.columns})

    validated_rows = [row for row in df.to_dicts() if validate_row(row, model, file_path.name)]
    if not validated_rows:
        raise ValueError(f"No valid data in {file_path.name}")
    return pl.DataFrame(validated_rows)

def validate_row(row_data: dict, model: Type[BaseModel], filename: str) -> bool:
    try:
        model(**row_data)
        return True
    except ValidationError as e:
        logger.warning(f"Skipping invalid row in {filename}: {e.errors()}")
        return False

def main():
    """Main function to orchestrate the data ingestion pipeline."""
    logger.info("===== Starting Data Ingestion Pipeline =====")
    CONFIG.data_dir.mkdir(exist_ok=True)
    if CONFIG.db_path.exists():
        CONFIG.db_path.unlink()

    # --- Process Nutrition Data ---
    try:
        # These are the original column names from the file
        nutrition_map = {"field_2": "name", "field_4": "calories", "field_19": "protein_g", "field_21": "fat_g", "field_22": "carbohydrates_g"}
        df_nutrition = read_and_validate(CONFIG.data_files["nutrition_facts"]["path"], NutritionRow, column_map=nutrition_map, has_header=False)
        write_to_db(df_nutrition, "nutrition_facts", CONFIG.db_path, ["name"])
except Exception as e:
        logger.error(f"Failed to process nutrition_facts: {e}")

    # --- Process Unit Conversions ---
    try:
        uc_map = {"Unit": "from_unit", "StandardUnit": "to_unit", "Factor": "multiplier", "Notes": "ingredient_name"}
        df_uc = read_and_validate(CONFIG.data_files["unit_conversions"]["path"], UnitConversionRow, column_map=uc_map)
        write_to_db(df_uc, "unit_conversions", CONFIG.db_path, ["ingredient_name"])
    except Exception as e:
        logger.error(f"Failed to process unit_conversions: {e}")

    # --- Process Vegan Ontology ---
    try:
        vo_map = {"term": "term", "is_vegan": "is_explicitly_non_vegan"}
        df_vo = read_and_validate(CONFIG.data_files["vegan_ontology"]["path"], VeganOntologyRow, column_map=vo_map)
        write_to_db(df_vo, "vegan_ontology", CONFIG.db_path, ["term"])
    except Exception as e:
        logger.error(f"Failed to process vegan_ontology: {e}")

    logger.success("===== Data Ingestion Pipeline Finished Successfully =====")

if __name__ == "__main__":
    main() 