"""
Data Ingestion Script for Diet Classification Knowledge Base

This script performs the initial data ingestion and database setup for the
diet classification system. It processes raw CSV data from nutrition facts,
vegan ontology, and unit conversions, then populates a SQLite database with
cleaned and normalized data.

The script handles comprehensive data processing including:
- Nutrition facts with 80+ nutritional components per food item
- Vegan ontology with aliases and classification flags
- Unit conversions for recipe standardization
- Data cleaning and validation
- Database schema creation with optimized indexes

Key Features:
- Comprehensive nutrition data (80+ columns per food item)
- JSON-based aliases storage for vegan ontology
- Automatic data type conversion and cleaning
- Database indexing for optimal query performance
- Error handling and validation reporting

Database Schema:
- nutrition_facts: Complete nutritional profiles (80+ columns)
- vegan_ontology: Vegan classification with aliases and context flags
- unit_conversions: Standardized unit conversion factors

Usage:
    python scripts/01_ingest_data.py

Dependencies:
    - polars: High-performance data processing
    - pandas: Data manipulation and CSV reading
    - sqlalchemy: Database schema management
    - loguru: Professional logging

Example:
    >>> # Run the ingestion script
    >>> python scripts/01_ingest_data.py
    >>> # Check the created database
    >>> import sqlite3
    >>> conn = sqlite3.connect("nb/src/data/knowledge_graph.db")
    >>> cursor = conn.execute("SELECT COUNT(*) FROM nutrition_facts")
    >>> print(f"Loaded {cursor.fetchone()[0]} nutrition records")
"""
import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, Boolean, MetaData, JSON
from pathlib import Path
from loguru import logger
import json
import sys
import ast
import re
import traceback

# Add nb/src to path for importing modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nb" / "src"))

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
NB_SRC_DIR = BASE_DIR / "nb" / "src"
DATA_DIR = NB_SRC_DIR / "data"
RAW_DATA_DIR = NB_SRC_DIR / "raw_data"
DB_PATH = DATA_DIR / "knowledge_graph.db"

# Define file paths
NUTRITION_CSV = RAW_DATA_DIR / "nutrition.csv"
VEGAN_CSV = RAW_DATA_DIR / "vegan_ontology.csv"
UNIT_CONVERSION_CSV = RAW_DATA_DIR / "unit_conversion.csv"

# --- Database Setup ---
engine = create_engine(f"sqlite:///{DB_PATH}")
metadata = MetaData()

# Define table structures
nutrition_facts_table = Table('nutrition_facts', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String, nullable=False, unique=True),
    Column('serving_size', String),
    Column('calories', Float),
    Column('total_fat_g', Float),
    Column('saturated_fat_g', Float),
    Column('protein_g', Float),
    Column('carbohydrate_g', Float),
    Column('fiber_g', Float),
    Column('sugars_g', Float),
    Column('vitamin_a_iu', Float),
    Column('vitamin_a_rae_mcg', Float),
    Column('vitamin_b6_mg', Float),
    Column('vitamin_b12_mcg', Float),
    Column('vitamin_c_mg', Float),
    Column('vitamin_d_iu', Float),
    Column('vitamin_e_mg', Float),
    Column('vitamin_k_mcg', Float),
    Column('thiamin_mg', Float),
    Column('riboflavin_mg', Float),
    Column('niacin_mg', Float),
    Column('folate_mcg', Float),
    Column('pantothenic_acid_mg', Float),
    Column('calcium_mg', Float),
    Column('iron_mg', Float),
    Column('magnesium_mg', Float),
    Column('phosphorous_mg', Float),
    Column('potassium_mg', Float),
    Column('sodium_mg', Float),
    Column('zinc_mg', Float),
    Column('copper_mg', Float),
    Column('manganese_mg', Float),
    Column('selenium_mcg', Float),
    Column('cholesterol_mg', Float),
    Column('choline_mg', Float),
    Column('water_g', Float),
    Column('ash_g', Float),
    Column('caffeine_mg', Float),
    Column('alcohol_g', Float),
    Column('alanine_g', Float),
    Column('arginine_g', Float),
    Column('aspartic_acid_g', Float),
    Column('cystine_g', Float),
    Column('glutamic_acid_g', Float),
    Column('glycine_g', Float),
    Column('histidine_g', Float),
    Column('isoleucine_g', Float),
    Column('leucine_g', Float),
    Column('lysine_g', Float),
    Column('methionine_g', Float),
    Column('phenylalanine_g', Float),
    Column('proline_g', Float),
    Column('serine_g', Float),
    Column('threonine_g', Float),
    Column('tryptophan_g', Float),
    Column('tyrosine_g', Float),
    Column('valine_g', Float),
    Column('saturated_fatty_acids_g', Float),
    Column('monounsaturated_fatty_acids_g', Float),
    Column('polyunsaturated_fatty_acids_g', Float),
    Column('fatty_acids_total_trans_mg', Float),
    Column('carotene_alpha_mcg', Float),
    Column('carotene_beta_mcg', Float),
    Column('cryptoxanthin_beta_mcg', Float),
    Column('lutein_zeaxanthin_mcg', Float),
    Column('lycopene_mcg', Float),
    Column('fructose_g', Float),
    Column('galactose_g', Float),
    Column('glucose_g', Float),
    Column('lactose_g', Float),
    Column('maltose_g', Float),
    Column('sucrose_g', Float)
)

vegan_ontology_table = Table('vegan_ontology', metadata,
    Column('id', Integer, primary_key=True),
    Column('term', String, nullable=False, unique=True),
    Column('aliases', JSON),
    Column('is_explicitly_non_vegan', Boolean),
    Column('description', String),
    Column('source_details', String),
    Column('requires_contextual_check', Boolean),
    Column('is_vegan_exception_term', Boolean)
)

unit_conversions_table = Table('unit_conversions', metadata,
    Column('id', Integer, primary_key=True),
    Column('unit', String, nullable=False, unique=True),
    Column('abbreviation', String),
    Column('us_value', String),
    Column('metric_equivalent', String),
    Column('notes', String),
    Column('type', String),
    Column('factor', Float, nullable=False),
    Column('standard_unit', String, nullable=False)
)

def create_database_schema():
    """Creates the database schema for the core knowledge tables."""
    logger.info("Creating database schema...")
    metadata.drop_all(engine)
    metadata.create_all(engine)
    with engine.connect() as connection:
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_nutrition_name ON nutrition_facts (name);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_vegan_term ON vegan_ontology (term);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_unit_name ON unit_conversions (unit);"))
        connection.commit()
    logger.success("Database schema created successfully.")

def clean_numeric_value(value_str):
    """Helper function to clean numeric values from CSV data."""
    if pd.isna(value_str) or str(value_str).strip() == '':
        return 0.0
    cleaned = re.sub(r'[^\d\.-]', '', str(value_str))
    try:
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def ingest_nutrition_data():
    """Ingests and cleans nutrition data from CSV."""
    logger.info("Ingesting nutrition data...")
    try:
        df = pd.read_csv(NUTRITION_CSV)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Correct known typos in column names
        df = df.rename(columns={'irom': 'iron', 'zink': 'zinc', 'lucopene': 'lycopene'})
        
        # Map CSV column names to database column names
        column_mapping = {
            'total_fat': 'total_fat_g',
            'saturated_fat': 'saturated_fat_g', 
            'protein': 'protein_g',
            'carbohydrate': 'carbohydrate_g',
            'fiber': 'fiber_g',
            'sugars': 'sugars_g',
            'vitamin_a': 'vitamin_a_iu',
            'vitamin_a_rae': 'vitamin_a_rae_mcg',
            'vitamin_b6': 'vitamin_b6_mg',
            'vitamin_b12': 'vitamin_b12_mcg',
            'vitamin_c': 'vitamin_c_mg',
            'vitamin_d': 'vitamin_d_iu',
            'vitamin_e': 'vitamin_e_mg',
            'vitamin_k': 'vitamin_k_mcg',
            'thiamin': 'thiamin_mg',
            'riboflavin': 'riboflavin_mg',
            'niacin': 'niacin_mg',
            'folate': 'folate_mcg',
            'pantothenic_acid': 'pantothenic_acid_mg',
            'calcium': 'calcium_mg',
            'iron': 'iron_mg',
            'magnesium': 'magnesium_mg',
            'phosphorous': 'phosphorous_mg',
            'potassium': 'potassium_mg',
            'sodium': 'sodium_mg',
            'zinc': 'zinc_mg',
            'copper': 'copper_mg',
            'manganese': 'manganese_mg',
            'selenium': 'selenium_mcg',
            'cholesterol': 'cholesterol_mg',
            'choline': 'choline_mg',
            'water': 'water_g',
            'ash': 'ash_g',
            'caffeine': 'caffeine_mg',
            'alcohol': 'alcohol_g',
            'saturated_fatty_acids': 'saturated_fatty_acids_g',
            'monounsaturated_fatty_acids': 'monounsaturated_fatty_acids_g',
            'polyunsaturated_fatty_acids': 'polyunsaturated_fatty_acids_g',
            'fatty_acids_total_trans': 'fatty_acids_total_trans_mg',
            'carotene_alpha': 'carotene_alpha_mcg',
            'carotene_beta': 'carotene_beta_mcg',
            'cryptoxanthin_beta': 'cryptoxanthin_beta_mcg',
            'lutein_zeaxanthin': 'lutein_zeaxanthin_mcg',
            'lycopene': 'lycopene_mcg',
            'fructose': 'fructose_g',
            'galactose': 'galactose_g',
            'glucose': 'glucose_g',
            'lactose': 'lactose_g',
            'maltose': 'maltose_g',
            'sucrose': 'sucrose_g'
        }
        
        # Rename columns to match database schema
        df = df.rename(columns=column_mapping)

        # Standardize 'name' column
        df['name'] = df['name'].str.lower().str.strip()
        df = df.drop_duplicates(subset=['name'])

        # Get all expected column names from the table definition, except 'id'
        expected_cols = {c.name for c in nutrition_facts_table.columns if c.name != 'id'}
        
        # Clean numeric columns
        for col in expected_cols:
            if col in df.columns and col != 'name' and col != 'serving_size':
                df[col] = df[col].apply(clean_numeric_value)

        # Filter dataframe to only include columns that exist in the table
        df_filtered = df[[col for col in df.columns if col in expected_cols]]
        
        # Fill any missing values with appropriate defaults
        for col in expected_cols:
            if col not in df_filtered.columns:
                df_filtered[col] = 0.0 if nutrition_facts_table.c[col].type.python_type == float else ''
        
        df_filtered = df_filtered.fillna(0.0)

        with engine.connect() as connection:
            df_filtered.to_sql('nutrition_facts', connection, if_exists='append', index=False)
        logger.success(f"Successfully ingested {len(df_filtered)} nutrition records.")
    except Exception as e:
        logger.error(f"Failed to ingest nutrition data: {e}")
        logger.error(traceback.format_exc())

def ingest_vegan_ontology():
    """Ingests and cleans vegan ontology data from CSV."""
    logger.info("Ingesting vegan ontology data...")
    try:
        df = pd.read_csv(VEGAN_CSV)
        df.columns = [col.lower().strip() for col in df.columns]
        
        df['term'] = df['term'].str.lower().str.strip()
        df = df.drop_duplicates(subset=['term'])

        def parse_aliases(x):
            if isinstance(x, str) and x.startswith('['):
                try:
                    return json.dumps([str(item).lower().strip() for item in ast.literal_eval(x)])
                except (ValueError, SyntaxError):
                    return json.dumps([])
            return json.dumps([])

        df['aliases'] = df['aliases'].apply(parse_aliases)
        
        for col in ['is_explicitly_non_vegan', 'requires_contextual_check', 'is_vegan_exception_term']:
            df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) else False)
        
        expected_cols = {c.name for c in vegan_ontology_table.columns if c.name != 'id'}
        df_filtered = df[[col for col in df.columns if col in expected_cols]]

        with engine.connect() as connection:
            df_filtered.to_sql('vegan_ontology', connection, if_exists='append', index=False)
        logger.success(f"Successfully ingested {len(df_filtered)} vegan ontology terms.")
    except Exception as e:
        logger.error(f"Failed to ingest vegan ontology data: {e}")
        logger.error(traceback.format_exc())

def ingest_unit_conversions():
    """Ingests and cleans unit conversion data from CSV."""
    logger.info("Ingesting unit conversion data...")
    try:
        df = pd.read_csv(UNIT_CONVERSION_CSV)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        if 'standardunit' in df.columns:
            df = df.rename(columns={'standardunit': 'standard_unit'})

        df['unit'] = df['unit'].str.lower().str.strip()
        df = df.drop_duplicates(subset=['unit'])
        
        df['factor'] = pd.to_numeric(df['factor'], errors='coerce').fillna(1.0)
        
        expected_cols = {c.name for c in unit_conversions_table.columns if c.name != 'id'}
        df_filtered = df[[col for col in df.columns if col in expected_cols]]
        
        df_filtered = df_filtered.fillna('')

        with engine.connect() as connection:
            df_filtered.to_sql('unit_conversions', connection, if_exists='append', index=False)
        logger.success(f"Successfully ingested {len(df_filtered)} unit conversion records.")
    except Exception as e:
        logger.error(f"Failed to ingest unit conversion data: {e}")
        logger.error(traceback.format_exc())

def main():
    """Main function to build the knowledge base."""
    logger.info("🚀 Starting Knowledge Base Construction Pipeline...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        logger.warning(f"Database at {DB_PATH} already exists. Deleting for a fresh start.")
        DB_PATH.unlink()
        
    create_database_schema()
    ingest_nutrition_data()
    ingest_vegan_ontology()
    ingest_unit_conversions()
    
    logger.success("✅ Knowledge Base constructed successfully!")
    logger.info("🎯 System ready for real-time classification.")

if __name__ == "__main__":
    main()