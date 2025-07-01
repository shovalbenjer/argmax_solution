import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, Boolean, MetaData, JSON
from pathlib import Path
from loguru import logger
import json
import asyncio
from typing import List, Dict, Any
from opensearchpy import OpenSearch
from tqdm import tqdm

# Import our local modules for enrichment
from ingredient_processor.processor import get_context_with_rapidfuzz_fallback

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"  # Database goes in nb/src/data
RAW_DATA_DIR = BASE_DIR / "raw_data"  # Raw CSV files are in nb/src/raw_data
DB_PATH = DATA_DIR / "knowledge_graph.db"

# Define file paths
NUTRITION_CSV = RAW_DATA_DIR / "nutrition.csv"
VEGAN_CSV = RAW_DATA_DIR / "vegan_ontology.csv"
UNIT_CONVERSION_CSV = RAW_DATA_DIR / "unit_conversion.csv"

# --- Database Setup ---
engine = create_engine(f"sqlite:///{DB_PATH}")
metadata = MetaData()

# Define table structures with comprehensive nutritional data
nutrition_facts_table = Table('nutrition_facts', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String, nullable=False),
    Column('serving_size', String),
    Column('calories', Float),
    # Macronutrients
    Column('total_fat_g', Float),
    Column('saturated_fat_g', Float),
    Column('protein_g', Float),
    Column('carbohydrate_g', Float),
    Column('fiber_g', Float),
    Column('sugars_g', Float),
    # Vitamins
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
    # Minerals
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
    # Other nutrients
    Column('cholesterol_mg', Float),
    Column('choline_mg', Float),
    Column('water_g', Float),
    Column('ash_g', Float),
    Column('caffeine_mg', Float),
    Column('alcohol_g', Float),
    # Amino acids
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
    # Fatty acids
    Column('saturated_fatty_acids_g', Float),
    Column('monounsaturated_fatty_acids_g', Float),
    Column('polyunsaturated_fatty_acids_g', Float),
    Column('fatty_acids_total_trans_mg', Float),
    # Carotenoids
    Column('carotene_alpha_mcg', Float),
    Column('carotene_beta_mcg', Float),
    Column('cryptoxanthin_beta_mcg', Float),
    Column('lutein_zeaxanthin_mcg', Float),
    Column('lycopene_mcg', Float),
    # Sugars breakdown
    Column('fructose_g', Float),
    Column('galactose_g', Float),
    Column('glucose_g', Float),
    Column('lactose_g', Float),
    Column('maltose_g', Float),
    Column('sucrose_g', Float)
)

vegan_ontology_table = Table('vegan_ontology', metadata,
    Column('id', Integer, primary_key=True),
    Column('term', String, nullable=False),
    Column('aliases', JSON),  # Store aliases as JSON array
    Column('is_explicitly_non_vegan', Boolean),
    Column('description', String),
    Column('source_details', String),
    Column('requires_contextual_check', Boolean),
    Column('is_vegan_exception_term', Boolean)
)

unit_conversions_table = Table('unit_conversions', metadata,
    Column('id', Integer, primary_key=True),
    Column('unit', String, nullable=False),
    Column('abbreviation', String),
    Column('us_value', String),
    Column('metric_equivalent', String),
    Column('notes', String),
    Column('type', String),  # volume or weight
    Column('factor', Float, nullable=False),  # conversion factor
    Column('standard_unit', String, nullable=False)  # ml or g
)

# SOTA Enhancement: Pre-enriched recipes table for fast runtime queries
enriched_recipes_table = Table('enriched_recipes', metadata,
    Column('id', Integer, primary_key=True),
    Column('recipe_id', String, nullable=False, unique=True),
    Column('title', String, nullable=False),
    Column('ingredients_raw', JSON, nullable=False),  # Raw ingredient list
    Column('ingredients_parsed', JSON, nullable=False),  # Parsed ingredients with quantities/units
    Column('enriched_context', JSON, nullable=False),  # Full nutritional context from hybrid search
    Column('total_carbs_estimate', Float, default=0),  # Pre-calculated carb estimate
    Column('has_animal_products', Boolean, default=False),  # Pre-calculated vegan flag
    Column('confidence_score', Float, default=0),  # Confidence in enrichment quality
    Column('processing_timestamp', String, nullable=False)  # When enriched
)

def create_database_schema():
    """Creates the database schema using SQLAlchemy.

    Detailed Description:
        - This function defines and creates the tables for the knowledge graph database.
        - It uses SQLAlchemy's metadata capabilities to define the table structures programmatically.
        - The function first drops all existing tables to ensure a clean setup (`metadata.drop_all`),
          then creates the new tables (`metadata.create_all`).
        - Finally, it creates indexes on the name/term columns of the tables to speed up
          future search operations.

    Libraries Used:
        - SQLAlchemy: A powerful SQL toolkit and Object-Relational Mapper (ORM). Here, it's used
          for schema definition and creation in a database-agnostic way.
        - loguru: For logging the progress of the schema creation.
    """
    logger.info("Creating database schema...")
    metadata.drop_all(engine)
    metadata.create_all(engine)
    with engine.connect() as connection:
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_nutrition_name ON nutrition_facts (name);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_vegan_term ON vegan_ontology (term);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_unit_name ON unit_conversions (unit);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_enriched_recipe_id ON enriched_recipes (recipe_id);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_enriched_title ON enriched_recipes (title);"))
    logger.success("Database schema created successfully.")

def clean_numeric_value(value_str):
    """Helper function to clean numeric values from CSV data."""
    import re
    if pd.isna(value_str) or value_str == '' or str(value_str).strip() == '0':
        return 0.0
    # Remove units and non-numeric characters, keep only digits and decimal points
    cleaned = re.sub(r'[^\d\.-]', '', str(value_str))
    try:
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def ingest_nutrition_data():
    """Ingests comprehensive nutrition data from CSV file with all nutritional components.

    This enhanced version captures the full nutritional profile including:
    - All macronutrients (carbs, protein, fats with breakdown)
    - Complete vitamin profile (A, B-complex, C, D, E, K)
    - Full mineral content (calcium, iron, potassium, etc.)
    - Amino acid profile
    - Fatty acid breakdown
    - Carotenoids and other phytonutrients
    """
    logger.info("Ingesting comprehensive nutrition data...")
    try:
        # Read with proper headers from the CSV
        df = pl.read_csv(NUTRITION_CSV)
        
        # Map CSV columns to our database schema (skipping index column)
        nutrition_mapping = {
            'name': 'name',
            'serving_size': 'serving_size', 
            'calories': 'calories',
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
            'irom': 'iron_mg',  # Note: CSV has 'irom' instead of 'iron'
            'magnesium': 'magnesium_mg',
            'phosphorous': 'phosphorous_mg',
            'potassium': 'potassium_mg',
            'sodium': 'sodium_mg',
            'zink': 'zinc_mg',  # Note: CSV has 'zink' instead of 'zinc'
            'copper': 'copper_mg',
            'manganese': 'manganese_mg',
            'selenium': 'selenium_mcg',
            'cholesterol': 'cholesterol_mg',
            'choline': 'choline_mg',
            'water': 'water_g',
            'ash': 'ash_g',
            'caffeine': 'caffeine_mg',
            'alcohol': 'alcohol_g',
            'alanine': 'alanine_g',
            'arginine': 'arginine_g',
            'aspartic_acid': 'aspartic_acid_g',
            'cystine': 'cystine_g',
            'glutamic_acid': 'glutamic_acid_g',
            'glycine': 'glycine_g',
            'histidine': 'histidine_g',
            'isoleucine': 'isoleucine_g',
            'leucine': 'leucine_g',
            'lysine': 'lysine_g',
            'methionine': 'methionine_g',
            'phenylalanine': 'phenylalanine_g',
            'proline': 'proline_g',
            'serine': 'serine_g',
            'threonine': 'threonine_g',
            'tryptophan': 'tryptophan_g',
            'tyrosine': 'tyrosine_g',
            'valine': 'valine_g',
            'saturated_fatty_acids': 'saturated_fatty_acids_g',
            'monounsaturated_fatty_acids': 'monounsaturated_fatty_acids_g',
            'polyunsaturated_fatty_acids': 'polyunsaturated_fatty_acids_g',
            'fatty_acids_total_trans': 'fatty_acids_total_trans_mg',
            'carotene_alpha': 'carotene_alpha_mcg',
            'carotene_beta': 'carotene_beta_mcg',
            'cryptoxanthin_beta': 'cryptoxanthin_beta_mcg',
            'lutein_zeaxanthin': 'lutein_zeaxanthin_mcg',
            'lucopene': 'lycopene_mcg',  # Note: CSV has 'lucopene' instead of 'lycopene'
            'fructose': 'fructose_g',
            'galactose': 'galactose_g',
            'glucose': 'glucose_g',
            'lactose': 'lactose_g',
            'maltose': 'maltose_g',
            'sucrose': 'sucrose_g'
        }
        
        # Clean the name column
        df = df.with_columns(
            pl.col("name").str.to_lowercase().str.strip_chars()
        )
        
        # Clean all numeric columns
        for csv_col, db_col in nutrition_mapping.items():
            if csv_col in df.columns and csv_col != 'name':
                df = df.with_columns(
                    pl.col(csv_col).map_elements(clean_numeric_value, return_dtype=pl.Float64).alias(db_col)
                )
        
        # Select only the columns we have mappings for
        available_cols = ['name'] + [db_col for csv_col, db_col in nutrition_mapping.items() 
                                    if csv_col in df.columns and csv_col != 'name']
        df_clean = df.select(available_cols)
        
        # Fill any remaining nulls
        df_clean = df_clean.fill_null(0.0)
        
        # Use SQLAlchemy Core for insertion
        with engine.connect() as connection:
            connection.execute(nutrition_facts_table.insert(), df_clean.to_dicts())
            connection.commit()
        logger.success(f"Successfully ingested {len(df_clean)} comprehensive nutrition records with {len(available_cols)-1} nutritional components.")
    except Exception as e:
        logger.error(f"Failed to ingest nutrition data: {e}")
        import traceback
        logger.error(traceback.format_exc())

def ingest_vegan_ontology():
    """Ingests comprehensive vegan ontology with full contextual information.
    
    This enhanced version captures:
    - Main terms and their aliases
    - Explicit non-vegan status flags
    - Detailed descriptions and source information
    - Contextual checking requirements
    - Exception handling for edge cases
    """
    logger.info("Ingesting comprehensive vegan ontology data...")
    try:
        df = pl.read_csv(VEGAN_CSV)
        
        # Clean and process the data
        df = df.with_columns([
            pl.col("term").str.to_lowercase().str.strip_chars(),
            # Parse aliases from string representation to proper JSON
            pl.col("aliases").map_elements(
                lambda x: json.loads(x) if x and x != '[]' else [], 
                return_dtype=pl.List(pl.String)
            ).alias("aliases"),
            # Convert boolean-like integers to actual booleans
            pl.col("is_explicitly_non_vegan").cast(pl.Boolean),
            pl.col("requires_contextual_check").cast(pl.Boolean),
            pl.col("is_vegan_exception_term").cast(pl.Boolean)
        ])
        
        # Convert aliases list to JSON string for SQLite storage
        df = df.with_columns(
            pl.col("aliases").map_elements(
                lambda x: json.dumps(x) if isinstance(x, list) else json.dumps([]),
                return_dtype=pl.String
            )
        )
        
        # Use SQLAlchemy Core for insertion
        with engine.connect() as connection:
            connection.execute(vegan_ontology_table.insert(), df.to_dicts())
            connection.commit()
        logger.success(f"Successfully ingested {len(df)} comprehensive vegan ontology terms with contextual information.")
    except Exception as e:
        logger.error(f"Failed to ingest vegan ontology data: {e}")
        import traceback
        logger.error(traceback.format_exc())

def ingest_unit_conversions():
    """Ingests comprehensive unit conversion data with proper measurement context.
    
    This enhanced version preserves all the rich conversion information:
    - Standard unit names and abbreviations
    - US vs Metric equivalents
    - Volume vs Weight classifications
    - Conversion factors to standard units (ml/g)
    - Contextual notes about usage
    """
    logger.info("Ingesting comprehensive unit conversion data...")
    try:
        df = pl.read_csv(UNIT_CONVERSION_CSV)
        
        # Clean column names (remove any extra whitespace)
        df = df.rename({col: col.strip() for col in df.columns})
        
        # Map CSV columns to database schema (preserving all information)
        column_mapping = {
            'Unit': 'unit',
            'Abbreviation': 'abbreviation', 
            'US Value': 'us_value',
            'Metric Equivalent': 'metric_equivalent',
            'Notes': 'notes',
            'Type': 'type',
            'Factor': 'factor',
            'StandardUnit': 'standard_unit'
        }
        
        # Rename columns to match our schema
        for csv_col, db_col in column_mapping.items():
            if csv_col in df.columns:
                df = df.rename({csv_col: db_col})
        
        # Clean and standardize the data
        df = df.with_columns([
            pl.col("unit").str.to_lowercase().str.strip_chars(),
            pl.col("abbreviation").str.strip_chars(),
            pl.col("type").str.to_lowercase().str.strip_chars(),
            pl.col("standard_unit").str.to_lowercase().str.strip_chars(),
            # Ensure factor is numeric
            pl.col("factor").cast(pl.Float64)
        ])
        
        # Fill null values appropriately (one column at a time to avoid Polars struct issues)
        df = df.with_columns([
            pl.col("abbreviation").fill_null(""),
            pl.col("us_value").fill_null(""),
            pl.col("metric_equivalent").fill_null(""),
            pl.col("notes").fill_null(""),
            pl.col("type").fill_null("volume"),
            pl.col("factor").fill_null(1.0),
            pl.col("standard_unit").fill_null("g")
        ])
        
        # Use SQLAlchemy Core for insertion
        with engine.connect() as connection:
            connection.execute(unit_conversions_table.insert(), df.to_dicts())
            connection.commit()
        logger.success(f"Successfully ingested {len(df)} comprehensive unit conversion records with full measurement context.")
    except Exception as e:
        logger.error(f"Failed to ingest unit conversion data: {e}")
        import traceback
        logger.error(traceback.format_exc())

def enrich_single_recipe(recipe_data: Dict) -> Dict:
    """Enriches a single recipe with comprehensive nutritional and dietary context.
    
    This function processes each recipe through our SOTA hybrid search cascade
    to pre-compute all nutritional information for fast runtime queries.
    """
    try:
        from ingredient_parser import parse_ingredient
        
        recipe_id = recipe_data.get('_id', 'unknown')
        title = recipe_data.get('_source', {}).get('title', 'Unknown Recipe')
        ingredients_raw = recipe_data.get('_source', {}).get('ingredients', [])
        
        if not ingredients_raw or not isinstance(ingredients_raw, list):
            return None
            
        # Parse and enrich each ingredient
        ingredients_parsed = []
        enriched_contexts = []
        total_carbs = 0.0
        has_animal_products = False
        confidence_scores = []
        
        for raw_ingredient in ingredients_raw:
            try:
                # Parse ingredient structure using ingredient-parser-nlp
                parsed = parse_ingredient(raw_ingredient)
                
                # Extract name from the parsed ingredient
                if parsed.name and len(parsed.name) > 0:
                    name = parsed.name[0].text
                else:
                    name = raw_ingredient
                
                # Extract quantity and unit
                quantity = 1.0
                unit = "unit"
                if parsed.amount and len(parsed.amount) > 0:
                    amount = parsed.amount[0]
                    if hasattr(amount, 'quantity') and amount.quantity:
                        try:
                            quantity = float(amount.quantity)
                        except (ValueError, TypeError):
                            quantity = 1.0
                    if hasattr(amount, 'unit') and amount.unit:
                        unit = str(amount.unit)
                
                ingredient_info = {
                    "raw": raw_ingredient,
                    "name": name.lower().strip(),
                    "quantity": quantity,
                    "unit": unit
                }
                ingredients_parsed.append(ingredient_info)
                
                # Get enriched context via hybrid search cascade
                context = get_context_with_rapidfuzz_fallback(raw_ingredient)
                enriched_contexts.append(context)
                
                # Pre-calculate nutritional metrics using comprehensive data
                if context.get('results'):
                    for result in context['results']:
                        # Extract carbohydrate information (now with more detailed breakdown)
                        carbs = result.get('carbohydrate_g', 0) or 0
                        if isinstance(carbs, (int, float)) and carbs > 0:
                            # Scale by quantity
                            scaled_carbs = carbs * (ingredient_info['quantity'] / 100.0)  # Per 100g basis
                            total_carbs += scaled_carbs
                            
                        # Enhanced vegan checking with comprehensive ontology
                        vegan_status = result.get('is_vegan_term')
                        if vegan_status is False or result.get('is_explicitly_non_vegan') is True:
                            has_animal_products = True
                            
                        # Calculate confidence based on match quality
                        match_score = result.get('match_score', 0)
                        if match_score:
                            confidence_scores.append(match_score / 100.0)  # Normalize to 0-1
                            
            except Exception as e:
                logger.debug(f"Error processing ingredient '{raw_ingredient}': {e}")
                # Add minimal fallback data
                ingredients_parsed.append({
                    "raw": raw_ingredient,
                    "name": raw_ingredient,
                    "quantity": 1.0,
                    "unit": "unit"
                })
                enriched_contexts.append({"original": raw_ingredient, "match_type": "none", "results": []})
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "recipe_id": recipe_id,
            "title": title,
            "ingredients_raw": ingredients_raw,
            "ingredients_parsed": ingredients_parsed,
            "enriched_context": enriched_contexts,
            "total_carbs_estimate": total_carbs,
            "has_animal_products": has_animal_products,
            "confidence_score": avg_confidence,
            "processing_timestamp": pl.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Failed to enrich recipe {recipe_data.get('_id', 'unknown')}: {e}")
        return None

def ingest_and_enrich_recipes():
    """SOTA Feature: Ingests and enriches the entire recipe dataset offline.
    
    This function performs comprehensive offline enrichment of all recipes,
    storing pre-computed nutritional and dietary information for fast runtime queries.
    """
    logger.info("🚀 Starting SOTA Recipe Enrichment Pipeline...")
    
    try:
        # Connect to OpenSearch to fetch all recipes
        os_client = OpenSearch([{"host": "localhost", "port": 9200}])
        
        # Check OpenSearch availability
        if not os_client.ping():
            logger.warning("OpenSearch not available. Skipping recipe enrichment.")
            logger.info("To enable recipe enrichment, ensure OpenSearch is running with recipe data.")
            return
            
        # Check if recipes index exists
        if not os_client.indices.exists(index="recipes"):
            logger.warning("No 'recipes' index found in OpenSearch. Skipping recipe enrichment.")
            logger.info("To enable recipe enrichment, first populate OpenSearch with recipe data.")
            return
            
        # Fetch all recipes in batches for memory efficiency
        batch_size = 1000
        scroll_timeout = "5m"
        
        # Initial search with scroll
        response = os_client.search(
            index="recipes",
            scroll=scroll_timeout,
            size=batch_size,
            body={"query": {"match_all": {}}}
        )
        
        scroll_id = response['_scroll_id']
        total_recipes = response['hits']['total']['value']
        processed_count = 0
        
        logger.info(f"Found {total_recipes} recipes to enrich. Processing in batches of {batch_size}...")
        
        # Process initial batch
        enriched_batch = []
        for hit in tqdm(response['hits']['hits'], desc="Enriching first batch"):
            enriched = enrich_single_recipe(hit)
            if enriched:
                enriched_batch.append(enriched)
                
        # Insert first batch
        if enriched_batch:
            with engine.connect() as connection:
                connection.execute(enriched_recipes_table.insert(), enriched_batch)
                connection.commit()
            processed_count += len(enriched_batch)
            logger.info(f"Processed first batch: {processed_count}/{total_recipes} recipes")
        
        # Continue with scroll for remaining batches
        while True:
            try:
                response = os_client.scroll(
                    scroll_id=scroll_id,
                    scroll=scroll_timeout
                )
                
                if not response['hits']['hits']:
                    break
                    
                # Process current batch
                enriched_batch = []
                for hit in tqdm(response['hits']['hits'], desc=f"Enriching batch {processed_count//batch_size + 1}"):
                    enriched = enrich_single_recipe(hit)
                    if enriched:
                        enriched_batch.append(enriched)
                        
                # Insert batch
                if enriched_batch:
                    with engine.connect() as connection:
                        connection.execute(enriched_recipes_table.insert(), enriched_batch)
                        connection.commit()
                    processed_count += len(enriched_batch)
                    logger.info(f"Processed batch: {processed_count}/{total_recipes} recipes")
                    
            except Exception as e:
                logger.error(f"Error during scroll processing: {e}")
                break
                
        # Clear scroll
        try:
            os_client.clear_scroll(scroll_id=scroll_id)
        except:
            pass
            
        logger.success(f"✅ Recipe enrichment completed! Processed {processed_count} recipes.")
        logger.info(f"💾 Enriched data stored in knowledge_graph.db for fast runtime queries.")
        
    except Exception as e:
        logger.error(f"Recipe enrichment failed: {e}")
        logger.info("Continuing with basic ingestion. Recipe enrichment can be run separately later.")

def main():
    """Main function to orchestrate the entire SOTA data ingestion and enrichment pipeline.

    Detailed Description:
        - This function serves as the entry point for the comprehensive data pipeline.
        - It ensures the output directory for the database exists.
        - It deletes any pre-existing database file to ensure a fresh start.
        - It then calls the functions in sequence to:
            1. Create the database schema (including enriched recipes table).
            2. Ingest nutrition data (foundation for enrichment).
            3. Ingest the vegan ontology (for dietary classification).
            4. Ingest unit conversions (for quantity normalization).
            5. **SOTA Enhancement**: Enrich entire recipe dataset offline for fast runtime queries.
        - This ensures a reproducible and ordered data pipeline with pre-computed enrichment.
    """
    logger.info("🚀 Starting SOTA Data Ingestion & Enrichment Pipeline...")
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        logger.info(f"Database already exists at {DB_PATH}. Deleting and recreating.")
        DB_PATH.unlink()
        
    # Phase 1: Core Knowledge Graph Construction
    logger.info("📊 Phase 1: Building Core Knowledge Graph...")
    create_database_schema()
    ingest_nutrition_data()
    ingest_vegan_ontology()
    ingest_unit_conversions()
    
    # Phase 2: SOTA Recipe Enrichment (Offline Processing)
    logger.info("🔬 Phase 2: SOTA Offline Recipe Enrichment...")
    ingest_and_enrich_recipes()
    
    logger.success("✅ Complete SOTA data pipeline finished successfully!")
    logger.info("🎯 System ready for fast runtime queries with pre-enriched data.")

if __name__ == "__main__":
    main() 