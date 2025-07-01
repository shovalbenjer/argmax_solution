import polars as pl
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, Boolean, MetaData
from pathlib import Path
from loguru import logger

# --- Configuration ---
# The script runs inside /usr/src/app. The 'src' directory is mounted here.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data" # This is for the output DB
RAW_DATA_DIR = BASE_DIR / "raw_data" # The raw data is inside the 'src' mount
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
    Column('name', String, nullable=False),
    Column('carbohydrates_g', Float),
    Column('protein_g', Float),
    Column('fat_g', Float)
)

vegan_ontology_table = Table('vegan_ontology', metadata,
    Column('id', Integer, primary_key=True),
    Column('term', String, nullable=False),
    Column('is_vegan', Boolean)
)

unit_conversions_table = Table('unit_conversions', metadata,
    Column('id', Integer, primary_key=True),
    Column('ingredient_name', String, nullable=False),
    Column('from_unit', String, nullable=False),
    Column('to_unit', String, nullable=False),
    Column('multiplier', Float, nullable=False)
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
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_unit_ingredient ON unit_conversions (ingredient_name);"))
    logger.success("Database schema created successfully.")

def ingest_nutrition_data():
    """Ingests and cleans nutrition data from a CSV file into the database.

    Detailed Description:
        - This function reads the raw nutrition CSV file using Polars.
        - It performs several cleaning and transformation steps:
            1. Selects specific columns from the raw CSV by their position.
            2. Renames columns to match the database schema.
            3. Cleans the 'name' column by converting it to lowercase and stripping whitespace.
            4. Cleans numeric columns by removing non-numeric characters (like 'g') and casting them to float.
            5. Fills any resulting null values with 0.
        - The cleaned data is then inserted into the `nutrition_facts` table.

    Libraries Used:
        - polars: For fast and memory-efficient reading and manipulation of the CSV data.
        - SQLAlchemy: To connect to the database and execute the insert operation.
    """
    logger.info("Ingesting nutrition data...")
    try:
        df = pl.read_csv(NUTRITION_CSV, has_header=False)
        # Select and rename relevant columns based on their positions
        df = df.select([
            pl.col("column_2").alias("name"),
            pl.col("column_22").alias("carbohydrates_g"),
            pl.col("column_4").alias("protein_g"),
            pl.col("column_5").alias("fat_g")
        ]).with_columns(
            pl.col("name").str.to_lowercase().str.strip_chars()
        )
        
        # Clean numeric columns by removing non-numeric characters (like 'g')
        for col in ["carbohydrates_g", "protein_g", "fat_g"]:
            df = df.with_columns(
                pl.col(col).cast(pl.String).str.replace(r"[^\d\.]", "").cast(pl.Float32, strict=False)
            )

        df = df.fill_null(0) # Fill any remaining nulls after cleaning and casting
        
        # Use SQLAlchemy Core for insertion
        with engine.connect() as connection:
            connection.execute(nutrition_facts_table.insert(), df.to_dicts())
            connection.commit()
        logger.success(f"Successfully ingested {len(df)} nutrition records.")
    except Exception as e:
        logger.error(f"Failed to ingest nutrition data: {e}")

def ingest_vegan_ontology():
    """Ingests vegan ontology terms from a CSV file into the database.

    Detailed Description:
        - This function reads the vegan ontology CSV, which contains a list of terms and
          whether they are considered vegan.
        - It performs a simple cleaning step on the 'term' column to standardize it
          (lowercase and stripped whitespace).
        - The cleaned data is then inserted into the `vegan_ontology` table.

    Libraries Used:
        - polars: For reading the CSV data.
        - SQLAlchemy: For inserting the data into the database.
    """
    logger.info("Ingesting vegan ontology data...")
    try:
        df = pl.read_csv(VEGAN_CSV).with_columns(
            pl.col("term").str.to_lowercase().str.strip_chars()
        )
        # Use SQLAlchemy Core for insertion
        with engine.connect() as connection:
            connection.execute(vegan_ontology_table.insert(), df.to_dicts())
            connection.commit()
        logger.success(f"Successfully ingested {len(df)} vegan ontology terms.")
    except Exception as e:
        logger.error(f"Failed to ingest vegan ontology data: {e}")

def ingest_unit_conversions():
    """Ingests and transforms unit conversion data from a CSV into the database.

    Detailed Description:
        - This function handles the ingestion of unit conversion data, which requires
          significant transformation to fit the target schema.
        - The source CSV format is not ideal, so this function makes some assumptions:
            - The 'Unit' column is used as a stand-in for the ingredient name.
            - It assumes all conversions are to a base unit of 'g' (grams).
        - It renames columns, creates new ones with literal values, and standardizes the
          ingredient name before inserting the data into the `unit_conversions` table.

    Libraries Used:
        - polars: For data reading and transformation.
        - SQLAlchemy: For database insertion.
    """
    logger.info("Ingesting unit conversion data...")
    try:
        df = pl.read_csv(UNIT_CONVERSION_CSV)
        df = df.rename({col: col.strip() for col in df.columns})

        # This data needs significant transformation to fit our schema.
        # We will assume 'Unit' can be a stand-in for 'ingredient_name' for now
        # and that 'Factor' is the multiplier. This is an imperfect mapping.
        df_transformed = df.select([
            pl.col("Unit").alias("ingredient_name"),
            pl.col("Unit").alias("from_unit"), # Placeholder
            pl.lit("g").alias("to_unit"),       # Assuming conversion to grams
            pl.col("Factor").alias("multiplier")
        ]).with_columns(
            pl.col("ingredient_name").str.to_lowercase().str.strip_chars()
        ).fill_null(0)

        # Use SQLAlchemy Core for insertion
        with engine.connect() as connection:
            connection.execute(unit_conversions_table.insert(), df_transformed.to_dicts())
            connection.commit()
        logger.success(f"Successfully ingested and transformed {len(df_transformed)} unit conversion records.")
    except Exception as e:
        logger.error(f"Failed to ingest unit conversion data: {e}")

def main():
    """Main function to orchestrate the entire data ingestion pipeline.

    Detailed Description:
        - This function serves as the entry point for the data ingestion script.
        - It ensures the output directory for the database exists.
        - It deletes any pre-existing database file to ensure a fresh start.
        - It then calls the other functions in sequence to:
            1. Create the database schema.
            2. Ingest nutrition data.
            3. Ingest the vegan ontology.
            4. Ingest unit conversions.
        - This ensures a reproducible and ordered data ingestion process.
    """
    logger.info("Starting data ingestion pipeline...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        logger.info(f"Database already exists at {DB_PATH}. Deleting and recreating.")
        DB_PATH.unlink()
        
    create_database_schema()
    ingest_nutrition_data()
    ingest_vegan_ontology()
    ingest_unit_conversions()
    logger.success("Data ingestion pipeline completed successfully.")

if __name__ == "__main__":
    main() 