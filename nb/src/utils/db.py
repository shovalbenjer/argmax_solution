from sqlalchemy import create_engine
from pathlib import Path

# This construct ensures that paths are relative to the project's root `nb` directory,
# making it robust to where the script is called from.
try:
    # Assumes the script is run from within the 'nb' directory structure
    BASE_DIR = Path(__file__).resolve().parent.parent
    if not (BASE_DIR / "data").exists():
        # Fallback for different execution contexts (like top-level)
        BASE_DIR = Path.cwd()
except NameError:
    # Fallback for interactive sessions
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "knowledge_graph.db"
DB_URI = f"sqlite:///{DB_PATH}"

# Ensure the data directory exists
DATA_DIR.mkdir(exist_ok=True)

engine = create_engine(DB_URI)

def get_db_connection():
    """Returns a new database connection from the engine."""
    return engine.connect()

def get_engine():
    """Returns the SQLAlchemy engine instance."""
    return engine 