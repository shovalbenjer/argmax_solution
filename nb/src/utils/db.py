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
    """Returns a new database connection from the centralized SQLAlchemy engine.

    Detailed Description:
        - This function provides a centralized way to obtain database connections throughout the application.
        - It uses the pre-configured SQLAlchemy engine to create connections, ensuring consistent
          database access patterns and connection pooling.
        - The connection should be used in a context manager to ensure proper cleanup.

    Returns:
        - sqlalchemy.engine.Connection: A new database connection object.

    Libraries Used:
        - SQLAlchemy: Provides robust database abstraction and connection management. It's preferred
          over raw sqlite3 connections because it offers connection pooling, better error handling,
          and database-agnostic SQL generation.

    Examples:
        >>> with get_db_connection() as conn:
        ...     result = conn.execute("SELECT COUNT(*) FROM nutrition_facts")
        ...     print(result.fetchone())

    Notes:
        - Always use the returned connection in a context manager to ensure proper resource cleanup.
        - The connection uses the SQLite database located at the configured DB_PATH.
    """
    return engine.connect()

def get_engine():
    """Returns the shared SQLAlchemy engine instance for direct access.

    Detailed Description:
        - This function provides access to the underlying SQLAlchemy engine for cases where
          direct engine access is needed (e.g., for pandas operations or bulk operations).
        - The engine is configured with the SQLite database and manages connection pooling automatically.

    Returns:
        - sqlalchemy.engine.Engine: The configured SQLAlchemy engine instance.

    Examples:
        >>> engine = get_engine()
        >>> df = pd.read_sql("SELECT * FROM nutrition_facts LIMIT 5", engine)
    """
    return engine 