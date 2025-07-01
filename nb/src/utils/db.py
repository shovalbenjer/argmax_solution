"""
Database utility functions - Updated to use shared modules
"""
import sys
import os
from pathlib import Path
from sqlalchemy import create_engine

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.config import app_config
from shared.database import db_manager

def get_engine():
    """Get SQLAlchemy engine for the knowledge graph database."""
    return create_engine(f"sqlite:///{app_config.DB_PATH}")

def get_db_connection():
    """Get database connection using shared database manager."""
    return db_manager.get_sqlite_connection() 