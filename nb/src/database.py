"""
Unified database access layer for all services.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
from opensearchpy import OpenSearch
from sqlalchemy import create_engine
from loguru import logger

from config import app_config

class DatabaseManager:
    """Unified database manager for SQLite and OpenSearch."""
    
    def __init__(self):
        self.sqlite_engine = create_engine(f"sqlite:///{app_config.DB_PATH}")
        self.opensearch_client = None
        self._init_opensearch()
    
    def _init_opensearch(self):
        """Initialize OpenSearch client with error handling."""
        try:
            self.opensearch_client = OpenSearch(
                hosts=[app_config.OPENSEARCH_URL],
                http_auth=None,
                use_ssl=False,
                verify_certs=False,
                ssl_show_warn=False,
            )
            if self.opensearch_client.ping():
                logger.info("OpenSearch connection established")
            else:
                logger.warning("OpenSearch ping failed")
                self.opensearch_client = None
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {e}")
            self.opensearch_client = None
    
    @contextmanager
    def get_sqlite_connection(self):
        """Get SQLite connection context manager."""
        conn = sqlite3.connect(str(app_config.DB_PATH))
        try:
            yield conn
        finally:
            conn.close()
    
    def get_opensearch_client(self) -> Optional[OpenSearch]:
        """Get OpenSearch client."""
        return self.opensearch_client
    
    def query_nutrition_data(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """Query nutrition data from SQLite database."""
        try:
            with self.get_sqlite_connection() as conn:
                query = "SELECT * FROM nutrition_facts WHERE name = ?"
                df = pd.read_sql(query, conn, params=[ingredient_name.lower().strip()])
                if not df.empty:
                    return df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Failed to query nutrition data for {ingredient_name}: {e}")
        return None
    
    def query_vegan_ontology(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """Query vegan ontology from SQLite database."""
        try:
            with self.get_sqlite_connection() as conn:
                # Try exact match first
                query = "SELECT * FROM vegan_ontology WHERE term = ?"
                df = pd.read_sql(query, conn, params=[ingredient_name.lower().strip()])
                
                if df.empty:
                    # Try alias match
                    query_aliases = "SELECT * FROM vegan_ontology WHERE aliases LIKE ?"
                    df = pd.read_sql(query_aliases, conn, params=[f'%"{ingredient_name}"%'])
                
                if not df.empty:
                    result = df.iloc[0].to_dict()
                    # Parse aliases back from JSON string
                    import json
                    try:
                        result['aliases'] = json.loads(result['aliases']) if result['aliases'] else []
                    except (json.JSONDecodeError, TypeError):
                        result['aliases'] = []
                    
                    result['is_vegan_term'] = not bool(result['is_explicitly_non_vegan'])
                    return result
        except Exception as e:
            logger.error(f"Failed to query vegan ontology for {ingredient_name}: {e}")
        return None
    
    def search_recipes(self, ingredient_query: str, size: int = 12) -> Dict[str, Any]:
        """Search recipes in OpenSearch."""
        if not self.opensearch_client:
            return {"hits": {"hits": [], "total": {"value": 0}}}
        
        query = {
            "query": {
                "match": {
                    "ingredients": {
                        "query": ingredient_query,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        
        try:
            response = self.opensearch_client.search(
                index="recipes",
                body=query,
                size=size
            )
            return response
        except Exception as e:
            logger.error(f"OpenSearch query failed: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}
    
    def get_all_ingredients(self, limit: int = 10000) -> List[str]:
        """Get all ingredients from OpenSearch."""
        if not self.opensearch_client:
            return []
        
        try:
            response = self.opensearch_client.search(
                index="ingredients", 
                body={"query": {"match_all": {}}}, 
                size=limit
            )
            return [hit["_source"]["ingredients"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Failed to get ingredients: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager() 