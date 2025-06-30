"""
Database Handler for Elasticsearch.

This module provides a dedicated interface for connecting to and querying
the Elasticsearch database, as specified by the Argmax requirements.
"""
from elasticsearch import Elasticsearch
from loguru import logger
from typing import Optional, Dict, Any

class DatabaseHandler:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseHandler, cls).__new__(cls)
            try:
                cls._instance.client = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=10)
                if not cls._instance.client.ping():
                    raise ConnectionError("Could not connect to Elasticsearch")
                logger.success("Connected to Elasticsearch successfully.")
            except Exception as e:
                logger.critical(f"Failed to connect to Elasticsearch: {e}")
                cls._instance.client = None
        return cls._instance

    def search_ingredient(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        if not self.client: return None
        try:
            # Using a 'match' query for better relevance scoring on text fields.
            res = self.client.search(index="recipes", body={"query": {"match": {"description": ingredient_name}}}, size=1)
            return res['hits']['hits'][0]['_source'] if res['hits']['hits'] else None
        except Exception as e:
            logger.error(f"Elasticsearch query for '{ingredient_name}' failed: {e}")
            return None 