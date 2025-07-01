"""
SOTA Enriched Query Helper

This module provides efficient access to pre-enriched recipe data stored in the knowledge database.
The offline enrichment process pre-computes all nutritional and dietary information, allowing for
extremely fast runtime queries without the need for real-time ingredient processing.

Key Features:
- Fast recipe lookup by ID or title
- Pre-computed carbohydrate estimates for keto classification
- Pre-computed animal product flags for vegan classification
- Rich nutritional context from hybrid search cascade
- Confidence scores for classification reliability
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import pandas as pd

# Database path
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "knowledge_graph.db"

class EnrichedQueryHelper:
    """Helper class for efficiently querying pre-enriched recipe data."""
    
    def __init__(self, db_path: str = None):
        """Initialize the query helper with database connection."""
        self.db_path = db_path or str(DB_PATH)
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Verify that the enriched database exists."""
        if not Path(self.db_path).exists():
            logger.warning(f"Enriched database not found at {self.db_path}")
            logger.info("Run 'python data_ingestion/ingest.py' to create the enriched database.")
    
    def get_enriched_recipe_by_id(self, recipe_id: str) -> Optional[Dict]:
        """Get a single enriched recipe by its ID.
        
        Returns pre-computed nutritional information including:
        - Parsed ingredients with quantities/units
        - Full nutritional context from hybrid search
        - Pre-calculated carb estimates
        - Pre-calculated vegan/animal product flags
        - Confidence scores
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM enriched_recipes WHERE recipe_id = ?",
                    (recipe_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        "recipe_id": row["recipe_id"],
                        "title": row["title"],
                        "ingredients_raw": json.loads(row["ingredients_raw"]),
                        "ingredients_parsed": json.loads(row["ingredients_parsed"]),
                        "enriched_context": json.loads(row["enriched_context"]),
                        "total_carbs_estimate": row["total_carbs_estimate"],
                        "has_animal_products": bool(row["has_animal_products"]),
                        "confidence_score": row["confidence_score"],
                        "processing_timestamp": row["processing_timestamp"]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error querying enriched recipe {recipe_id}: {e}")
            return None
    
    def get_enriched_recipes_batch(self, recipe_ids: List[str]) -> List[Dict]:
        """Get multiple enriched recipes by their IDs in a single query."""
        if not recipe_ids:
            return []
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join(["?"] * len(recipe_ids))
                cursor = conn.execute(
                    f"SELECT * FROM enriched_recipes WHERE recipe_id IN ({placeholders})",
                    recipe_ids
                )
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        "recipe_id": row["recipe_id"],
                        "title": row["title"],
                        "ingredients_raw": json.loads(row["ingredients_raw"]),
                        "ingredients_parsed": json.loads(row["ingredients_parsed"]),
                        "enriched_context": json.loads(row["enriched_context"]),
                        "total_carbs_estimate": row["total_carbs_estimate"],
                        "has_animal_products": bool(row["has_animal_products"]),
                        "confidence_score": row["confidence_score"],
                        "processing_timestamp": row["processing_timestamp"]
                    })
                return results
                
        except Exception as e:
            logger.error(f"Error querying enriched recipes batch: {e}")
            return []
    
    def fast_classify_keto(self, recipe_id: str, carb_threshold: float = 20.0) -> Optional[bool]:
        """Ultra-fast keto classification using pre-computed carb estimates."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT total_carbs_estimate FROM enriched_recipes WHERE recipe_id = ?",
                    (recipe_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return row[0] <= carb_threshold
                return None
                
        except Exception as e:
            logger.error(f"Error in fast keto classification for {recipe_id}: {e}")
            return None
    
    def fast_classify_vegan(self, recipe_id: str) -> Optional[bool]:
        """Ultra-fast vegan classification using pre-computed animal product flags."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT has_animal_products FROM enriched_recipes WHERE recipe_id = ?",
                    (recipe_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return not bool(row[0])  # Vegan = no animal products
                return None
                
        except Exception as e:
            logger.error(f"Error in fast vegan classification for {recipe_id}: {e}")
            return None
    
    def get_classification_summary(self, recipe_ids: List[str]) -> Dict[str, Any]:
        """Get classification summary for multiple recipes."""
        if not recipe_ids:
            return {"total": 0, "keto": 0, "vegan": 0, "both": 0, "neither": 0}
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join(["?"] * len(recipe_ids))
                cursor = conn.execute(
                    f"""
                    SELECT 
                        recipe_id,
                        total_carbs_estimate,
                        has_animal_products,
                        confidence_score
                    FROM enriched_recipes 
                    WHERE recipe_id IN ({placeholders})
                    """,
                    recipe_ids
                )
                rows = cursor.fetchall()
                
                total = len(rows)
                keto_count = sum(1 for row in rows if row[1] <= 20.0)  # carbs <= 20g
                vegan_count = sum(1 for row in rows if not row[2])  # no animal products
                both_count = sum(1 for row in rows if row[1] <= 20.0 and not row[2])
                neither_count = total - (keto_count + vegan_count - both_count)
                
                avg_confidence = sum(row[3] for row in rows) / total if total > 0 else 0.0
                
                return {
                    "total": total,
                    "keto": keto_count,
                    "vegan": vegan_count,
                    "both": both_count,
                    "neither": neither_count,
                    "keto_percentage": (keto_count / total * 100) if total > 0 else 0,
                    "vegan_percentage": (vegan_count / total * 100) if total > 0 else 0,
                    "average_confidence": avg_confidence
                }
                
        except Exception as e:
            logger.error(f"Error getting classification summary: {e}")
            return {"error": str(e)}
    
    def search_enriched_recipes(self, 
                              title_query: str = None, 
                              min_confidence: float = 0.0,
                              max_carbs: float = None,
                              vegan_only: bool = False,
                              limit: int = 100) -> List[Dict]:
        """Search enriched recipes with various filters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build dynamic query
                conditions = ["confidence_score >= ?"]
                params = [min_confidence]
                
                if title_query:
                    conditions.append("title LIKE ?")
                    params.append(f"%{title_query}%")
                    
                if max_carbs is not None:
                    conditions.append("total_carbs_estimate <= ?")
                    params.append(max_carbs)
                    
                if vegan_only:
                    conditions.append("has_animal_products = ?")
                    params.append(False)
                
                where_clause = " AND ".join(conditions)
                params.append(limit)
                
                cursor = conn.execute(
                    f"""
                    SELECT * FROM enriched_recipes 
                    WHERE {where_clause}
                    ORDER BY confidence_score DESC, total_carbs_estimate ASC
                    LIMIT ?
                    """,
                    params
                )
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        "recipe_id": row["recipe_id"],
                        "title": row["title"],
                        "total_carbs_estimate": row["total_carbs_estimate"],
                        "has_animal_products": bool(row["has_animal_products"]),
                        "confidence_score": row["confidence_score"],
                        "is_keto": row["total_carbs_estimate"] <= 20.0,
                        "is_vegan": not bool(row["has_animal_products"])
                    })
                return results
                
        except Exception as e:
            logger.error(f"Error searching enriched recipes: {e}")
            return []

# Global instance for easy access
enriched_query = EnrichedQueryHelper()

# Convenience functions for direct use
def get_recipe_fast(recipe_id: str) -> Optional[Dict]:
    """Get enriched recipe data quickly."""
    return enriched_query.get_enriched_recipe_by_id(recipe_id)

def classify_fast(recipe_id: str) -> Dict[str, bool]:
    """Get both keto and vegan classifications quickly."""
    keto = enriched_query.fast_classify_keto(recipe_id)
    vegan = enriched_query.fast_classify_vegan(recipe_id)
    return {
        "is_keto": keto if keto is not None else False,
        "is_vegan": vegan if vegan is not None else False
    }

def search_recipes(title_query: str = None, keto_only: bool = False, vegan_only: bool = False, limit: int = 50) -> List[Dict]:
    """Search recipes with common filters."""
    max_carbs = 20.0 if keto_only else None
    return enriched_query.search_enriched_recipes(
        title_query=title_query,
        max_carbs=max_carbs,
        vegan_only=vegan_only,
        limit=limit
    )
