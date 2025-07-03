"""
Context-Aware Diet Classifier - Function-Calling Architecture

Integrates structured query results with LLM reasoning for diet classification.
Uses a modern Qwen Agent -> JSON -> SQL pipeline.
"""
import sqlite3
import json
import asyncio
from typing import Dict, List, Any
from pathlib import Path
import logging
import polars as pl

from config import app_config
from llm_client import LLMClient
from database import db_manager
from function_calling_handler import FunctionCallingHandler
from query_engine import translate_json_to_sql

# Configure professional logging
logger = logging.getLogger(__name__)

def execute_sql_query(sql: str, params: List[Any]) -> str:
    """Executes a SQL query with parameters and returns the formatted results."""
    try:
        with db_manager.get_sqlite_connection() as conn:
            df = pl.read_database(query=sql, connection=conn, execute_options={"parameters": params})
            
            if df.is_empty():
                return "No matching data found in the database."
            
            # Convert to a list of dictionaries for clean JSON output
            return json.dumps(df.to_dicts(), indent=2)
            
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return f"Database query error: {e}"

class ContextAwareDietClassifier:
    """
    Enhanced diet classifier using a function-calling agent to query
    the knowledge base and a separate LLM for final classification.
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.function_handler = FunctionCallingHandler()
        
    async def _get_context_for_ingredient(self, ingredient: str) -> str:
        """
        Generates a JSON query, translates it to SQL, and executes it
        to retrieve context for a single ingredient.
        """
        question = f"What are the nutritional values and vegan status for the ingredient: {ingredient}?"
        
        # Step 1: LLM generates a structured JSON query
        json_query = await self.function_handler.generate_json_query(question)
        if "error" in json_query:
            error_msg = f"Failed to generate JSON query for '{ingredient}': {json_query['error']}"
            logger.error(error_msg)
            return error_msg

        # Step 2: Translate JSON to a safe SQL query
        try:
            sql, params = translate_json_to_sql(json_query)
            logger.info(f"Translated SQL for '{ingredient}': {sql} with params {params}")
        except ValueError as e:
            error_msg = f"Failed to translate JSON to SQL for '{ingredient}': {e}"
            logger.error(error_msg)
            return error_msg
        
        # Step 3: Execute the SQL query
        return execute_sql_query(sql, params)

    async def classify_single_ingredient(self, ingredient: str) -> Dict[str, Any]:
        """Classifies a single ingredient using the modern Text-to-JSON pipeline."""
        
        logger.info(f"Starting classification for ingredient: {ingredient}")
        
        retrieved_context = await self._get_context_for_ingredient(ingredient)

        # Step 4: Use a separate LLM for final classification based on the retrieved context
        prompt = f"""You are a nutrition expert judging an ingredient based on factual data.

        **Ingredient:**
        {ingredient}

        **Data from Knowledge Base:**
        {retrieved_context}

        **Your Task:**
        Analyze the data to classify the ingredient's compliance with keto and vegan diets.
        - **Keto:** Net carbs (carbohydrate_g - fiber_g) should be low (e.g., <= {app_config.KETO_CARBS_THRESHOLD}g per 100g).
        - **Vegan:** Must not be an animal product. Check `is_explicitly_non_vegan` flags and look for indicators like `cholesterol_mg > 0`.

        Provide your final judgment in a valid JSON format.

        **JSON Output Format:**
        {{
          "is_keto": boolean,
          "is_vegan": boolean,
          "reasoning": "A brief, evidence-based explanation for your decision.",
          "confidence": "high | medium | low"
        }}
        """
        
        model_name = "qwen/qwen3-0.6b-gguf:q8_0"
        if not self.llm_client.is_model_available(model_name):
            logger.error(f"Classification model '{model_name}' is not available.")
            return {"error": "Classification model not available"}

        result = await self.llm_client.query_async(model_name, prompt, as_json=True)
        return result

    async def classify_recipe(self, ingredients: List[str]) -> Dict[str, Any]:
        """
        Classifies an entire recipe by classifying each ingredient individually
        and aggregating the results.
        """
        classifications = await asyncio.gather(
            *(self.classify_single_ingredient(ing) for ing in ingredients)
        )
        
        recipe_is_keto = True
        recipe_is_vegan = True
        full_reasoning = []
        
        for i, result in enumerate(classifications):
            ingredient = ingredients[i]
            if "error" in result:
                # If any ingredient fails, the whole recipe is conservatively marked as non-compliant
                recipe_is_keto = False
                recipe_is_vegan = False
                full_reasoning.append(f"{ingredient}: Classification failed - {result['error']}")
                continue

            if not result.get("is_keto", False):
                recipe_is_keto = False
            if not result.get("is_vegan", False):
                recipe_is_vegan = False
            
            full_reasoning.append(f"{ingredient}: {result.get('reasoning', 'No reasoning provided.')}")
            
        return {
            "recipe_is_keto": recipe_is_keto,
            "recipe_is_vegan": recipe_is_vegan,
            "ingredient_analysis": classifications,
            "summary_reasoning": " ".join(full_reasoning)
        }

if __name__ == '__main__':
    async def test_classifier():
        classifier = ContextAwareDietClassifier()

        # Test 1: Single ingredient (non-vegan, keto)
        print("\n--- Testing Single Ingredient: Chicken Breast ---")
        result1 = await classifier.classify_single_ingredient("100g chicken breast")
        print(json.dumps(result1, indent=2))

        # Test 2: Single ingredient (vegan, not keto)
        print("\n--- Testing Single Ingredient: Potato ---")
        result2 = await classifier.classify_single_ingredient("1 medium potato")
        print(json.dumps(result2, indent=2))
        
        # Test 3: Full Recipe (Keto, Not Vegan)
        recipe1 = ["100g chicken breast", "50g spinach", "1 tbsp olive oil"]
        print(f"\n--- Testing Recipe: {recipe1} ---")
        recipe_result1 = await classifier.classify_recipe(recipe1)
        print(json.dumps(recipe_result1, indent=2))
        
        # Test 4: Full Recipe (Vegan, Not Keto)
        recipe2 = ["1 medium potato", "1 cup of rice", "1 tbsp sugar"]
        print(f"\n--- Testing Recipe: {recipe2} ---")
        recipe_result2 = await classifier.classify_recipe(recipe2)
        print(json.dumps(recipe_result2, indent=2))

    asyncio.run(test_classifier()) 