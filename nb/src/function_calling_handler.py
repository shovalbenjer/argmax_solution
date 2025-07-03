"""
Function-Calling Handler for Query Generation

This module uses an LLM (Qwen) to translate natural language question
into a structured JSON query, based on a predefined schema. This decouples
the LLM from SQL syntax and improves robustness.
"""
import json
from typing import Dict, Any, List, Optional
import logging

from llm_client import LLMClient
from config import app_config

logger = logging.getLogger(__name__)

class FunctionCallingHandler:
    """
    Handles translation of natural language questions to structured JSON queries
    using a function-calling approach with an LLM.
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.schema_context = self._build_schema_context()
        self.json_schema = self._get_json_schema()

    def _get_json_schema(self) -> str:
        """Defines the structured JSON schema the LLM should generate."""
        schema = {
            "operation": "search | aggregate",
            "table": "nutrition_facts | vegan_ontology | unit_conversions",
            "select_fields": ["field1", "field2"],
            "filters": [{
                "field": "column_name",
                "operator": "= | != | > | < | >= | <= | LIKE | IN",
                "value": "string | number | list"
            }],
            "aggregation_type": "AVG | COUNT | SUM | MIN | MAX",
            "aggregation_field": "column_name",
            "groupBy": "column_name",
            "limit": "integer"
        }
        return json.dumps(schema, indent=2)

    def _build_schema_context(self) -> str:
        """Builds the database schema context for the LLM prompt."""
        context = """
DATABASE SCHEMA CONTEXT:

1. `nutrition_facts` table:
   - Columns: `name`, `calories`, `carbohydrate_g`, `fiber_g`, `protein_g`, `total_fat_g`, `cholesterol_mg`, `vitamin_b12_mcg`
   - Description: Contains nutritional information per 100g for various food items.

2. `vegan_ontology` table:
   - Columns: `term`, `aliases`, `is_explicitly_non_vegan`, `description`
   - Description: A knowledge base of vegan and non-vegan terms. `is_explicitly_non_vegan` is a boolean.

3. `unit_conversions` table:
   - Columns: `unit`, `factor`, `type`
   - Description: Converts common cooking units to grams.
"""
        return context

    async def generate_json_query(self, question: str) -> Dict[str, Any]:
        """
        Generates a structured JSON query from a natural language question.
        """
        prompt = f"""You are an expert at converting user questions into structured JSON queries.
Your task is to populate the following JSON schema to answer the user's question.

DATABASE SCHEMA:
{self.schema_context}

JSON QUERY FORMAT:
{self.json_schema}

User Question: "{question}"

Based on the user's question, generate a JSON object that matches the specified format.
Ensure all values are valid JSON. For `LIKE` operations, include '%' wildcards in the value.
For `IN` operations, the value must be a list of strings or numbers.
Respond with ONLY the generated JSON object, nothing else.
"""
        model_name = "qwen/qwen3-0.6b-gguf:q8_0"
        
        if not self.llm_client.is_model_available(model_name):
            logger.error(f"Model '{model_name}' is not available for function calling.")
            return {"error": f"Model not available: {model_name}"}

        try:
            result = await self.llm_client.query_async(model_name, prompt, as_json=True)
            
            if "error" in result:
                raise ValueError(result.get("error", "Unknown LLM error"))
            
            # The result from the LLM should already be a dict if as_json=True worked
            if isinstance(result, dict):
                logger.info(f"Successfully generated JSON query for: '{question}'")
                return result
            else:
                raise TypeError(f"LLM returned unexpected type: {type(result)}")

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to generate or parse JSON query for '{question}': {e}")
            return {"error": f"Failed to generate valid JSON: {e}"}

if __name__ == '__main__':
    # Example Usage and Testing
    import asyncio
    
    async def main():
        handler = FunctionCallingHandler()
        
        # Test 1: Simple Search
        question1 = "What are the calories for chicken breast?"
        print(f"\n1. Testing: '{question1}'")
        json_query1 = await handler.generate_json_query(question1)
        print(json.dumps(json_query1, indent=2))
        
        # Test 2: Aggregation
        question2 = "What is the average protein content for items with more than 50g of protein?"
        print(f"\n2. Testing: '{question2}'")
        json_query2 = await handler.generate_json_query(question2)
        print(json.dumps(json_query2, indent=2))
        
        # Test 3: Complex Search
        question3 = "Find non-vegan items from the ontology that are either milk or cheese"
        print(f"\n3. Testing: '{question3}'")
        json_query3 = await handler.generate_json_query(question3)
        print(json.dumps(json_query3, indent=2))

    # Note: This test requires the Qwen model to be loaded and available in Ollama.
    # If the model is not running, the tests will fail with an error message.
    asyncio.run(main()) 