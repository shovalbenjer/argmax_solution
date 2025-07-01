"""
Unified Diet Classification System

This module provides consistent dietary classification across all services.
Eliminates code duplication and ensures reliable keto/vegan classification.
"""
import json
import asyncio
from functools import lru_cache
from typing import List, Dict, Optional, Any
from loguru import logger

try:
    from ingredient_parser import parse_ingredient
except ImportError:
    logger.warning("ingredient_parser not available, using fallback parsing")
    parse_ingredient = None

from .config import app_config
from .database import db_manager
from .llm_client import LLMClient

class IngredientProcessor:
    """Processes and analyzes individual ingredients."""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def parse_ingredient(self, raw_ingredient: str) -> Dict[str, Any]:
        """Parse ingredient string into structured components."""
        if parse_ingredient:
            try:
                parsed = parse_ingredient(raw_ingredient, discard_isolated_stop_words=True)
                name = parsed.name[0].text if isinstance(parsed.name, list) else parsed.name.text
                
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
                
                return {
                    "name": name.lower().strip(),
                    "quantity": quantity,
                    "unit": unit
                }
            except Exception as e:
                logger.warning(f"Parsing failed for '{raw_ingredient}': {e}")
        
        # Fallback parsing
        return {
            "name": raw_ingredient.lower().strip(),
            "quantity": 1.0,
            "unit": "unit"
        }
    
    def get_ingredient_context(self, raw_ingredient: str) -> Dict[str, Any]:
        """Get comprehensive context for an ingredient."""
        parsed_info = self.parse_ingredient(raw_ingredient)
        
        # Get nutrition data
        nutrition_data = db_manager.query_nutrition_data(parsed_info["name"])
        
        # Get vegan status
        vegan_data = db_manager.query_vegan_ontology(parsed_info["name"])
        
        return {
            "original": raw_ingredient,
            "parsed": parsed_info,
            "nutrition": nutrition_data,
            "vegan_info": vegan_data
        }

class DietClassifier:
    """Main diet classification engine."""
    
    def __init__(self):
        self.processor = IngredientProcessor()
    
    def is_ingredient_keto(self, ingredient: str) -> bool:
        """Check if a single ingredient is keto-friendly."""
        context = self.processor.get_ingredient_context(ingredient)
        
        nutrition = context.get('nutrition')
        if nutrition and 'carbohydrate_g' in nutrition:
            carbs = nutrition['carbohydrate_g']
            if isinstance(carbs, (int, float)):
                return carbs < app_config.KETO_CARBS_THRESHOLD
        
        # If no nutrition data, be conservative
        return False
    
    def is_ingredient_vegan(self, ingredient: str) -> bool:
        """Check if a single ingredient is vegan-friendly."""
        context = self.processor.get_ingredient_context(ingredient)
        
        vegan_info = context.get('vegan_info')
        if vegan_info:
            # If explicitly marked as non-vegan, return False
            if vegan_info.get('is_explicitly_non_vegan'):
                return False
            # If we have vegan term status, use it
            if 'is_vegan_term' in vegan_info:
                return vegan_info['is_vegan_term']
        
        # If no explicit information, assume vegan (plants are generally vegan)
        return True
    
    def is_keto(self, ingredients: List[str]) -> bool:
        """Classify if a recipe is keto-friendly."""
        if not ingredients:
            return False
        
        # All ingredients must be keto for the recipe to be keto
        return all(self.is_ingredient_keto(ingredient) for ingredient in ingredients)
    
    def is_vegan(self, ingredients: List[str]) -> bool:
        """Classify if a recipe is vegan-friendly."""
        if not ingredients:
            return True
        
        # All ingredients must be vegan for the recipe to be vegan
        return all(self.is_ingredient_vegan(ingredient) for ingredient in ingredients)
    
    async def classify_with_llm(self, ingredients: List[str]) -> Dict[str, Any]:
        """Use LLM for complex classification cases."""
        contexts = [self.processor.get_ingredient_context(ing) for ing in ingredients]
        
        prompt = f"""
        Analyze this recipe for dietary compliance.
        
        KETO: Recipe is keto if total carbohydrates are very low (<{app_config.KETO_CARBS_THRESHOLD}g per 100g).
        VEGAN: Recipe is vegan if it contains NO animal products.
        
        Ingredient contexts: {json.dumps(contexts, indent=2)}
        
        Respond with JSON: {{"is_keto": boolean, "is_vegan": boolean, "reasoning": "explanation"}}
        """
        
        try:
            response = await self.processor.llm_client.query_async("qwen:latest", prompt, as_json=True)
            if isinstance(response, dict) and "error" not in response:
                return response
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
        
        # Fallback to rule-based classification
        return {
            "is_keto": self.is_keto(ingredients),
            "is_vegan": self.is_vegan(ingredients),
            "reasoning": "Rule-based classification (LLM unavailable)"
        }

# Global classifier instance
diet_classifier = DietClassifier()

# Convenience functions for backward compatibility
def is_keto(ingredients: List[str]) -> bool:
    """Check if recipe is keto-friendly."""
    return diet_classifier.is_keto(ingredients)

def is_vegan(ingredients: List[str]) -> bool:
    """Check if recipe is vegan-friendly."""
    return diet_classifier.is_vegan(ingredients) 