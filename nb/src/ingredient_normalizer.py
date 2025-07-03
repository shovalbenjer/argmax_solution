"""
Ingredient Normalizer for Consistent Caching

This module provides deterministic ingredient normalization using the
ingredient-parser-nlp library to ensure consistent cache keys.
"""

import re
import logging
from typing import Optional, Dict, Any

# Configure professional logging
logger = logging.getLogger(__name__)

try:
    from ingredient_parser import parse_ingredient
    PARSER_AVAILABLE = True
except ImportError:
    logger.warning("ingredient-parser-nlp not available, using fallback normalization")
    PARSER_AVAILABLE = False

class IngredientNormalizer:
    """
    Normalizes ingredient strings for consistent caching and lookup.
    Uses ingredient-parser-nlp when available, falls back to regex-based normalization.
    """
    
    def __init__(self):
        self.common_units = {
            # Volume
            'cup', 'cups', 'c', 'tablespoon', 'tablespoons', 'tbsp', 'tbs', 'tb',
            'teaspoon', 'teaspoons', 'tsp', 'ts', 'fluid ounce', 'fluid ounces', 'fl oz',
            'pint', 'pints', 'pt', 'quart', 'quarts', 'qt', 'gallon', 'gallons', 'gal',
            'liter', 'liters', 'l', 'milliliter', 'milliliters', 'ml',
            # Weight
            'pound', 'pounds', 'lb', 'lbs', 'ounce', 'ounces', 'oz',
            'gram', 'grams', 'g', 'kilogram', 'kilograms', 'kg',
            # Count
            'piece', 'pieces', 'slice', 'slices', 'clove', 'cloves',
            'medium', 'large', 'small', 'whole', 'half', 'quarter'
        }
        
        self.preparation_terms = {
            'chopped', 'diced', 'minced', 'sliced', 'grated', 'shredded',
            'cooked', 'raw', 'fresh', 'frozen', 'dried', 'canned',
            'boneless', 'skinless', 'lean', 'extra virgin', 'organic'
        }
    
    def normalize_ingredient(self, ingredient_text: str) -> str:
        """
        Normalizes an ingredient string to a consistent format for caching.
        
        Examples:
        - "100g chicken breast" -> "chicken_breast"
        - "1 cup of olive oil" -> "olive_oil"
        - "2 medium eggs" -> "egg"
        """
        if not ingredient_text or not ingredient_text.strip():
            return ""
        
        # Try using ingredient-parser-nlp first
        if PARSER_AVAILABLE:
            try:
                parsed = parse_ingredient(ingredient_text)
                if parsed and hasattr(parsed, 'name') and parsed.name:
                    return self._clean_ingredient_name(parsed.name.text)
            except Exception as e:
                logger.debug(f"Parser failed for '{ingredient_text}': {e}, using fallback")
        
        # Fallback to regex-based normalization
        return self._fallback_normalize(ingredient_text)
    
    def _clean_ingredient_name(self, name: str) -> str:
        """Clean and standardize the ingredient name."""
        if not name:
            return ""
        
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove preparation terms
        for prep_term in self.preparation_terms:
            name = re.sub(rf'\b{re.escape(prep_term)}\b', '', name)
        
        # Clean up extra spaces and punctuation
        name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
        name = re.sub(r'\s+', '_', name.strip())  # Replace spaces with underscores
        
        # Handle plurals (simple approach)
        if name.endswith('s') and len(name) > 3:
            singular = name[:-1]
            # Don't remove 's' from words that naturally end in 's'
            if not any(singular.endswith(ending) for ending in ['s', 'us', 'ss']):
                name = singular
        
        return name
    
    def _fallback_normalize(self, ingredient_text: str) -> str:
        """Fallback normalization using regex patterns."""
        text = ingredient_text.lower().strip()
        
        # Remove quantities and units
        # Pattern: number + optional fraction + unit
        quantity_pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:\/|\-)\s*\d+)?\s*(?:' + '|'.join(self.common_units) + r')\b'
        text = re.sub(quantity_pattern, '', text)
        
        # Remove common quantity words
        text = re.sub(r'\b(?:of|a|an|the|some|about|approximately)\b', '', text)
        
        # Remove preparation terms
        for prep_term in self.preparation_terms:
            text = re.sub(rf'\b{re.escape(prep_term)}\b', '', text)
        
        # Clean up and normalize
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', '_', text.strip())  # Replace spaces with underscores
        
        # Handle plurals
        if text.endswith('s') and len(text) > 3:
            singular = text[:-1]
            if not any(singular.endswith(ending) for ending in ['s', 'us', 'ss']):
                text = singular
        
        return text
    
    def get_cache_key(self, ingredient_text: str) -> str:
        """
        Generate a cache key for the ingredient.
        Returns a normalized string suitable for use as a Redis key.
        """
        normalized = self.normalize_ingredient(ingredient_text)
        if not normalized:
            # Fallback for empty normalization
            normalized = re.sub(r'[^\w]', '_', ingredient_text.lower().strip())
        
        return f"ingredient:{normalized}"

# Global normalizer instance
normalizer = IngredientNormalizer()

if __name__ == '__main__':
    # Test the normalizer
    test_cases = [
        "100g chicken breast",
        "1 cup of olive oil", 
        "2 medium eggs",
        "1 tbsp sugar",
        "50g fresh spinach leaves",
        "1 slice of white bread",
        "2 cloves garlic, minced",
        "1 lb ground beef, lean"
    ]
    
    print("Testing Ingredient Normalizer:")
    print("=" * 40)
    
    for test_case in test_cases:
        normalized = normalizer.normalize_ingredient(test_case)
        cache_key = normalizer.get_cache_key(test_case)
        print(f"Input: '{test_case}'")
        print(f"  -> Normalized: '{normalized}'")
        print(f"  -> Cache Key: '{cache_key}'")
        print() 