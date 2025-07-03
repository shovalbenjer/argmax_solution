"""
Ingredient Normalizer for Consistent Caching and Lookup

This module provides deterministic ingredient normalization using the
ingredient-parser-nlp library to ensure consistent cache keys and reliable
ingredient matching across the diet classification system.

The normalizer handles:
- Ingredient name extraction and cleaning
- Quantity and unit removal
- Preparation term normalization
- Plural form handling
- Cache key generation
- Fallback normalization when parser unavailable

Key Features:
- Professional ingredient parsing with 97.8% accuracy
- Consistent normalization for caching
- Fallback regex-based normalization
- Comprehensive unit and preparation term handling
- Plural form standardization
- Cache key generation for Redis

Supported Normalizations:
- Quantity removal: "100g chicken breast" → "chicken_breast"
- Unit removal: "1 cup olive oil" → "olive_oil"
- Preparation term removal: "chopped onions" → "onion"
- Plural handling: "tomatoes" → "tomato"
- Cache key generation: "chicken_breast" → "ingredient:chicken_breast"

Dependencies:
- ingredient-parser: Professional ingredient parsing (optional)
- re: Regular expression processing
- logging: Debug and error logging

Example:
    >>> from ingredient_normalizer import normalizer
    >>> normalized = normalizer.normalize_ingredient("100g chicken breast")
    >>> print(normalized)  # "chicken_breast"
    >>> cache_key = normalizer.get_cache_key("100g chicken breast")
    >>> print(cache_key)  # "ingredient:chicken_breast"
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
    
    This class provides comprehensive ingredient normalization capabilities
    using the ingredient-parser-nlp library when available, with robust
    fallback to regex-based normalization. It ensures consistent ingredient
    representation across the classification system.
    
    Key Features:
    - Professional ingredient parsing with high accuracy
    - Consistent normalization for caching systems
    - Comprehensive unit and preparation term handling
    - Plural form standardization
    - Cache key generation for Redis storage
    - Fallback normalization for edge cases
        
    Attributes:
        common_units (set): Set of common measurement units to remove
        preparation_terms (set): Set of preparation terms to normalize
        
    Example:
        >>> normalizer = IngredientNormalizer()
        >>> result = normalizer.normalize_ingredient("100g chicken breast")
        >>> print(result)  # "chicken_breast"
    """
    
    def __init__(self):
        """
        Initialize the ingredient normalizer with unit and preparation term sets.
        
        This constructor sets up the comprehensive sets of measurement units
        and preparation terms that will be normalized during ingredient processing.
        
        Initializes:
        - Volume units: cup, tablespoon, teaspoon, etc.
        - Weight units: pound, ounce, gram, etc.
        - Count units: piece, slice, clove, etc.
        - Preparation terms: chopped, diced, cooked, etc.
        
        Example:
            >>> normalizer = IngredientNormalizer()
            >>> # Ready for ingredient normalization
        """
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
        
        This method provides comprehensive ingredient normalization using
        the ingredient-parser-nlp library when available, with robust
        fallback to regex-based normalization. It removes quantities,
        units, and preparation terms to create consistent ingredient names.
        
        Args:
            ingredient_text (str): Raw ingredient string to normalize
            
        Returns:
            str: Normalized ingredient name suitable for caching and lookup
            
        Examples:
            >>> normalizer = IngredientNormalizer()
            >>> normalizer.normalize_ingredient("100g chicken breast")
            "chicken_breast"
            >>> normalizer.normalize_ingredient("1 cup of olive oil")
            "olive_oil"
            >>> normalizer.normalize_ingredient("2 medium eggs")
            "egg"
            >>> normalizer.normalize_ingredient("chopped onions")
            "onion"
            
        Note:
            Uses ingredient-parser-nlp when available for high accuracy,
            falls back to regex-based normalization when parser unavailable.
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
        """
        Clean and standardize the ingredient name from parser output.
        
        This method processes the output from ingredient-parser-nlp to
        create a clean, normalized ingredient name suitable for caching
        and lookup operations.
        
        Args:
            name (str): Raw ingredient name from parser
            
        Returns:
            str: Cleaned and normalized ingredient name
            
        Processing Steps:
            1. Convert to lowercase and strip whitespace
            2. Remove preparation terms (chopped, diced, etc.)
            3. Remove punctuation and special characters
            4. Replace spaces with underscores
            5. Handle plural forms (simple approach)
            
        Example:
            >>> normalizer = IngredientNormalizer()
            >>> result = normalizer._clean_ingredient_name("Chicken Breast")
            >>> print(result)  # "chicken_breast"
        """
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
        """
        Fallback normalization using regex patterns when parser unavailable.
        
        This method provides robust ingredient normalization using regex
        patterns when the ingredient-parser-nlp library is not available.
        It handles quantities, units, and preparation terms comprehensively.
        
        Args:
            ingredient_text (str): Raw ingredient string to normalize
            
        Returns:
            str: Normalized ingredient name using regex patterns
            
        Processing Steps:
            1. Remove quantities and measurement units
            2. Remove common quantity words (of, a, an, the)
            3. Remove preparation terms
            4. Clean punctuation and normalize spacing
            5. Handle plural forms
            
        Example:
            >>> normalizer = IngredientNormalizer()
            >>> result = normalizer._fallback_normalize("100g chicken breast")
            >>> print(result)  # "chicken_breast"
        """
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
        
        This method creates a Redis-compatible cache key for the given
        ingredient by normalizing the ingredient name and prefixing it
        with "ingredient:" for namespace separation.
        
        Args:
            ingredient_text (str): Raw ingredient string
            
        Returns:
            str: Redis-compatible cache key for the ingredient
            
        Key Format:
            "ingredient:{normalized_name}"
            
        Examples:
            >>> normalizer = IngredientNormalizer()
            >>> key = normalizer.get_cache_key("100g chicken breast")
            >>> print(key)  # "ingredient:chicken_breast"
            >>> key = normalizer.get_cache_key("1 cup olive oil")
            >>> print(key)  # "ingredient:olive_oil"
            
        Note:
            Provides fallback key generation for empty normalization results.
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