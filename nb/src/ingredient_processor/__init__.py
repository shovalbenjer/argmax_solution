"""
Enhanced Ingredient Processor Module

Provides comprehensive ingredient processing with normalization, fuzzy matching,
and database access optimized for caching and performance.
"""

from .processor import (
    EnhancedIngredientProcessor,
    processor,
    get_all_ingredient_names,
    get_ingredient_names_cached,
    get_nutrition_data_batch,
    get_nutrition_data,
    get_vegan_info,
    get_context_with_rapidfuzz_fallback
)

__all__ = [
    'EnhancedIngredientProcessor',
    'processor',
    'get_all_ingredient_names',
    'get_ingredient_names_cached',
    'get_nutrition_data_batch',
    'get_nutrition_data',
    'get_vegan_info',
    'get_context_with_rapidfuzz_fallback'
]
