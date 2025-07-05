"""
Diet Classifiers for Flask Web Application

This file will be copied from nb/src/diet_classifiers.py after the implementation
is completed and tested. For now, it contains placeholder implementations.

The Flask app requires fast, synchronous classification for real-time web responses.
"""

import json
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_ingredient_keto(ingredient: str) -> bool:
    """
    Placeholder implementation for web app.
    Will be replaced with the full Arctic -> Qwen pipeline from nb/src/
    """
    # TODO: Copy implementation from nb/src/diet_classifiers.py
    logger.warning("Using placeholder keto classification")
    return False


def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Placeholder implementation for web app.
    Will be replaced with the full Arctic -> Qwen pipeline from nb/src/
    """
    # TODO: Copy implementation from nb/src/diet_classifiers.py
    logger.warning("Using placeholder vegan classification")
    return False


def is_keto(ingredients: List[str]) -> bool:
    """Check if recipe is keto-friendly (all ingredients must be keto)."""
    return all(map(is_ingredient_keto, ingredients))


def is_vegan(ingredients: List[str]) -> bool:
    """Check if recipe is vegan-friendly (all ingredients must be vegan)."""
    return all(map(is_ingredient_vegan, ingredients))
