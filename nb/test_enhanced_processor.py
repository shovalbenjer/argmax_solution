#!/usr/bin/env python3
"""
Test script for the enhanced ingredient processor.
"""

from src.ingredient_processor import processor
import json

def test_enhanced_processor():
    """Test the enhanced ingredient processor functionality."""
    print("Testing Enhanced Ingredient Processor")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "100g chicken breast",
        "1 cup olive oil", 
        "2 medium eggs",
        "1 tbsp sugar",
        "fresh spinach leaves"
    ]
    
    for test_case in test_cases:
        print(f"\nProcessing: '{test_case}'")
        
        # Test normalization
        normalized = processor.normalize_ingredient(test_case)
        cache_key = processor.get_cache_key(test_case)
        
        print(f"  Normalized: '{normalized}'")
        print(f"  Cache Key: '{cache_key}'")
        
        # Test comprehensive processing
        try:
            result = processor.process_ingredient_comprehensive(test_case)
            print(f"  Match Type: {result['match_type']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            
            if result['nutrition_data']:
                nutrition = result['nutrition_data']
                calories = nutrition.get('calories', 'N/A')
                protein = nutrition.get('protein', 'N/A')
                print(f"  Nutrition: {calories} cal, {protein}g protein")
            
            if result['vegan_info']:
                vegan = result['vegan_info']
                status = vegan.get('vegan_status', 'Unknown')
                print(f"  Vegan Status: {status}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_enhanced_processor() 