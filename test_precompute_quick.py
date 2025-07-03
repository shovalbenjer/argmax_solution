#!/usr/bin/env python3
"""Quick test of pre-computation pipeline"""

import sys
import time
from pathlib import Path

# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "nb" / "src"))

from diet_classifiers import is_keto, is_vegan

def test_quick():
    """Test pre-computation on sample ingredients."""
    print("🧪 Testing SOTA Classifier")
    
    test_ingredients = ["spinach", "chicken breast", "olive oil"]
    
    for ingredient in test_ingredients:
        print(f"\nTesting: {ingredient}")
        
        start_time = time.time()
        try:
            keto_result = is_keto([ingredient])
            vegan_result = is_vegan([ingredient])
            end_time = time.time()
            
            print(f"  Keto: {keto_result}, Vegan: {vegan_result}")
            print(f"  Time: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_quick() 