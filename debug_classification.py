import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "nb" / "src"))

import sqlite3
import asyncio
from nb.src.context_aware_classifier import ContextAwareDietClassifier

async def debug_classification():
    print("=== DEBUGGING CLASSIFICATION LOGIC ===\n")
    
    # Test 1: Check database data for key ingredients
    print("1. Database Check:")
    conn = sqlite3.connect('nb/src/data/knowledge_graph.db')
    cursor = conn.cursor()
    
    test_ingredients = ['chicken breast', 'sugar', 'spinach', 'butter']
    for ingredient in test_ingredients:
        cursor.execute("""
            SELECT name, carbohydrate_g, protein_g, total_fat_g, cholesterol_mg, lactose_g 
            FROM nutrition_facts 
            WHERE name LIKE ? 
            LIMIT 1
        """, (f'%{ingredient}%',))
        
        result = cursor.fetchone()
        if result:
            name, carbs, protein, fat, cholesterol, lactose = result
            print(f"  {ingredient}: {carbs}g carbs, {protein}g protein, {fat}g fat, {cholesterol}mg chol, {lactose}g lactose")
            
            # Manual classification
            is_keto_manual = carbs <= 10.0  # Per 100g
            is_vegan_manual = cholesterol == 0 and lactose == 0 and 'chicken' not in name.lower() and 'butter' not in name.lower()
            print(f"    Manual: Keto={is_keto_manual}, Vegan={is_vegan_manual}")
        else:
            print(f"  {ingredient}: NOT FOUND in database")
    
    conn.close()
    print()
    
    # Test 2: Test the actual classifier
    print("2. Classifier Test:")
    classifier = ContextAwareDietClassifier()
    
    for ingredient in ['chicken breast', 'sugar']:
        print(f"\n--- Testing: {ingredient} ---")
        
        # Get raw context
        context = await classifier._get_context_for_ingredient(ingredient)
        print(f"Raw context: {context[:200]}...")
        
        # Get final classification
        result = await classifier.classify_single_ingredient(ingredient)
        print(f"Final result: {result}")

if __name__ == "__main__":
    asyncio.run(debug_classification()) 