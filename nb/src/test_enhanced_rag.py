import asyncio
from context_aware_classifier import ContextAwareDietClassifier

async def test_enhanced_rag_pipeline():
    """Test the enhanced RAG pipeline with various ingredients."""
    classifier = ContextAwareDietClassifier()
    
    print("Testing Enhanced RAG Pipeline with Fallback Mechanism")
    print("=" * 55)
    
    # Test ingredients with different dietary characteristics
    test_cases = [
        ("chicken breast", "Expected: keto=True, vegan=False (animal protein, low carbs)"),
        ("spinach", "Expected: keto=True, vegan=True (leafy green, very low carbs)"),
        ("sugar", "Expected: keto=False, vegan=True (pure carbs, plant-based)"),
        ("olive oil", "Expected: keto=True, vegan=True (pure fat, plant-based)"),
        ("bread", "Expected: keto=False, vegan=depends (high carbs, may contain eggs/milk)")
    ]
    
    for ingredient, expected in test_cases:
        print(f"\n--- Testing: {ingredient} ---")
        print(f"Expected: {expected}")
        
        result = await classifier.classify_single_ingredient(ingredient)
        
        print(f"Result: keto={result.get('is_keto')}, vegan={result.get('is_vegan')}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        print(f"Method: {result.get('method', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_rag_pipeline()) 