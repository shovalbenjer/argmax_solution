import asyncio
from context_aware_classifier import SOTASemanticClassifier

async def test():
    classifier = SOTASemanticClassifier()
    result = await classifier.classify_single_ingredient("spinach")
    print(result)

asyncio.run(test()) 