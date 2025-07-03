#!/usr/bin/env python3
"""Edge Cases Test Suite"""

import pytest
import asyncio
from context_aware_classifier import SOTASemanticClassifier

def test_ingredient_parsing():
    """Test ingredient parsing."""
    classifier = SOTASemanticClassifier()
    result = classifier.extract_ingredient_name("3 pounds pork shoulder, cut into chunks")
    assert "pork" in result.lower()

@pytest.mark.asyncio
async def test_classification():
    """Test classification."""
    classifier = SOTASemanticClassifier()
    result = await classifier.classify_single_ingredient("spinach")
    assert "is_keto" in result
    assert "is_vegan" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 