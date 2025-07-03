#!/usr/bin/env python3
"""
Quick test of the SOTA semantic classifier system
"""

import asyncio
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def quick_test():
    """Run a quick test."""
    print("🚀 Testing SOTA Semantic Classifier")
    
    try:
        from context_aware_classifier import SOTASemanticClassifier
        classifier = SOTASemanticClassifier()
        
        # Test basic functionality
        result = await classifier.classify_single_ingredient("spinach")
        print(f"✅ Spinach classification: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test()) 