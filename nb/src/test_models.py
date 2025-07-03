import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_client import LLMClient

async def test_model_availability():
    """Test if models are available and working."""
    print("Testing Model Availability and Basic Functionality")
    print("=" * 50)
    
    client = LLMClient()
    
    # Test 1: List available models
    print("\n1. Available Models:")
    models = client.list_models()
    for model in models:
        print(f"   - {model.get('name', 'Unknown')}")
    
    # Test 2: Check Arctic model availability
    arctic_name = "snowflake/arctic-text2sql-r1-7b:latest"
    print(f"\n2. Arctic Model Check:")
    print(f"   Model name: {arctic_name}")
    print(f"   Available: {client.is_model_available(arctic_name)}")
    
    # Test 3: Check Qwen model availability
    qwen_name = "qwen/qwen3-0.6b-gguf:q8_0"
    print(f"\n3. Qwen Model Check:")
    print(f"   Model name: {qwen_name}")
    print(f"   Available: {client.is_model_available(qwen_name)}")
    
    # Test 4: Simple Arctic query
    print(f"\n4. Arctic Text2SQL Test:")
    if client.is_model_available(arctic_name):
        try:
            result = await client.query_async(
                arctic_name,
                "SELECT name FROM nutrition_facts WHERE name LIKE '%chicken%' LIMIT 1",
                as_json=False
            )
            print(f"   Arctic response: {result.get('content', 'No content')[:100]}...")
        except Exception as e:
            print(f"   Arctic error: {e}")
    else:
        print("   Arctic model not available for testing")
    
    # Test 5: Simple Qwen query
    print(f"\n5. Qwen Classification Test:")
    if client.is_model_available(qwen_name):
        try:
            result = await client.query_async(
                qwen_name,
                "Is chicken breast keto-friendly? Answer with JSON: {\"is_keto\": boolean, \"reasoning\": \"text\"}",
                as_json=True
            )
            print(f"   Qwen response: {result}")
        except Exception as e:
            print(f"   Qwen error: {e}")
    else:
        print("   Qwen model not available for testing")

if __name__ == "__main__":
    asyncio.run(test_model_availability()) 