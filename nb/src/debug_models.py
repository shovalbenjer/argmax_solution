import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_client import LLMClient
import json

def debug_model_detection():
    """Debug model detection issues."""
    print("Debugging Model Detection")
    print("=" * 30)
    
    client = LLMClient()
    
    # Get raw model list
    models = client.list_models()
    print(f"\nRaw models response: {models}")
    print(f"Type: {type(models)}")
    print(f"Length: {len(models) if models else 'None'}")
    
    if models:
        print(f"\nModel details:")
        for i, model in enumerate(models):
            print(f"  {i}: {model}")
            print(f"      Type: {type(model)}")
            if isinstance(model, dict):
                print(f"      Keys: {list(model.keys())}")
                print(f"      Name: {model.get('name', 'NO NAME')}")
    
    # Test specific model checks
    test_models = [
        "snowflake/arctic-text2sql-r1-7b:latest",
        "snowflake/arctic-text2sql-r1-7b",
        "qwen/qwen3-0.6b-gguf:q8_0",
        "qwen/qwen3-0.6b-gguf"
    ]
    
    print(f"\nTesting model availability:")
    for model_name in test_models:
        available = client.is_model_available(model_name)
        print(f"  {model_name}: {available}")
        
        # Manual check
        if models:
            manual_check = any(model.get('name', '') == model_name for model in models)
            startswith_check = any(model.get('name', '').startswith(model_name) for model in models)
            print(f"    Manual exact match: {manual_check}")
            print(f"    Startswith check: {startswith_check}")

if __name__ == "__main__":
    debug_model_detection() 