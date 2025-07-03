import requests
import json
from pathlib import Path

def debug_model_creation():
    """Debug the model creation process."""
    # Get absolute path to the model file
    model_path = Path("./qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf").resolve()
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    # Simple modelfile
    modelfile_content = f"FROM {model_path}"
    
    print(f"\nModelfile content:")
    print(repr(modelfile_content))
    
    # Create payload
    payload = {
        'name': 'qwen-test', 
        'modelfile': modelfile_content
    }
    
    print(f"\nPayload:")
    print(json.dumps(payload, indent=2))
    
    # Send request
    print(f"\nSending request...")
    response = requests.post('http://localhost:11434/api/create', json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    debug_model_creation() 