import requests
import json
import os
from pathlib import Path

def load_qwen_model():
    """Load Qwen model into Ollama."""
    # Get absolute path to the model file
    model_path = Path("./qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf").resolve()
    print(f"Qwen model path: {model_path}")
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    # Create modelfile content
    modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER presence_penalty 1.5
PARAMETER num_ctx 4096"""
    
    print("Creating Qwen model...")
    response = requests.post('http://localhost:11434/api/create', 
                            json={'name': 'qwen/qwen3-0.6b-gguf:q8_0', 'modelfile': modelfile_content},
                            stream=True)
    
    success = True
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                if 'status' in data:
                    print(f"Qwen: {data['status']}")
                if 'error' in data:
                    print(f"ERROR: {data['error']}")
                    success = False
            except:
                print(line.decode())
    
    return success

def load_arctic_model():
    """Load Arctic Text2SQL model into Ollama."""
    # Get absolute path to the model file
    model_path = Path("./Arctic-Text2SQL-R1-7B-GGUF/Arctic-Text2SQL-R1-7B.Q4_K_M.gguf").resolve()
    print(f"Arctic model path: {model_path}")
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    # Create modelfile content
    modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop ";"
PARAMETER num_ctx 8192"""
    
    print("Creating Arctic Text2SQL model...")
    response = requests.post('http://localhost:11434/api/create', 
                            json={'name': 'snowflake/arctic-text2sql-r1-7b', 'modelfile': modelfile_content},
                            stream=True)
    
    success = True
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                if 'status' in data:
                    print(f"Arctic: {data['status']}")
                if 'error' in data:
                    print(f"ERROR: {data['error']}")
                    success = False
            except:
                print(line.decode())
    
    return success

def list_models():
    """List available models in Ollama."""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable models:")
            for model in models.get('models', []):
                print(f"  - {model.get('name', 'Unknown')}")
        else:
            print(f"Failed to list models: {response.status_code}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    print("Loading models into Ollama...")
    
    # Change to models directory
    os.chdir(Path(__file__).parent)
    
    # Load models
    qwen_success = load_qwen_model()
    arctic_success = load_arctic_model()
    
    # List available models
    list_models()
    
    if qwen_success and arctic_success:
        print("\nSUCCESS: Both models loaded successfully!")
    else:
        print("\nWARNING: Some models failed to load") 
    list_models()
    
    if qwen_success and arctic_success:
        print("\nSUCCESS: Both models loaded successfully!")
    else:
        print("\nWARNING: Some models failed to load") 