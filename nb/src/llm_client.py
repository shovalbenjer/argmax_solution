"""
Unified LLM Client for Ollama Integration

Provides consistent LLM access across all services, replacing mock implementations.
"""
import asyncio
import json
from typing import Dict, Any, Optional
import logging

try:
    import ollama
except ImportError:
    logging.warning("Ollama not available, LLM functionality will be limited")
    ollama = None

from config import app_config

# Configure professional logging
logger = logging.getLogger(__name__)

class LLMClient:
    """Unified client for Ollama LLM interactions."""
    
    def __init__(self, host: Optional[str] = None):
        self.host = host or app_config.OLLAMA_URL
        self.client = None
        self.async_client = None
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Ollama clients."""
        if not ollama:
            logger.warning("Ollama not available, LLM features disabled")
            return
        
        try:
            self.client = ollama.Client(host=self.host)
            self.async_client = ollama.AsyncClient(host=self.host)
            logger.info(f"LLM client initialized for {self.host}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama clients: {e}")
    
    def list_models(self) -> list:
        """List available models from Ollama."""
        if not self.client:
            return []
        
        try:
            models = self.client.list()
            return models.get('models', [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def query(self, model: str, prompt: str, as_json: bool = True) -> Dict[str, Any]:
        """Synchronous query to Ollama model."""
        if not self.client:
            return {"error": "Ollama client not available"}
        
        try:
            response = self.client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' if as_json else ''
            )
            
            content = response['message']['content']
            
            if as_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response: {content}")
                    return {"error": "Invalid JSON response", "raw_content": content}
            
            return {"content": content}
            
        except Exception as e:
            logger.error(f"Query failed for model {model}: {e}")
            return {"error": str(e)}
    
    async def query_async(self, model: str, prompt: str, as_json: bool = True) -> Dict[str, Any]:
        """Asynchronous query to Ollama model."""
        if not self.async_client:
            return {"error": "Ollama async client not available"}
        
        try:
            response = await self.async_client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' if as_json else ''
            )
            
            content = response['message']['content']
            
            if as_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response: {content}")
                    return {"error": "Invalid JSON response", "raw_content": content}
            
            return {"content": content}
            
        except Exception as e:
            logger.error(f"Async query failed for model {model}: {e}")
            return {"error": str(e)}
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        models = self.list_models()
        if not models:
            return False
        
        # Check for exact match first
        for model in models:
            # Handle both dict format (API response) and Model object format (Python client)
            if hasattr(model, 'model'):
                model_id = model.model
            elif isinstance(model, dict):
                model_id = model.get('name', '') or model.get('model', '')
            else:
                continue
                
            if model_id == model_name:
                return True
        
        # Check for startswith match (for partial names)
        for model in models:
            # Handle both dict format (API response) and Model object format (Python client)
            if hasattr(model, 'model'):
                model_id = model.model
            elif isinstance(model, dict):
                model_id = model.get('name', '') or model.get('model', '')
            else:
                continue
                
            if model_id.startswith(model_name):
                return True
        
        return False
    
    def get_recommended_model(self) -> Optional[str]:
        """Get the recommended model for classification tasks."""
        preferred_models = ['qwen:latest', 'gemma:latest', 'llama2:latest']
        
        models = self.list_models()
        available_models = []
        for model in models:
            # Handle both dict format (API response) and Model object format (Python client)
            if hasattr(model, 'model'):
                model_name = model.model
            elif isinstance(model, dict):
                model_name = model.get('name', '') or model.get('model', '')
            else:
                continue
                
            if model_name:
                available_models.append(model_name)
        
        for preferred in preferred_models:
            for available in available_models:
                if available.startswith(preferred.split(':')[0]):
                    return available
        
        # Return first available model if none of the preferred ones are found
        return available_models[0] if available_models else None

# Global LLM client instance
llm_client = LLMClient() 