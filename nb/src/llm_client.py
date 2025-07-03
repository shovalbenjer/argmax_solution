"""
Unified LLM Client for Ollama Integration

This module provides a consistent interface for interacting with Ollama-based
language models across the diet classification system. It supports both
synchronous and asynchronous operations with proper error handling and
JSON response parsing.

The LLMClient class abstracts the complexity of Ollama API interactions,
providing a simple interface for model queries, model discovery, and
connection management. It includes automatic fallback mechanisms when
Ollama is unavailable.

Key Features:
- Synchronous and asynchronous query support
- Automatic JSON parsing and validation
- Model availability checking and discovery
- Connection health monitoring
- Graceful error handling and logging

Example:
    >>> from llm_client import llm_client
    >>> response = llm_client.query("qwen:latest", "What is 2+2?", as_json=False)
    >>> models = llm_client.list_models()
    >>> is_available = llm_client.is_model_available("qwen:latest")
"""
import asyncio
import json
from typing import Dict, Any, Optional
import logging
import os
import glob

try:
    import ollama
except ImportError:
    logging.warning("Ollama not available, LLM functionality will be limited")
    ollama = None

from config import app_config

# Configure professional logging
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified client for Ollama LLM interactions.
    
    This class provides a high-level interface for communicating with Ollama
    language models. It handles connection management, model discovery,
    and provides both synchronous and asynchronous query capabilities.
    
    The client supports automatic JSON parsing, model availability checking,
    and includes comprehensive error handling for production use.
    
    Attributes:
        host: Ollama service URL
        client: Synchronous Ollama client instance
        async_client: Asynchronous Ollama client instance
        
    Example:
        >>> client = LLMClient("http://localhost:11434")
        >>> response = client.query("qwen:latest", "Classify this ingredient")
        >>> async_response = await client.query_async("qwen:latest", "Analyze recipe")
    """
    
    def __init__(self, host: Optional[str] = None):
        """
        Initialize the LLM client with Ollama connection.
        
        Args:
            host: Ollama service URL (defaults to config value)
            
        Note:
            If Ollama is not available, the client will operate in
            limited mode with appropriate warning messages.
        """
        self.host = host or app_config.OLLAMA_URL
        self.client = None
        self.async_client = None
        self._init_clients()
    
    def _init_clients(self):
        """
        Initialize Ollama client connections.
        
        Creates both synchronous and asynchronous client instances
        with proper error handling. If initialization fails, the
        system continues operating with limited functionality.
        
        Raises:
            None: All exceptions are caught and logged
        """
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
        """
        List available models from Ollama service.
        
        Retrieves a comprehensive list of all models available on the
        Ollama server, including model names, sizes, and metadata.
        
        Returns:
            List of available model information dictionaries
            
        Example:
            >>> models = llm_client.list_models()
            >>> for model in models:
            ...     print(f"Model: {model.get('name', 'Unknown')}")
        """
        if not self.client:
            return []
        
        try:
            models = self.client.list()
            return models.get('models', [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def query(self, model: str, prompt: str, as_json: bool = True) -> Dict[str, Any]:
        """
        Execute synchronous query to Ollama model.
        
        Sends a prompt to the specified model and returns the response.
        Supports both plain text and JSON-formatted responses based on
        the as_json parameter.
        
        Args:
            model: Name of the Ollama model to query
            prompt: Input prompt for the model
            as_json: Whether to request JSON-formatted response (default: True)
            
        Returns:
            Dict containing response data or error information
            
        Example:
            >>> response = llm_client.query("qwen:latest", "What is keto?", as_json=True)
            >>> if "error" not in response:
            ...     print(response.get("content", "No content"))
        """
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
        """
        Execute asynchronous query to Ollama model.
        
        Sends a prompt to the specified model asynchronously and returns
        the response. This method is suitable for high-throughput scenarios
        and integration with async frameworks.
        
        Args:
            model: Name of the Ollama model to query
            prompt: Input prompt for the model
            as_json: Whether to request JSON-formatted response (default: True)
            
        Returns:
            Dict containing response data or error information
            
        Example:
            >>> response = await llm_client.query_async("qwen:latest", "Analyze this")
            >>> if "error" not in response:
            ...     result = response.get("content")
        """
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
        """
        Check if a specific model is available on the Ollama server.
        
        Performs both exact match and partial name matching to determine
        if the requested model is available. This is useful for model
        selection and fallback strategies.
        
        Args:
            model_name: Name of the model to check for availability
            
        Returns:
            bool: True if model is available, False otherwise
            
        Example:
            >>> if llm_client.is_model_available("qwen:latest"):
            ...     response = llm_client.query("qwen:latest", "Hello")
            ... else:
            ...     print("Model not available")
        """
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
        """
        Get the recommended model for classification tasks.
        
        Returns the best available model from the models present in the models directory.
        Falls back to the first available model if none of the preferred ones are found.
        
        Returns:
            str: Name of the recommended model or None if no models available
            
        Example:
            >>> recommended = llm_client.get_recommended_model()
            >>> if recommended:
            ...     response = llm_client.query(recommended, "Classify this")
        """
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        preferred_models = []
        if os.path.isdir(models_dir):
            for entry in os.listdir(models_dir):
                entry_path = os.path.join(models_dir, entry)
                if os.path.isdir(entry_path):
                    gguf_files = glob.glob(os.path.join(entry_path, '*.gguf'))
                    if gguf_files:
                        preferred_models.append(entry)
        
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