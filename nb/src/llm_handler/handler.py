import sys
from pathlib import Path
from loguru import logger
import json
import asyncio

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.llm_client import llm_client, LLMClient

# Legacy LLMHandler class - now using shared LLM client
class LLMHandler:
    """Legacy wrapper for shared LLM client."""
    
    def __init__(self, host: str = "http://ollama:11434"):
        """Initialize with shared LLM client."""
        self.client = LLMClient(host)
        logger.info(f"LLM Handler initialized using shared client for {host}")

    async def list_models(self):
        """List available models."""
        return self.client.list_models()

    async def async_query(self, model: str, prompt: str, as_json: bool = True):
        """Query model asynchronously."""
        return await self.client.query_async(model, prompt, as_json)

# Example usage (for testing)
async def main_test():
    """An example function to test the LLMHandler.

    Detailed Description:
        - This function demonstrates how to use the `LLMHandler`.
        - It initializes the handler, lists the available models on the server,
          and then sends a sample query to a specified model.
        - This is intended for development and testing purposes to ensure the handler
          is working correctly.
    """
    handler = LLMHandler(host="http://localhost:11434")
    available_models = await handler.list_models()
    logger.info(f"Available models: {available_models}")

    if available_models:
        model_to_use = "qwen:latest" # Ensure you have a model like qwen running
        test_prompt = "Why is the sky blue? Respond in valid JSON."
        result = await handler.async_query(model=model_to_use, prompt=test_prompt)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main_test()) 