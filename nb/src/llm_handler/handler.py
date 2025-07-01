import ollama
from loguru import logger
import json
import asyncio

class LLMHandler:
    def __init__(self, host: str = "http://ollama:11434"):
        self.client = ollama.AsyncClient(host=host)
        logger.info(f"LLM Handler initialized, connected to Ollama at {host}")

    async def list_models(self):
        """Lists available models in Ollama."""
        try:
            models = await self.client.list()
            return models['models']
        except Exception as e:
            logger.error(f"Failed to list models from Ollama: {e}")
            return []

    async def async_query(self, model: str, prompt: str, as_json: bool = True):
        """
        Sends a prompt asynchronously to a specified model in Ollama.
        """
        logger.debug(f"Querying model '{model}' with prompt: {prompt[:100]}...")
        try:
            response = await self.client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' if as_json else ''
            )
            content = response['message']['content']
            logger.debug(f"Received response: {content[:100]}...")
            
            if as_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode LLM response as JSON: {content}")
                    return {"error": "Invalid JSON response from model"}

            return content
        except Exception as e:
            logger.error(f"Error querying Ollama model '{model}': {e}")
            return {"error": str(e)}

# Example usage (for testing)
async def main_test():
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