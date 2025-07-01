import ollama
from loguru import logger
import json
import asyncio

class LLMHandler:
    """A client handler for interacting with a local Ollama server.

    Detailed Description:
        - This class provides a dedicated interface for communicating with an Ollama server,
          which hosts and serves local large language models.
        - It uses an asynchronous client to allow for non-blocking communication with the API.

    Libraries Used:
        - ollama: The official Python library for Ollama, used to interact with the server.
        - loguru: For logging information and errors.
        - json: For decoding JSON responses from the model.
        - asyncio: The foundation for the asynchronous methods.
    """
    def __init__(self, host: str = "http://ollama:11434"):
        """Initializes the asynchronous Ollama client.

        Parameters:
            - host (str): The URL of the Ollama server.
        """
        self.client = ollama.AsyncClient(host=host)
        logger.info(f"LLM Handler initialized, connected to Ollama at {host}")

    async def list_models(self):
        """Fetches the list of available models from the Ollama server.

        Detailed Description:
            - This asynchronous method sends a request to the Ollama API to list all
              currently available models that can be queried.

        Returns:
            - list: A list of model information dictionaries, or an empty list if the API call fails.
        """
        try:
            models = await self.client.list()
            return models['models']
        except Exception as e:
            logger.error(f"Failed to list models from Ollama: {e}")
            return []

    async def async_query(self, model: str, prompt: str, as_json: bool = True):
        """Sends a prompt asynchronously to a specified model and gets a structured response.

        Detailed Description:
            - This is the primary method for querying a model on the Ollama server.
            - It sends the user's prompt to the specified model.
            - It includes a `format='json'` parameter in the API call if `as_json` is True,
              which instructs capable models to return a valid JSON string.
            - It includes robust error handling for both the API call itself and for potential
              JSON decoding errors if the model fails to produce valid JSON.

        Parameters:
            - model (str): The name of the model to query (e.g., "qwen:latest").
            - prompt (str): The user's input prompt.
            - as_json (bool): If True, requests a JSON-formatted response from the model.

        Returns:
            - dict | str: A dictionary if `as_json` is True and the response is valid JSON,
              a plain string if `as_json` is False, or a dictionary with an "error" key if any step fails.
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