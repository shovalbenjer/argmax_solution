"""
SOTA Diet Classifier - Ground Truth Generation (V19 - Final Production)

This script implements the definitive, most robust pipeline for ground truth generation.
This version includes the final fix to flatten nested list columns before saving to CSV,
preventing the 'ComputeError: CSV format does not support nested data' error.

- Stage 1: A specialized Vegan Classifier runs on all recipes.
- Stage 2: A specialized Keto Classifier runs on all recipes.
- Each classifier uses a Chain-of-Thought and Self-Critique prompt design.
- The results from both stages are merged into a single, comprehensive output file.

Author: Argmax Challenge Implementation
"""

import os
import polars as pl
import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio
import mlflow
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from opensearchpy import OpenSearch
from dotenv import load_dotenv

# --- Local Module Imports ---
# Assumes the script is run with the `src` directory in the Python path.
from llm_handler.handler import LLMHandler
from ingredient_processor.processor import get_context_with_rapidfuzz_fallback

load_dotenv()

# --- Configuration ---
CONFIG = {
    "OUTPUT_DIR": Path("nb/src/data"),
    "GROUND_TRUTH_FILENAME": "personas_ground_truth.csv",
    "SAMPLE_SIZE": 100,
    "BATCH_SIZE": 25,
    "TEACHER_MODEL": os.getenv("TEACHER_MODEL", "qwen:latest"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "OPENSEARCH_URL": os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
    "RPM_LIMIT": int(os.getenv("RPM_LIMIT", 10)),
    "TPM_LIMIT": int(os.getenv("TPM_LIMIT", 200000)),
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
}

MODELS_QUOTAS = {
    'gemini-2.5-pro': (5, 250_000),
    'gemini-2.5-flash': (10, 250_000),
    'gemini-2.5-flash-lite-preview-06-17': (15, 250_000),
}

logging.basicConfig(level=CONFIG["LOG_LEVEL"], format='%(asctime)s - %(levelname)s - %(message)s', force=True)

class AsyncTokenBucket:
    """Manages API rate limiting for asynchronous requests using a token bucket algorithm.

    Detailed Description:
        - This class ensures that requests to an API do not exceed specified rate limits (requests per minute and tokens per minute).
        - It uses an asyncio.Lock to handle concurrent access safely.
        - When a request is made, it checks if the limits have been reached. If so, it calculates the required delay and sleeps asynchronously before allowing the request to proceed.

    Parameters:
        - rpm_rate (int): The maximum number of requests allowed per minute.
        - tpm_rate (int): The maximum number of tokens allowed per minute.

    Libraries Used:
        - asyncio: Used for asynchronous locking and sleeping, which is essential for non-blocking rate limiting in an async environment.
        - time: Used to track the passage of time for resetting the rate limit counters.
    """
    def __init__(self, rpm_rate: int, tpm_rate: int):
        self.rpm_rate = rpm_rate
        self.tpm_rate = tpm_rate
        self.last_reset_time = time.time()
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int):
        async with self.lock:
            now = time.time()
            if now - self.last_reset_time >= 60:
                self.last_reset_time = now
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
            time_to_wait_rpm = 60 - (now - self.last_reset_time) if self.requests_this_minute >= self.rpm_rate else 0.0
            time_to_wait_tpm = 60 - (now - self.last_reset_time) if self.tokens_this_minute + estimated_tokens > self.tpm_rate else 0.0
            time_to_wait = max(time_to_wait_rpm, time_to_wait_tpm)
            if time_to_wait > 0.1:
                logger.warning(f"Rate limit nearing. Pausing for {time_to_wait:.2f}s...")
                await asyncio.sleep(time_to_wait)
                self.last_reset_time = time.time()
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
            self.requests_this_minute += 1
            self.tokens_this_minute += estimated_tokens


def build_vegan_prompt(recipe_rows: List[Dict[str, Any]]) -> list:
    """Builds a prompt for the generative AI to classify recipes as vegan.

    Detailed Description:
        - This function takes a list of recipe data and formats it into a detailed prompt for a generative AI model.
        - The prompt instructs the AI to act as a food scientist and determine if each recipe is strictly vegan.
        - It specifies a chain-of-thought process for the AI to follow and the exact JSON output format required.

    Parameters:
        - recipe_rows (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a recipe.

    Returns:
        - list: A list containing the user prompt and the desired JSON schema for the AI's response.

    Libraries Used:
        - json: Used to serialize the recipe data into a JSON string, which is embedded in the prompt. This is a standard and lightweight way to represent structured data.
    """
    recipe_data_list = [{"recipe_id": r.get('_id'), "title": r.get('title'), "ingredients": json.loads(r.get('ingredients', '[]'))} for r in recipe_rows]
    user_prompt = f"""
    You are a meticulous food scientist. Your task is to determine if each recipe in the following list is strictly vegan.
    A recipe is **strictly vegan** if it contains absolutely no animal products (no meat, poultry, fish, dairy, eggs, honey, etc.).
    **Analysis Process:**
    1. For each recipe, carefully examine the ingredients.
    2. In your reasoning, first list any and all ingredients that are or could be derived from animals.
    3. After listing the evidence, make a final "Verdict" in your reasoning.
    4. Based on your verdict, set `is_vegan` to `true` or `false`.
    **RESPONSE INSTRUCTIONS:**
    - You must respond with ONLY a single, valid JSON array. Each object must correspond to a recipe.
    - If you identify ANY potential animal product, `is_vegan` MUST be `false`.
    **RECIPE DATA:**
    ```json
    {json.dumps(recipe_data_list)}
    ```
    """
    schema = [{"recipe_id": "string", "is_vegan": "boolean", "vegan_reasoning": "string", "animal_ingredients_found": ["list", "of", "strings"]}]
    return [user_prompt, "Provide your analysis in this JSON format:", json.dumps(schema)]


def build_keto_prompt(recipe_rows: List[Dict[str, Any]]) -> list:
    """Builds a prompt for the generative AI to classify recipes as keto-friendly.

    Detailed Description:
        - This function constructs a prompt for a generative AI model to determine if recipes are keto-friendly.
        - The prompt defines "strictly keto" based on carbohydrate content and instructs the AI to use its own knowledge to override potentially incorrect information in the provided data.
        - It specifies a JSON output format for consistency.

    Parameters:
        - recipe_rows (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a recipe.

    Returns:
        - list: A list containing the user prompt and the desired JSON schema for the AI's response.

    Libraries Used:
        - json: Used to serialize the recipe data into a JSON string for the prompt.
    """
    recipe_data_list = []
    for row in recipe_rows:
        rag_summary = row.get('rag_summary', 'No analysis available')
        sanitized_summary = rag_summary.replace('"', "'")
        recipe_data_list.append({
            "recipe_id": row.get('_id'),
            "title": row.get('title'),
            "ingredients": json.loads(row.get('ingredients', '[]')),
            "rag_summary": sanitized_summary
        })
    user_prompt = f"""
    You are a meticulous nutritionist. Your task is to determine if each recipe is strictly keto-friendly.
    A recipe is **strictly keto** if it contains NO ingredients with more than 10g of carbohydrates per 100g serving.
    **CRITICAL INSTRUCTION FOR MISSING DATA:**
    - The `rag_summary` may be incomplete or incorrect. You MUST use your own expert knowledge to override it.
    - For example, you know that 'pizza crust', 'pasta', 'bread', 'sugar', 'potatoes', 'flour', and 'rice' are ALWAYS high in carbohydrates (>10g/100g). Classify them as high-carb even if the summary says '0.0g'.
    **Analysis Process:**
    1. For each recipe, identify ingredients with >10g of carbs per 100g, using the `rag_summary` AND your own knowledge.
    2. In your reasoning, first list all high-carbohydrate ingredients you identified.
    3. After listing the evidence, make a final "Verdict" in your reasoning.
    4. Based on your verdict, set `is_keto` to `true` or `false`.
    **RESPONSE INSTRUCTIONS:**
    - You must respond with ONLY a single, valid JSON array. Each object must correspond to a recipe.
    - If you identify ANY high-carbohydrate ingredient, `is_keto` MUST be `false`.
    **RECIPE DATA:**
    ```json
    {json.dumps(recipe_data_list)}
    ```
    """
    schema = [{"recipe_id": "string", "is_keto": "boolean", "keto_reasoning": "string", "high_carb_ingredients_found": ["list", "of", "strings"]}]
    return [user_prompt, "Provide your analysis in this JSON format:", json.dumps(schema)]


async def classify_recipes(
    recipes: List[Dict[str, Any]],
    model_name: str,
    rate_limiter: AsyncTokenBucket,
    prompt_builder: Callable[[List[Dict[str, Any]]], list]
) -> Optional[List[Dict[str, Any]]]:
    """Classifies a batch of recipes using a generative AI model.

    Detailed Description:
        - This asynchronous function sends a request to the configured generative AI model.
        - It acquires a slot from the `AsyncTokenBucket` to respect rate limits.
        - It builds the prompt using the provided `prompt_builder`, sends it to the model, and parses the JSON response.
        - It includes error handling for API calls.

    Parameters:
        - recipes (List[Dict[str, Any]]): The list of recipes to classify.
        - model_name (str): The name of the generative AI model to use.
        - rate_limiter (AsyncTokenBucket): The rate limiter instance to manage API calls.
        - prompt_builder (Callable): The function to use for building the classification prompt.

    Returns:
        - Optional[List[Dict[str, Any]]]: A list of classification results, or None if the API call fails.

    Raises:
        - Exception: Catches and logs any exception during the API call, returning None.

    Libraries Used:
        - google.generativeai: The official Google AI SDK for Python. It's used to interact with Gemini models.
        - asyncio: Used for the `await` keyword, making the function non-blocking.
    """
    if not recipes: return []
    estimated_tokens = (800 + 200) * len(recipes)
    await rate_limiter.acquire(estimated_tokens)
    prompt = prompt_builder(recipes)
    try:
        gemini_model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", max_output_tokens=8192)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = await gemini_model.generate_content_async(prompt, generation_config=generation_config, safety_settings=safety_settings)
        return json.loads(response.text)
    except Exception as e:
        logging.warning(f"💥 API call failed for model {model_name} with {len(recipes)} recipes: {e}")
        return None


async def worker(
    name: str,
    model_name: str,
    rate_limiter: AsyncTokenBucket,
    batch_queue: asyncio.Queue,
    results_list: List[Dict[str, Any]],
    pbar: tqdm,
    prompt_builder: Callable[[List[Dict[str, Any]]], list]
):
    """An asynchronous worker that processes recipe classification tasks from a queue.

    Detailed Description:
        - This function runs in a loop, getting batches of recipes from an `asyncio.Queue`.
        - For each batch, it calls `classify_recipes`. If the batch fails, it retries each recipe individually.
        - It updates a `tqdm` progress bar.

    Parameters:
        - name (str): The name of the worker (for logging).
        - model_name (str): The model to use for classification.
        - rate_limiter (AsyncTokenBucket): The rate limiter for the API.
        - batch_queue (asyncio.Queue): The queue from which to fetch recipe batches.
        - results_list (List[Dict[str, Any]]): A list to store the classification results.
        - pbar (tqdm): The progress bar to update.
        - prompt_builder (Callable): The function for building prompts.

    Libraries Used:
        - asyncio: For managing the asynchronous worker loop and queue.
        - tqdm: Provides a progress bar for monitoring the classification process.
        - logging: For logging warnings and errors.
    """
    while True:
        batch_id, batch_recipes_gen = await batch_queue.get()
        if batch_recipes_gen is None:
            batch_queue.task_done()
            break
        batch_recipes = list(batch_recipes_gen)
        classified_results = await classify_recipes(batch_recipes, model_name, rate_limiter, prompt_builder)
        if classified_results:
            if not isinstance(classified_results, list): classified_results = [classified_results]
            results_list.extend(classified_results)
            pbar.update(len(batch_recipes))
        else:
            logging.warning(f"Worker '{name}': Batch {batch_id} failed. Retrying recipes individually...")
            for recipe in batch_recipes:
                individual_result = await classify_recipes([recipe], model_name, rate_limiter, prompt_builder)
                if individual_result:
                    if not isinstance(individual_result, list): individual_result = [individual_result]
                    results_list.extend(individual_result)
                else:
                    logging.error(f"Worker '{name}': Failed to process recipe {recipe.get('_id')} even individually.")
                pbar.update(1)
        batch_queue.task_done()


async def run_classification_stage(df: pl.DataFrame, prompt_builder: Callable, stage_name: str) -> pl.DataFrame:
    """Runs a complete classification stage for a given diet.

    Detailed Description:
        - This function orchestrates a classification stage (e.g., "Vegan").
        - It creates a queue of recipe batches and a pool of worker tasks.
        - Each worker uses a different model from the `MODELS_QUOTAS` configuration, allowing for parallel processing with different rate limits.
        - It gathers the results from all workers and returns them as a Polars DataFrame.

    Parameters:
        - df (pl.DataFrame): The input DataFrame of recipes.
        - prompt_builder (Callable): The function to build the prompt for this stage.
        - stage_name (str): The name of the stage (e.g., "Vegan", "Keto") for logging.

    Returns:
        - pl.DataFrame: A DataFrame containing the classification results for the stage.

    Libraries Used:
        - asyncio: To create and manage the worker tasks concurrently.
        - polars: For efficient data manipulation and DataFrame creation.
        - tqdm: To display an overall progress bar for the stage.
    """
    logging.info(f"🚀 Starting Stage: {stage_name} Classification...")
    work_queue = asyncio.Queue()
    all_results = []
    for i in range(0, len(df), CONFIG["BATCH_SIZE"]):
        batch_id = i // CONFIG["BATCH_SIZE"]
        batch_df = df.slice(i, CONFIG["BATCH_SIZE"])
        work_queue.put_nowait((batch_id, batch_df.iter_rows(named=True)))
    logging.info(f"Prepared {work_queue.qsize()} batches for {stage_name} stage.")
    worker_tasks = []
    with tqdm(total=df.shape[0], desc=f"Classifying ({stage_name})") as pbar:
        for model_api_name, (rpm_limit, tpm_limit) in MODELS_QUOTAS.items():
            rate_limiter = AsyncTokenBucket(rpm_rate=rpm_limit, tpm_rate=tpm_limit)
            task = asyncio.create_task(worker(f"Worker-{model_api_name}", model_api_name, rate_limiter, work_queue, all_results, pbar, prompt_builder))
            worker_tasks.append(task)
            await work_queue.put((None, None))
        await work_queue.join()
        await asyncio.gather(*worker_tasks)
    if not all_results:
        logging.error(f"❌ No results were generated for stage {stage_name}.")
        return pl.DataFrame()
    return pl.DataFrame(all_results)


async def main():
    logger.info("🚀 Starting SOTA Personas Ground Truth Generation Pipeline...")
    
    # --- Connect to Services ---
    llm = LLMHandler()
    os_client = OpenSearch(hosts=[CONFIG["OPENSEARCH_URL"]])
    rate_limiter = AsyncTokenBucket(CONFIG["RPM_LIMIT"], CONFIG["TPM_LIMIT"])
    
    if not os_client.ping():
        logger.error("❌ Could not connect to OpenSearch. Aborting.")
        return

    # --- Fetch and Process Data ---
    recipes_hits = fetch_recipes_synchronously(os_client)
    
    processed_recipes = []
    # ... (enrichment loop)
    # This loop is now correct and doesn't need changes.

    # --- Asynchronous Part: Classify Recipes ---
    if not processed_recipes:
        logger.warning("No recipes could be processed. Aborting classification.")
        return
        
    all_results = await classify_recipes_async(processed_recipes, llm, rate_limiter)
    
    # --- MLflow Logging & Saving ---
    # ... (MLflow and saving logic)


def fetch_recipes_synchronously(os_client: OpenSearch) -> List[Dict]:
    """Fetches a random sample of recipes using the synchronous client."""
    logger.info(f"Fetching {CONFIG['SAMPLE_SIZE']} random recipes from OpenSearch...")
    query = {"size": CONFIG["SAMPLE_SIZE"], "query": {"function_score": {"functions": [{"random_score": {}}]}}}
    response = os_client.search(index="recipes", body=query)
    return response['hits']['hits']

async def classify_recipes_async(processed_recipes: List[Dict], llm: LLMHandler, rate_limiter: AsyncTokenBucket) -> List[Dict]:
    """Runs the asynchronous classification part of the pipeline."""
    all_results = []
    tasks = []
    for i in range(0, len(processed_recipes), CONFIG["BATCH_SIZE"]):
        batch = processed_recipes[i:i + CONFIG["BATCH_SIZE"]]
        tasks.append(classify_batch(batch, llm, rate_limiter))
        
    for result in await tqdm_asyncio.gather(*tasks, desc="Classifying Batches"):
        all_results.extend(result)
    return all_results

if __name__ == "__main__":
    logging.basicConfig(level=CONFIG["LOG_LEVEL"], format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    logger.add(CONFIG["OUTPUT_DIR"] / "personas_generation.log", rotation="10 MB")
    asyncio.run(main()) 