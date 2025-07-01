import os
import polars as pl
import json
import time
import asyncio
import mlflow
from loguru import logger
from pathlib import Path
from typing import Dict, List, Any, Callable
from tqdm.asyncio import tqdm_asyncio
from opensearchpy import OpenSearch, AsyncOpenSearch

# --- Local Module Imports ---
# This structure assumes the script is run from the project root (nb/)
# and the project is installed in editable mode.
from llm_handler.handler import LLMHandler
from ingredient_processor.processor import get_ingredient_context

# --- Configuration ---
CONFIG = {
    "OUTPUT_DIR": Path(os.getenv("DATA_DIR", "nb/data")),
    "GROUND_TRUTH_FILENAME": "ground_truth_sota.csv",
    "SAMPLE_SIZE": int(os.getenv("RECIPE_SAMPLE_SIZE", 50)),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 5)),
    "TEACHER_MODEL": os.getenv("TEACHER_MODEL", "qwen:latest"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "OPENSEARCH_URL": os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
    "RPM_LIMIT": int(os.getenv("RPM_LIMIT", 10)), # Requests per minute
    "TPM_LIMIT": int(os.getenv("TPM_LIMIT", 200000)), # Tokens per minute
}

# --- Rate Limiter ---
class AsyncTokenBucket:
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
            
            time_to_wait_rpm = 0.0
            if self.requests_this_minute >= self.rpm_rate:
                time_to_wait_rpm = 60 - (now - self.last_reset_time)
            
            time_to_wait_tpm = 0.0
            if self.tokens_this_minute + estimated_tokens > self.tpm_rate:
                time_to_wait_tpm = 60 - (now - self.last_reset_time)
            
            time_to_wait = max(0.0, time_to_wait_rpm, time_to_wait_tpm)
            if time_to_wait > 0.1:
                logger.warning(f"Rate limit nearing. Pausing for {time_to_wait:.2f}s...")
                await asyncio.sleep(time_to_wait)
                # Reset counters after waiting
                self.last_reset_time = time.time()
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
            
            self.requests_this_minute += 1
            self.tokens_this_minute += estimated_tokens

# --- Prompt Builders ---
def build_vegan_prompt(recipe_batch: List[Dict[str, Any]]) -> str:
    """Builds the specialized prompt for vegan classification."""
    recipe_data_list = [{
        "recipe_id": r.get('recipe_id'), 
        "title": r.get('title'), 
        "ingredients": json.loads(r.get('ingredients', '[]'))
    } for r in recipe_batch]
    
    prompt = f"""
    You are a meticulous food scientist. Your task is to determine if each recipe in the following list is strictly vegan.
    A recipe is **strictly vegan** if it contains absolutely no animal products (no meat, poultry, fish, dairy, eggs, honey, etc.).
    
    **Analysis Process:**
    1. For each recipe, carefully examine the ingredients.
    2. In your reasoning, first list any and all ingredients that are or could be derived from animals.
    3. After listing the evidence, make a final "Verdict" in your reasoning.
    4. Based on your verdict, set `is_vegan` to `true` or `false`.
    
    **RESPONSE INSTRUCTIONS:**
    - You must respond with ONLY a single, valid JSON array of objects. Each object must correspond to a recipe.
    - Each object must have the keys: "recipe_id", "is_vegan", "vegan_reasoning", "animal_ingredients_found".
    - If you identify ANY potential animal product, `is_vegan` MUST be `false`.
    
    **RECIPE DATA:**
    ```json
    {json.dumps(recipe_data_list, indent=2)}
    ```
    """
    return prompt

def build_keto_prompt(recipe_batch: List[Dict[str, Any]]) -> str:
    """Builds the specialized prompt for keto classification using our processed context."""
    recipe_data_list = [{
        "recipe_id": r.get('recipe_id'), 
        "title": r.get('title'), 
        "ingredients_context": json.loads(r.get('processed_context', '[]'))
    } for r in recipe_batch]

    prompt = f"""
    You are a meticulous nutritionist. Your task is to determine if each recipe is strictly keto-friendly.
    A recipe is **strictly keto** if it contains NO ingredients known to be high in carbohydrates.
    
    Your primary source of truth is the `processed_context` for each recipe, which includes deterministic calculations.
    
    **Analysis Process:**
    1. For each recipe, analyze the `ingredients_context`. Pay close attention to the `normalized.calculated_carbs_g` and `parsed.name`.
    2. Use your own expert knowledge to override or flag ingredients. For example, 'bread', 'sugar', 'potatoes', 'flour', and 'rice' are ALWAYS high-carb.
    3. In your reasoning, first list all high-carbohydrate ingredients you identified.
    4. After listing the evidence, make a final "Verdict" in your reasoning.
    5. Based on your verdict, set `is_keto` to `true` or `false`.
    
    **RESPONSE INSTRUCTIONS:**
    - You must respond with ONLY a single, valid JSON array of objects. Each object must correspond to a recipe.
    - Each object must have the keys: "recipe_id", "is_keto", "keto_reasoning", "high_carb_ingredients_found".
    - If you identify ANY high-carbohydrate ingredient, `is_keto` MUST be `false`.

    **RECIPE DATA:**
    ```json
    {json.dumps(recipe_data_list, indent=2)}
    ```
    """
    return prompt

# --- Core Logic ---
async def classify_batch(
    batch: List[Dict[str, Any]],
    llm: LLMHandler,
    rate_limiter: AsyncTokenBucket,
    prompt_builder: Callable[[List[Dict[str, Any]]], str]
) -> List[Dict[str, Any]]:
    if not batch: return []
    # Simplified token estimation
    prompt = prompt_builder(batch)
    await rate_limiter.acquire(len(prompt))
    
    response = await llm.async_query(CONFIG["TEACHER_MODEL"], prompt, as_json=True)
    
    if "error" in response or not isinstance(response, list):
        logger.error(f"Batch failed. Response: {response}")
        return []
    return response


async def process_recipes_in_stages(df: pl.DataFrame, llm: LLMHandler, rate_limiter: AsyncTokenBucket) -> pl.DataFrame:
    # Stage 1: Vegan Classification
    logger.info("--- Stage 1: Vegan Classification ---")
    vegan_tasks = []
    for i in tqdm(range(0, len(df), CONFIG["BATCH_SIZE"]), desc="Creating Vegan batches"):
        batch_df = df.slice(i, CONFIG["BATCH_SIZE"])
        vegan_tasks.append(classify_batch(batch_df.to_dicts(), llm, rate_limiter, build_vegan_prompt))
    
    vegan_results = await tqdm_asyncio.gather(*vegan_tasks, desc="Processing Vegan batches")
    vegan_flat_results = [item for sublist in vegan_results for item in sublist]
    vegan_df = pl.DataFrame(vegan_flat_results)

    # Stage 2: Keto Classification
    logger.info("--- Stage 2: Keto Classification ---")
    keto_tasks = []
    for i in tqdm(range(0, len(df), CONFIG["BATCH_SIZE"]), desc="Creating Keto batches"):
        batch_df = df.slice(i, CONFIG["BATCH_SIZE"])
        keto_tasks.append(classify_batch(batch_df.to_dicts(), llm, rate_limiter, build_keto_prompt))
        
    keto_results = await tqdm_asyncio.gather(*keto_tasks, desc="Processing Keto batches")
    keto_flat_results = [item for sublist in keto_results for item in sublist]
    keto_df = pl.DataFrame(keto_flat_results)
    
    # Merge results
    final_df = df.join(vegan_df, on="recipe_id", how="left")
    final_df = final_df.join(keto_df, on="recipe_id", how="left")
    
    return final_df


async def main():
    logger.info("🚀 Starting SOTA Ground Truth Generation Pipeline...")
    
    # --- Connect to Services ---
    llm = LLMHandler()
    os_client = AsyncOpenSearch(hosts=[CONFIG["OPENSEARCH_URL"]])
    rate_limiter = AsyncTokenBucket(CONFIG["RPM_LIMIT"], CONFIG["TPM_LIMIT"])
    
    if not await os_client.ping():
        logger.error("❌ Could not connect to OpenSearch. Aborting.")
        return

    # --- Fetch and Process Data ---
    logger.info(f"Fetching {CONFIG['SAMPLE_SIZE']} random recipes from OpenSearch...")
    query = {"size": CONFIG["SAMPLE_SIZE"], "query": {"function_score": {"functions": [{"random_score": {}}]}}}
    response = await os_client.search(index="recipes", body=query)
    recipes_hits = response['hits']['hits']
    
    processed_recipes = []
    for hit in recipes_hits:
        source = hit['_source']
        ingredients = source.get('ingredients', [])
        if not ingredients or not isinstance(ingredients, list): continue
        
        # Enrich with our deterministic processor
        context = [get_ingredient_context(ing) for ing in ingredients]
        
        processed_recipes.append({
            "recipe_id": hit['_id'],
            "title": source.get('title'),
            "ingredients": json.dumps(ingredients),
            "processed_context": json.dumps(context)
        })
    
    df = pl.DataFrame(processed_recipes)

    # --- Run MLflow Experiment ---
    mlflow.set_tracking_uri(CONFIG["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_name="SOTA Ground Truth Generation") as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        mlflow.log_params(CONFIG)
        
        final_df = await process_recipes_in_stages(df, llm, rate_limiter)
        
        output_path = CONFIG["OUTPUT_DIR"] / CONFIG["GROUND_TRUTH_FILENAME"]
        final_df.write_csv(output_path)
        mlflow.log_artifact(str(output_path), "generated_datasets")
        logger.success(f"✅ Ground truth generation complete. Saved to {output_path}")

    await os_client.close()

if __name__ == "__main__":
    # Setup logging and paths
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    logger.add(CONFIG["OUTPUT_DIR"] / "ground_truth_generation.log", rotation="10 MB")
    
    asyncio.run(main()) 