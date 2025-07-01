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

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# --- Configuration ---
CONFIG = {
    "INPUT_DIR": Path("data"),
    "ENRICHED_INPUT_FILE": Path("data/enriched_recipes_for_llm.csv"),
    "OUTPUT_DIR": Path("data/final_classifications"),
    "SAMPLE_SIZE": 500,
    "BATCH_SIZE": 10,
    "LOG_LEVEL": logging.INFO
}

MODELS_QUOTAS = {
    'gemini-2.5-pro': (5, 250_000),
    'gemini-2.5-flash': (10, 250_000),
    'gemini-2.5-flash-lite-preview-06-17': (15, 250_000),
}

logging.basicConfig(level=CONFIG["LOG_LEVEL"], format='%(asctime)s - %(levelname)s - %(message)s', force=True)

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
                await asyncio.sleep(time_to_wait)
                self.last_reset_time = time.time()
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
            self.requests_this_minute += 1
            self.tokens_this_minute += estimated_tokens


def build_vegan_prompt(recipe_rows: List[Dict[str, Any]]) -> list:
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


async def main_async():
    logging.info("🚀 Starting SOTA Decomposed Ground Truth Generation Pipeline...")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.error("🚫 GEMINI_API_KEY not found."); return
    genai.configure(api_key=gemini_api_key)

    input_file = CONFIG["ENRICHED_INPUT_FILE"]
    if not input_file.exists():
        logging.error(f"❌ Input file not found: {input_file}"); return

    df = pl.read_csv(input_file).head(CONFIG["SAMPLE_SIZE"])
    logging.info(f"📖 Loaded and sampled {len(df)} recipes from {input_file}")

    vegan_results_df = await run_classification_stage(df, build_vegan_prompt, "Vegan")
    if vegan_results_df.is_empty():
        logging.error("Vegan classification stage failed. Aborting."); return

    keto_results_df = await run_classification_stage(df, build_keto_prompt, "Keto")
    if keto_results_df.is_empty():
        logging.error("Keto classification stage failed. Aborting."); return

    logging.info("✅ All classification stages complete. Merging results...")
    base_df = df.select(["_id", "title", "ingredients", "rag_summary"])
    final_df = base_df.join(vegan_results_df, left_on="_id", right_on="recipe_id", how="left")
    final_df = final_df.join(keto_results_df, left_on="_id", right_on="recipe_id", how="left")

    # --- THIS IS THE FIX ---
    # Convert list-type columns to comma-separated strings before writing to CSV.
    final_df = final_df.with_columns(
        pl.col("animal_ingredients_found").list.join(", ").alias("animal_ingredients_found"),
        pl.col("high_carb_ingredients_found").list.join(", ").alias("high_carb_ingredients_found")
    )
    # --- END FIX ---

    output_dir = CONFIG["OUTPUT_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_classifications.csv"
    final_df.write_csv(output_path)

    logging.info(f"💾 Successfully saved {len(final_df)} final classifications to {output_path}")
    logging.info("\n--- Vegan Classification Summary ---")
    logging.info(final_df['is_vegan'].value_counts())
    logging.info("\n--- Keto Classification Summary ---")
    logging.info(final_df['is_keto'].value_counts())
    logging.info("\n🎉 Ground truth generation pipeline completed successfully!")


if __name__ == '__main__':
    asyncio.run(main_async())