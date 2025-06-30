# --- File: diet_classifiers.py (Refined for Production) ---

"""
Fully Integrated, Production-Grade Dietary Classifier.

This single, robust script contains the complete, end-to-end system for
dietary classification, designed for reliability and performance. It integrates:
1.  An Elasticsearch database handler.
2.  The Hybrid Retrieval Cascade and Normalization Engine.
3.  A `transformers`-based LLM Handler for the Qwen-0.6B model.
"""
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Any
import os
import sys
from argparse import ArgumentParser

import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from ingredient_parser import parse_ingredient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    from sklearn.metrics import classification_report
except ImportError:
    logger.warning("scikit-learn not installed. Evaluation metrics will be limited.")
    classification_report = lambda y_true, y_pred, target_names: "scikit-learn not available."

try:
    from data_manager import data_manager
except ImportError:
    logger.critical("Could not import data_manager. Make sure it is in the same directory.")
    sys.exit(1)

KETO_CARBS_THRESHOLD = 20  # g of carbs per 100g of ingredient

# --- 1. DATABASE HANDLER (Elasticsearch) ---

class DatabaseHandler:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseHandler, cls).__new__(cls)
            try:
                cls._instance.client = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)
                if not cls._instance.client.ping():
                    raise ConnectionError("Could not connect to Elasticsearch")
                logger.success("Connected to Elasticsearch successfully.")
            except Exception as e:
                logger.critical(f"Failed to connect to Elasticsearch: {e}")
                cls._instance.client = None
        return cls._instance

    def search_ingredient(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        if not self.client: return None
        try:
            res = self.client.search(index="recipes", body={"query": {"match": {"description": ingredient_name}}}, size=1)
            return res['hits']['hits'][0]['_source'] if res['hits']['hits'] else None
        except Exception as e:
            logger.error(f"Elasticsearch query for '{ingredient_name}' failed: {e}")
            return None

# --- 2. LLM HANDLER (Qwen-0.6B via Transformers) ---

class LLMHandler:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMHandler, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        model_name = "Qwen/Qwen3-0.6B"
        logger.info(f"Initializing Transformers pipeline for model: {model_name}")
        try:
            self.pipe = pipeline("text-generation", model=model_name, token=hf_token, device_map="auto", torch_dtype="auto")
            logger.success(f"Qwen model loaded successfully on device: {self.pipe.device}")
        except Exception as e:
            logger.critical(f"Failed to load model from Hugging Face: {e}")
            raise

    def query(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            outputs = self.pipe(prompt_formatted, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.5)
            response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Error during LLM query: {e}")
            return '{"error": "LLM query failed"}'

# --- 3. INGREDIENT PROCESSOR & CLASSIFICATION LOGIC ---
db_handler = DatabaseHandler()
llm_handler = LLMHandler()

def get_ingredient_context(raw_ingredient: str) -> Dict:
    # A. Ingredient Parsing
    try:
        parsed = parse_ingredient(raw_ingredient, discard_isolated_stop_words=True)
        name = ' '.join([item.text for item in parsed.name]) if isinstance(parsed.name, list) else parsed.name.text
        
        # --- FIX: Convert Unit object to string ---
        unit_text = ""
        if parsed.amount and parsed.amount[0].unit:
            # The Unit object is not JSON serializable, so we convert it to a string.
            unit_text = str(parsed.amount[0].unit)
            
        parsed_info = {
            "name": name, 
            "quantity": float(parsed.amount[0].quantity) if parsed.amount else 1.0, 
            "unit": unit_text
        }
    except Exception as e:
        logger.warning(f"Parsing failed for '{raw_ingredient}': {e}. Using raw string.")
        parsed_info = {"name": raw_ingredient, "quantity": 1.0, "unit": "unit"}

    # B. Symbolic Retrieval (from Elasticsearch)
    retrieved_data = db_handler.search_ingredient(parsed_info["name"])

    # C. Normalization Engine
    normalized = {}
    if retrieved_data and "carbohydrates" in retrieved_data:
        try:
            total_carbs = float(retrieved_data["carbohydrates"]) * parsed_info["quantity"]
            normalized["total_carbs_g"] = total_carbs
        except (ValueError, TypeError):
            normalized["total_carbs_g"] = None
    
    return {"original": raw_ingredient, "parsed": parsed_info, "retrieved": retrieved_data, "normalized": normalized}

@lru_cache(maxsize=512)
def classify_ingredients(ingredients: tuple) -> Dict:
    contexts = [get_ingredient_context(ing) for ing in ingredients]
    prompt = f"""
You are an expert dietary classifier. Based on the following structured data, classify the recipe.
A recipe is 'keto' if its total carbohydrates are very low (e.g., under 20g).
A recipe is 'vegan' if it contains no animal products.
Your response MUST be a single, valid JSON object with two boolean keys: "is_keto" and "is_vegan".

Context: {json.dumps(contexts, indent=2)}

JSON Response:
"""
    response_str = llm_handler.query(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        return {"is_keto": False, "is_vegan": False}

def is_keto(ingredients: List[str]) -> bool:
    """Classifies if a list of ingredients is keto-friendly."""
    if not ingredients: return False
    return all(map(is_ingredient_keto, ingredients))

def is_vegan(ingredients: List[str]) -> bool:
    """Classifies if a list of ingredients is vegan-friendly."""
    if not ingredients: return True
    return all(map(is_ingredient_vegan, ingredients))

# --- Final Evaluation ---

def run_evaluation():
    """Runs the final evaluation against the ground truth dataset."""
    logger.info("===== Running Final Evaluation =====")
    gt_path = Path(__file__).resolve().parent / "data" / "ground_truth_sample.csv"
    if not gt_path.exists():
        logger.error(f"Ground truth file not found at {gt_path}. Run generate_ground_truth.py first.")
        return

    ground_truth = pd.read_csv(gt_path)
    
    # Safely evaluate string-formatted lists
    ground_truth['ingredients'] = ground_truth['ingredients'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])

    start_time = time.time()
    ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
    ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)
    end_time = time.time()

    logger.info("--- Keto Classification Report ---")
    print(classification_report(ground_truth['keto'], ground_truth['keto_pred']))
    
    logger.info("--- Vegan Classification Report ---")
    print(classification_report(ground_truth['vegan'], ground_truth['vegan_pred']))

    total_time = end_time - start_time
    avg_time = total_time / len(ground_truth) if len(ground_truth) > 0 else 0
    logger.success(f"Evaluation finished in {total_time:.2f}s ({avg_time:.3f}s/recipe).")

def is_ingredient_keto(ingredient: str) -> bool:
    """
    Checks if a single ingredient is keto-friendly based on its carbohydrate content.
    """
    try:
        # Use the ingredient parser to get the clean name of the ingredient
        parsed = parse_ingredient(ingredient)
        name = parsed.name.text.lower().strip()
    except Exception:
        # Fallback to the raw ingredient string if parsing fails
        name = ingredient.lower().strip()

    if name in data_manager.nutrition_df.index:
        carbs = data_manager.nutrition_df.loc[name, "carbohydrates_g"]
        return carbs < KETO_CARBS_THRESHOLD
    
    # If the ingredient is not in our database, assume it is not keto to be safe.
    return False

def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Checks if a single ingredient is vegan-friendly by looking for non-vegan terms.
    """
    try:
        parsed = parse_ingredient(ingredient)
        name = parsed.name.text.lower().strip()
    except Exception:
        name = ingredient.lower().strip()
        
    # An ingredient is vegan if its name does NOT appear in our non-vegan ontology set.
    return name not in data_manager.vegan_ontology_set

def main(args):
    """Runs the evaluation against the provided ground truth file."""
    logger.info("--- Starting Evaluation ---")
    try:
        ground_truth = pd.read_csv(args.ground_truth, index_col=None)
        ground_truth['ingredients'] = ground_truth['ingredients'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    except FileNotFoundError:
        logger.error(f"Ground truth file not found at: {args.ground_truth}")
        return -1

    start_time = time()
    ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
    ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)
    end_time = time.time()

    print("\n" + "="*20 + " Keto Classification " + "="*20)
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred'], target_names=['Not Keto', 'Keto'], zero_division=0
    ))
    
    print("\n" + "="*20 + " Vegan Classification " + "="*20)
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred'], target_names=['Not Vegan', 'Vegan'], zero_division=0
    ))
    
    logger.success(f"== Evaluation finished in {end_time - start_time:.2f} seconds ==")
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str, default="data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))