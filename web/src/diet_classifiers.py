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
    """A singleton handler for Elasticsearch database operations.

    Detailed Description:
        - This class implements the singleton pattern to ensure only one Elasticsearch
          connection is maintained throughout the application lifecycle.
        - It provides a simple interface for searching ingredient information in the
          Elasticsearch recipes index.
        - The singleton pattern prevents connection overhead and ensures consistent
          database state across the application.

    Libraries Used:
        - elasticsearch: The official Python client for Elasticsearch, providing robust
          connection management and query building capabilities over raw HTTP requests.
    """
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
        """Searches for ingredient information in the Elasticsearch recipes index.

        Detailed Description:
            - This method performs a fuzzy match search against the recipe descriptions
              to find relevant ingredient information.
            - It uses Elasticsearch's built-in relevance scoring to return the most
              relevant result.

        Parameters:
            - ingredient_name (str): The name of the ingredient to search for.

        Returns:
            - Optional[Dict[str, Any]]: The source document of the best matching recipe,
              or None if no results are found or if the query fails.
        """
        if not self.client: return None
        try:
            res = self.client.search(index="recipes", body={"query": {"match": {"description": ingredient_name}}}, size=1)
            return res['hits']['hits'][0]['_source'] if res['hits']['hits'] else None
        except Exception as e:
            logger.error(f"Elasticsearch query for '{ingredient_name}' failed: {e}")
            return None

# --- 2. LLM HANDLER (Qwen-0.6B via Transformers) ---

class LLMHandler:
    """A singleton handler for the Qwen language model using Transformers.

    Detailed Description:
        - This class manages the loading and querying of the Qwen-0.6B model for
          dietary classification tasks.
        - It uses the Transformers library's pipeline interface for simplified model interaction.
        - The singleton pattern ensures the expensive model loading operation occurs only once.

    Libraries Used:
        - transformers: Hugging Face's library for state-of-the-art NLP models. It provides
          high-level abstractions and optimizations over raw PyTorch implementations.
        - torch: The underlying deep learning framework, used for tensor operations and model inference.
    """
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
        """Sends a prompt to the Qwen model and returns the generated response.

        Detailed Description:
            - This method formats the prompt using the model's chat template.
            - It uses controlled generation parameters (temperature, top_p, repetition_penalty)
              to balance creativity and consistency in the model's responses.
            - It extracts the assistant's response from the full generated text.

        Parameters:
            - prompt (str): The input prompt for the model.

        Returns:
            - str: The model's generated response, or a JSON error string if generation fails.
        """
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
    """Processes a raw ingredient string to extract structured context information.

    Detailed Description:
        - This function implements a multi-stage processing pipeline:
          1. **Parsing:** Uses ingredient-parser to extract name, quantity, and unit.
          2. **Retrieval:** Searches Elasticsearch for additional ingredient information.
          3. **Normalization:** Calculates total carbohydrates based on quantity and retrieved data.
        - It includes robust error handling to gracefully handle parsing failures.

    Parameters:
        - raw_ingredient (str): The raw ingredient string (e.g., "2 cups flour").

    Returns:
        - Dict: A dictionary containing original, parsed, retrieved, and normalized information.

    Libraries Used:
        - ingredient_parser: Specialized library for parsing recipe ingredient strings,
          superior to regex approaches for handling natural language variations.
    """
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
    """Classifies a tuple of ingredients using the LLM with caching for performance.

    Detailed Description:
        - This function uses the LLM to perform joint keto and vegan classification.
        - It processes all ingredients to get their contexts, then sends a structured
          prompt to the language model for classification.
        - The @lru_cache decorator provides automatic memoization for repeated ingredient
          combinations, significantly improving performance for common recipes.

    Parameters:
        - ingredients (tuple): A tuple of ingredient strings (tuple required for hashing).

    Returns:
        - Dict: A dictionary with 'is_keto' and 'is_vegan' boolean keys.

    Libraries Used:
        - functools.lru_cache: For automatic memoization, providing O(1) lookup for
          previously classified ingredient combinations.
    """
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
    """Determines if a list of ingredients is keto-friendly.

    Detailed Description:
        - This function implements a conservative keto classification approach.
        - It checks each ingredient individually using `is_ingredient_keto`.
        - A recipe is considered keto only if ALL ingredients are keto-friendly.
        - This approach minimizes false positives in keto classification.

    Parameters:
        - ingredients (List[str]): A list of ingredient strings to classify.

    Returns:
        - bool: True if all ingredients are keto-friendly, False otherwise.

    Examples:
        >>> is_keto(["chicken breast", "spinach", "olive oil"])
        True
        >>> is_keto(["chicken breast", "bread"])
        False
    """
    if not ingredients: return False
    return all(map(is_ingredient_keto, ingredients))

def is_vegan(ingredients: List[str]) -> bool:
    """Determines if a list of ingredients is vegan-friendly.

    Detailed Description:
        - This function implements vegan classification by checking each ingredient
          against the vegan ontology.
        - A recipe is considered vegan only if ALL ingredients are vegan-friendly.
        - Empty ingredient lists are considered vegan by default.

    Parameters:
        - ingredients (List[str]): A list of ingredient strings to classify.

    Returns:
        - bool: True if all ingredients are vegan-friendly, False otherwise.

    Examples:
        >>> is_vegan(["tomatoes", "basil", "olive oil"])
        True
        >>> is_vegan(["tomatoes", "cheese"])
        False
    """
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
    """Classifies a single ingredient as keto-friendly based on carbohydrate content.

    Detailed Description:
        - This function uses the nutrition database to determine if an ingredient
          meets the keto carbohydrate threshold (< 20g per 100g).
        - It first attempts to parse the ingredient to get a clean name.
        - If the ingredient is not found in the database, it conservatively returns False.

    Parameters:
        - ingredient (str): The ingredient string to classify.

    Returns:
        - bool: True if the ingredient is keto-friendly, False otherwise.

    Libraries Used:
        - ingredient_parser: For extracting clean ingredient names from complex strings.
        - data_manager: For accessing the preloaded nutrition database.
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
    """Classifies a single ingredient as vegan-friendly using the vegan ontology.

    Detailed Description:
        - This function checks if an ingredient appears in the non-vegan ontology set.
        - It first attempts to parse the ingredient to get a clean name.
        - An ingredient is considered vegan if it does NOT appear in the non-vegan set.

    Parameters:
        - ingredient (str): The ingredient string to classify.

    Returns:
        - bool: True if the ingredient is vegan-friendly, False otherwise.

    Libraries Used:
        - ingredient_parser: For extracting clean ingredient names.
        - data_manager: For accessing the preloaded vegan ontology set.
    """
    try:
        parsed = parse_ingredient(ingredient)
        name = parsed.name.text.lower().strip()
    except Exception:
        name = ingredient.lower().strip()
        
    # An ingredient is vegan if its name does NOT appear in our non-vegan ontology set.
    return name not in data_manager.vegan_ontology_set

def main(args):
    """Runs the dietary classification evaluation against a ground truth dataset.

    Detailed Description:
        - This function serves as the main entry point for evaluation mode.
        - It loads a ground truth CSV file containing recipes with known classifications.
        - It applies the `is_keto` and `is_vegan` functions to each recipe.
        - It generates classification reports comparing predictions to ground truth labels.

    Parameters:
        - args (argparse.Namespace): Command-line arguments containing the ground truth file path.

    Returns:
        - int: Exit code (0 for success, -1 for failure).

    Libraries Used:
        - pandas: For loading and manipulating the ground truth dataset.
        - scikit-learn: For generating detailed classification reports with precision, recall, and F1 scores.
    """
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