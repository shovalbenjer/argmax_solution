"""
Final Evaluation Script

This script runs the dietary classification system against all specified
ground truth files and reports the final performance metrics.
"""
import sys
import time
from pathlib import Path
from typing import List
from argparse import ArgumentParser
import pandas as pd
from loguru import logger
import os

# Definitive path correction
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from nb.src.diet_classifiers import is_keto, is_vegan
    from sklearn.metrics import classification_report
except ImportError:
    logger.critical("Could not import required modules. Make sure all scripts are in place.")
    sys.exit(1)

def load_and_prepare_data(data_paths: List[Path]) -> pd.DataFrame:
    """Loads, combines, and prepares all ground truth data for evaluation.

    Detailed Description:
        - This function takes a list of file paths to CSVs containing ground truth data.
        - It iterates through each path, loading it into a pandas DataFrame.
        - If a file does not contain 'keto' or 'vegan' columns, it infers the ground truth
          label from the filename (e.g., a file named 'strict_keto.csv' is assumed to contain keto recipes).
        - It combines all DataFrames into a single one.
        - It then processes the 'ingredients' column, which is stored as a stringified list,
          by safely evaluating it into an actual list of strings using a helper function.

    Parameters:
        - data_paths (List[Path]): A list of `pathlib.Path` objects pointing to the ground truth CSV files.

    Returns:
        - pd.DataFrame: A single DataFrame containing the combined and cleaned data.

    Raises:
        - FileNotFoundError: If no valid ground truth files are found at the specified paths.
    """
    all_dfs = []
    for path in data_paths:
        if not path.exists():
            logger.warning(f"File not found: {path}. Skipping.")
            continue
        
        df = pd.read_csv(path)
        
        # Create ground truth labels based on filename if not present
        if 'keto' not in df.columns:
            df['keto'] = "keto" in path.name
        if 'vegan' not in df.columns:
            df['vegan'] = "vegan" in path.name
            
        all_dfs.append(df)
        logger.info(f"Loaded {len(df)} records from {path.name}")
        
    if not all_dfs:
        raise FileNotFoundError("No valid ground truth files were found.")
        
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Robustly evaluate the 'ingredients' column
    def safe_eval_ingredients(val):
        if isinstance(val, str) and val.startswith('['):
            try:
                # The string looks like a list, so we can use eval
                return eval(val)
            except:
                # If eval fails, return an empty list
                return []
        # If it's already a list or something else, return as is or empty
        return val if isinstance(val, list) else []
        
    combined_df['ingredients_list'] = combined_df['ingredients'].apply(safe_eval_ingredients)
    
    return combined_df

def main(args):
    """Orchestrates the final model evaluation process.

    Detailed Description:
        - This is the main execution function for the evaluation script.
        - It defines the paths to the various ground truth datasets based on command-line arguments.
        - It calls `load_and_prepare_data` to get the complete, cleaned evaluation dataset.
        - It then applies the `is_keto` and `is_vegan` classification functions to the 'ingredients_list'
          column to generate predictions.
        - Finally, it uses `sklearn.metrics.classification_report` to generate and print a detailed
          report comparing the ground truth labels to the predictions for both keto and vegan classifications.

    Parameters:
        - args (argparse.Namespace): An object containing the command-line arguments, which specify the
          paths to the different ground truth data files.

    Returns:
        - int: An exit code (0 for success, -1 for failure).

    Libraries Used:
        - pandas: For all data manipulation.
        - scikit-learn: For generating the final classification report, which provides key metrics like
          precision, recall, and F1-score. It is a standard for machine learning evaluation.
        - loguru: For logging progress and errors.
    """
    data_files = [
        Path(args.ground_truth),
        Path(args.strict_keto),
        Path(args.borderline_keto),
        Path(args.strict_vegan)
    ]
    
    logger.info("--- Loading and Preparing Evaluation Data ---")
    try:
        ground_truth_df = load_and_prepare_data(data_files)
    except FileNotFoundError as e:
        logger.error(e)
        return -1

    logger.info("--- Starting Model Predictions ---")
    start_time = time.time()
    
    try:
        ground_truth_df['keto_pred'] = ground_truth_df['ingredients_list'].apply(is_keto)
        ground_truth_df['vegan_pred'] = ground_truth_df['ingredients_list'].apply(is_vegan)
    except Exception as e:
        logger.critical(f"An error occurred during prediction: {e}")
        return -1
        
    end_time = time.time()
    
    logger.info("--- Evaluation Results ---")
    print("\n" + "="*20 + " Keto Classification " + "="*20)
    print(classification_report(
        ground_truth_df['keto'], ground_truth_df['keto_pred'], target_names=['Not Keto', 'Keto']
    ))
    
    print("\n" + "="*20 + " Vegan Classification " + "="*20)
    print(classification_report(
        ground_truth_df['vegan'], ground_truth_df['vegan_pred'], target_names=['Not Vegan', 'Vegan']
    ))
    
    total_time = end_time - start_time
    avg_time = total_time / len(ground_truth_df) if len(ground_truth_df) > 0 else 0
    print("\n" + "="*54)
    logger.success(f"Evaluation finished in {total_time:.2f}s ({avg_time * 1000:.2f} ms/recipe).")
    
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    base_data_path = Path(__file__).resolve().parent / "data"
    
    parser.add_argument("--ground_truth", type=str, default=str(base_data_path / "ground_truth_sample.csv"))
    parser.add_argument("--strict_keto", type=str, default=str(base_data_path / "strict_keto.csv"))
    parser.add_argument("--borderline_keto", type=str, default=str(base_data_path / "borderline_keto.csv"))
    parser.add_argument("--strict_vegan", type=str, default=str(base_data_path / "strict_vegan.csv"))
    
    sys.exit(main(parser.parse_args())) 