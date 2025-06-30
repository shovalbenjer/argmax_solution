import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Updated paths to the raw data files
NUTRITION_PATH = Path(__file__).resolve().parent / "raw_data" / "nutrition.csv"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "data" / "ground_truth_sample.csv"
sns.set_style("whitegrid")

print(f"--- 1. Raw Nutrition Data EDA ---")
if not NUTRITION_PATH.exists():
    print(f"❌ Nutrition data file not found at {NUTRITION_PATH}.")
else:
    # EDA: Carbohydrate Distribution
    df_nutrition = pd.read_csv(NUTRITION_PATH)
    # The carbohydrate column needs to be identified and cleaned
    # Assuming it is one of the columns that can be converted to numeric
    carb_col = None
    for col in df_nutrition.columns:
        if 'carb' in col.lower():
             carb_col = col
             break

    if carb_col:
        df_nutrition['carbs_numeric'] = pd.to_numeric(df_nutrition[carb_col], errors='coerce')
        df_nutrition.dropna(subset=['carbs_numeric'], inplace=True)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df_nutrition['carbs_numeric'], bins=50, kde=True, color='skyblue')
        plt.title('Distribution of Carbohydrates per 100g in Raw Data', fontsize=16)
        plt.xlabel('Carbohydrates (g)')
        plt.ylabel('Frequency of Ingredients')
        plt.axvline(10, color='red', linestyle='--', linewidth=2, label='Keto Threshold (10g)')
        plt.xlim(0, 100)
        plt.legend()
        plt.show()
    else:
        print("Could not identify a carbohydrate column in nutrition.csv for EDA.")


print(f"\n--- 2. Ground Truth EDA ---")
if not GROUND_TRUTH_PATH.exists():
    print("❌ GROUND TRUTH FILE NOT FOUND. Run generate_ground_truth.py first.")
else:
    df_gt = pd.read_csv(GROUND_TRUTH_PATH)
    
    # EDA 2.1: Distribution of Ground Truth Personas
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_gt, y='classification', order=df_gt['classification'].value_counts().index)
    plt.title('Distribution of Ground Truth Personas', fontsize=16)
    plt.xlabel('Number of Recipes')
    plt.ylabel('Classification Persona')
    plt.show() 