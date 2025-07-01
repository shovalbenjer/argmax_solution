import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import numpy as np

# Use a modern, publication-quality plotting style
sns.set_theme(style="whitegrid", palette="viridis")

# --- Configuration ---
NUTRITION_PATH = Path("nb/src/raw_data/nutrition.csv")
GROUND_TRUTH_PATH = Path("nb/src/data/ground_truth_sample.csv") # Assumes this is generated
EMBEDDINGS_PATH = Path("nb/src/data/ingredient_embeddings.npy") # Assumes this is generated

# --- 1. Raw Nutrition Data EDA ---
print("--- 1. Comprehensive EDA on Raw Nutrition Data ---")
if not NUTRITION_PATH.exists():
    print(f"❌ Nutrition data file not found at {NUTRITION_PATH}.")
else:
    df_nutrition = pd.read_csv(NUTRITION_PATH)
    
    # Identify the carbohydrate column robustly
    carb_col = next((col for col in df_nutrition.columns if 'carb' in col.lower()), None)
    
    if carb_col:
        df_nutrition['carbs_numeric'] = pd.to_numeric(df_nutrition[carb_col], errors='coerce')
        df_nutrition.dropna(subset=['carbs_numeric'], inplace=True)

        print("\n📝 Summary Statistics for Carbohydrates (per 100g):")
        summary_stats = df_nutrition['carbs_numeric'].describe()
        print(summary_stats)

        # Generate a figure with multiple subplots for a dashboard-style view
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Analysis of Carbohydrate Distribution in Ingredients', fontsize=20, weight='bold')

        # Distribution Plot (Histogram + KDE)
        sns.histplot(df_nutrition['carbs_numeric'], bins=50, kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title('Distribution of Carbohydrates', fontsize=16)
        axes[0].set_xlabel('Carbohydrates (g) per 100g')
        axes[0].set_ylabel('Frequency of Ingredients')
        axes[0].axvline(10, color='r', linestyle='--', linewidth=2.5, label='Keto Threshold (10g)')
        axes[0].set_xlim(0, 100)
        axes[0].legend()

        # Outlier Detection (Boxenplot for enhanced detail)
        sns.boxenplot(x=df_nutrition['carbs_numeric'], ax=axes[1], color="lightgreen")
        axes[1].set_title('Outlier and Distribution Analysis via Boxenplot', fontsize=16)
        axes[1].set_xlabel('Carbohydrates (g) per 100g')
        axes[1].axvline(10, color='r', linestyle='--', linewidth=2.5)
        axes[1].set_xlim(0, 100)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- 2. Ground Truth Dataset Analysis ---
print("\n--- 2. Analysis of Gemini-Generated Ground Truth Personas ---")
if not GROUND_TRUTH_PATH.exists():
    print("❌ GROUND TRUTH FILE NOT FOUND. Run ground truth generation first.")
else:
    df_gt = pd.read_csv(GROUND_TRUTH_PATH)
    
    # Assuming the ground truth CSV has 'is_vegan' and 'is_keto' columns
    if 'is_vegan' in df_gt.columns and 'is_keto' in df_gt.columns:
        df_gt['classification_persona'] = 'Neither'
        df_gt.loc[df_gt['is_vegan'] == True, 'classification_persona'] = 'Vegan'
        df_gt.loc[df_gt['is_keto'] == True, 'classification_persona'] = 'Keto'
        df_gt.loc[(df_gt['is_vegan'] == True) & (df_gt['is_keto'] == True), 'classification_persona'] = 'Both'
        
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df_gt, x='classification_persona', order=['Vegan', 'Keto', 'Both', 'Neither'], palette="magma")
        plt.title('Distribution of Dietary Personas in Ground Truth Dataset', fontsize=18, weight='bold')
        plt.xlabel('Classification Persona', fontsize=14)
        plt.ylabel('Number of Recipes', fontsize=14)
        plt.show()

# --- 3. SOTA Visualization: Ingredient Embedding Space ---
print("\n--- 3. SOTA Validation: Visualizing Ingredient Embeddings with UMAP ---")
if not EMBEDDINGS_PATH.exists():
    print(f"❌ Ingredient embeddings not found at {EMBEDDINGS_PATH}. Cannot generate UMAP plot.")
else:
    print("Loading embeddings and generating UMAP plot... (this may take a moment)")
    # UMAP is generally faster and better at preserving global structure than t-SNE. [1, 2]
    try:
        from umap import UMAP
        embeddings = np.load(EMBEDDINGS_PATH)
        # Assuming you have a corresponding df_nutrition with a 'category' column
        ingredient_categories = df_nutrition.head(len(embeddings))['Food group'] # Example category

        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            hue=ingredient_categories,
            palette=sns.color_palette("hsv", len(ingredient_categories.unique())),
            s=50,
            alpha=0.7
        )
        plt.title('UMAP Projection of Ingredient Embeddings', fontsize=20, weight='bold')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(title='Ingredient Category', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        print("✅ UMAP plot generated. Check for meaningful clusters to validate embedding quality.")
    except ImportError:
        print("❌ UMAP not installed. Please run 'pip install umap-learn' for this visualization.")
    except Exception as e:
        print(f"An error occurred during UMAP visualization: {e}")