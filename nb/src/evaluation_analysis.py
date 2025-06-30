import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from IPython.display import display

# Updated path to the real evaluation output
RESULTS_PATH = Path("nb/src/data/evaluation_results/misclassifications.csv")

if not RESULTS_PATH.exists():
    print(f"❌ Evaluation results file not found at {RESULTS_PATH}.")
    print("Please run `run_final_evaluation.py` to generate the results.")
else:
    df_results = pd.read_csv(RESULTS_PATH)
    
    # --- Analysis 4.1: Confusion Matrices ---
    # We can reconstruct the confusion matrix from the true and predicted labels
    # in the misclassifications file, but a more direct approach is to
    # simply display the errors. The confusion matrix is already saved as a PNG.
    
    print("--- Analysis of Misclassifications ---")

    # --- Analysis 4.2: Detailed Error Breakdown ---
    if not df_results.empty:
        print(f"Displaying {len(df_results)} misclassified recipes for review:")
        display(df_results[['ingredients', 'is_keto_true', 'is_keto_pred', 'is_vegan_true', 'is_vegan_pred']])
    else:
        print("✅ No misclassifications found in the latest run!")

    # You can also load and display the confusion matrix images directly
    keto_cm_path = RESULTS_PATH.parent / "keto_confusion_matrix.png"
    if keto_cm_path.exists():
        print("\n--- Keto Confusion Matrix ---")
        # from IPython.display import Image
        # display(Image(filename=str(keto_cm_path)))
        # The above line works in notebooks, for scripts we just print the path
        print(f"Keto confusion matrix saved at: {keto_cm_path}")

    vegan_cm_path = RESULTS_PATH.parent / "vegan_confusion_matrix.png"
    if vegan_cm_path.exists():
        print("\n--- Vegan Confusion Matrix ---")
        print(f"Vegan confusion matrix saved at: {vegan_cm_path}")