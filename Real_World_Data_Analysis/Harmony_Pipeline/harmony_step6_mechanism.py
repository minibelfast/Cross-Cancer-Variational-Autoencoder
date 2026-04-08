import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

out_dir = "results_real_cross_cancer_harmony"
os.makedirs(f"{out_dir}/mechanism", exist_ok=True)

try:
    mac_prop = pd.read_csv(f"{out_dir}/macrophages/Macrophages_Proportions_by_Cancer.csv", index_col=0)
    fib_prop = pd.read_csv(f"{out_dir}/fibroblasts/Fibroblasts_Proportions_by_Cancer.csv", index_col=0)
    t_prop = pd.read_csv(f"{out_dir}/tcells/T cells_Proportions_by_Cancer.csv", index_col=0)

    key_subtypes = {}
    for col in mac_prop.columns:
        if col != 'Unknown': key_subtypes[f'Mac: {col}'] = mac_prop[col]
    for col in fib_prop.columns:
        if col != 'Unknown': key_subtypes[f'Fib: {col}'] = fib_prop[col]
    for col in t_prop.columns:
        if col != 'Unknown': key_subtypes[f'T: {col}'] = t_prop[col]
    
    df = pd.DataFrame(key_subtypes)
    
    if not df.empty:
        df.to_csv(f"{out_dir}/mechanism/PanCancer_All_Subtypes_Proportions.csv")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='viridis', fmt=".1f")
        plt.title('Proportions of All Discovered Subtypes Across Cancers (%)')
        plt.ylabel('Cancer Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/mechanism/PanCancer_All_Subtypes_Heatmap.pdf", bbox_inches='tight')
        plt.close()

    print("=== Step 6 Mechanism Summary Complete ===")
except Exception as e:
    print(f"Error in mechanism summary: {e}")
