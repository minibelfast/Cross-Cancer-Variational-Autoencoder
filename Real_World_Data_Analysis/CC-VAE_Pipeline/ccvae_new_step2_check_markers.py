import pandas as pd
import sys

out_dir = "results_new_cross_cancer_ccvae"
marker_file = f"{out_dir}/02_All_Cells_Cluster_Markers_CCVAE.csv"

try:
    df = pd.read_csv(marker_file)
except FileNotFoundError:
    print(f"File {marker_file} not found. Step 1 is not complete yet.")
    sys.exit(0)

print("Top 10 markers for each CC-VAE cluster:")
for col in df.columns:
    top_genes = df[col].head(10).tolist()
    print(f"Cluster {col}: {', '.join(top_genes)}")

print("\n--- Based on these markers, you should create the cluster_mapping dict in ccvae_new_step2_annotate.py ---")
