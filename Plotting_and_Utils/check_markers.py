import pandas as pd
import os

out_dir = "results_real_cross_cancer"
marker_file = f"{out_dir}/02_All_Cells_Cluster_Markers.csv"

if os.path.exists(marker_file):
    df = pd.read_csv(marker_file)
    print("=== Top 10 Markers per Cluster ===")
    for col in df.columns:
        top_genes = df[col].head(10).tolist()
        print(f"Cluster {col}: {', '.join(top_genes)}")
else:
    print("Marker file not found yet.")
