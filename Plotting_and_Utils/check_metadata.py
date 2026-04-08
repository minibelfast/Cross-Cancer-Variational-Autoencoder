import scanpy as sc
import pandas as pd

adata_mac = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Macrophages_adata_sctour_ccvae_fixed.h5ad")
print("Available columns in obs:")
print(list(adata_mac.obs.columns))

print("\nValue counts for potential sample columns:")
for col in ['sample', 'Sample', 'patient', 'Patient', 'batch', 'Dataset', 'Cancer_Type', 'orig.ident', 'Library']:
    if col in adata_mac.obs.columns:
        print(f"\n--- {col} ---")
        print(f"Number of unique values: {adata_mac.obs[col].nunique()}")
        print(adata_mac.obs[col].value_counts().head(5))

