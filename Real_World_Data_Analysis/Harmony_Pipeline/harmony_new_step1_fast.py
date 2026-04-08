import scanpy as sc
import pandas as pd
import glob
import os
import scanpy.external as sce
import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

# Since CC-VAE already integrated this, we can just load the merged adata and re-integrate using Harmony!
print("Loading already merged adata from CC-VAE to save time...")
adata = sc.read_h5ad("results_new_cross_cancer/integrated_adata.h5ad")
print(f"Loaded adata shape: {adata.shape}")

# Clean up CC-VAE specific stuff if any
for col in ['leiden_ccvae']:
    if col in adata.obs.columns:
        del adata.obs[col]
if 'X_ccvae' in adata.obsm.keys():
    del adata.obsm['X_ccvae']

print("Highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', layer='counts')

print("PCA...")
sc.tl.pca(adata, svd_solver='arpack')

print("Harmony integration...")
sce.pp.harmony_integrate(adata, 'Batch')

print("Clustering and UMAP...")
sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden_harmony')

print("Saving integration plots...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata, color='Batch', show=False, ax=axs[1], legend_loc='none')
sc.pl.umap(adata, color='leiden_harmony', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/01_All_Cells_UMAP_Harmony.pdf", bbox_inches='tight')

print("Calculating Markers...")
sc.tl.rank_genes_groups(adata, groupby='leiden_harmony', method='wilcoxon')
markers_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
markers_df.to_csv(f"{out_dir}/02_All_Cells_Cluster_Markers_Harmony.csv", index=False)

print("Saving AnnData...")
adata.write(f"{out_dir}/integrated_adata_harmony.h5ad")
print("=== Step 1 Fast Complete ===")
