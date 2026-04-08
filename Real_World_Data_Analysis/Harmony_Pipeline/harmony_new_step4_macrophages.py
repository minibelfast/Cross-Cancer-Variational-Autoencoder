import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scanpy.external as sce

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

out_dir = "results_new_cross_cancer_harmony"

fine_markers = {
    'Macrophages': {
        'M1 Macrophages': ['CD86', 'CXCL9', 'CXCL10', 'IL1B'],
        'M2 Macrophages': ['CD163', 'MRC1', 'TREM2', 'CD206', 'VSIG4'],
        'TAMs (Tumor-Associated)': ['SPP1', 'APOE', 'TREM2', 'C1QA'],
        'Cycling Macrophages': ['MKI67', 'TOP2A', 'PCNA']
    }
}

ct = 'Macrophages'
print(f"\n{'='*40}")
print(f"Processing {ct}")
print(f"{'='*40}")

in_file = f"{out_dir}/05_{ct.replace(' ', '')}_adata_harmony.h5ad"
if not os.path.exists(in_file):
    print(f"File {in_file} not found. Please run step 2 first.")
    exit(1)

adata = sc.read_h5ad(in_file)
print(f"Original shape: {adata.shape}")

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=False, layer='counts')
sc.tl.pca(adata, svd_solver='arpack')

print("Running Harmony...")
sce.pp.harmony_integrate(adata, 'Batch')

print("Clustering...")
sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=20)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.6, key_added='leiden_sub')

print("Scoring subtypes...")
markers = fine_markers[ct]
for subtype, genes in markers.items():
    valid_genes = [g for g in genes if g in adata.var_names]
    if valid_genes:
        sc.tl.score_genes(adata, gene_list=valid_genes, score_name=f'{subtype}_score')
        
score_cols = [f'{subtype}_score' for subtype in markers.keys() if f'{subtype}_score' in adata.obs.columns]
cluster_scores = adata.obs.groupby('leiden_sub')[score_cols].mean()

cluster_mapping = {}
for cluster in cluster_scores.index:
    scores = cluster_scores.loc[cluster]
    if scores.max() > 0:
        best_subtype = scores.idxmax().replace('_score', '')
        cluster_mapping[cluster] = best_subtype
    else:
        cluster_mapping[cluster] = 'Unknown'
        
adata.obs['CellType_Fine'] = adata.obs['leiden_sub'].map(cluster_mapping)

print(f"Cluster mapping for {ct}:")
for k, v in cluster_mapping.items():
    print(f"Cluster {k}: {v}")
    
print("Saving UMAPs...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata, color='leiden_sub', show=False, ax=axs[1])
sc.pl.umap(adata, color='CellType_Fine', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/06_{ct.replace(' ', '')}_Subcluster_UMAP.pdf", bbox_inches='tight')
plt.close()

print("Calculating Subcluster Markers...")
sc.tl.rank_genes_groups(adata, groupby='CellType_Fine', method='wilcoxon')
markers_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
markers_df.to_csv(f"{out_dir}/07_{ct.replace(' ', '')}_Subcluster_Markers.csv", index=False)

print("Saving annotated data...")
adata.write(f"{out_dir}/08_{ct.replace(' ', '')}_adata_fine_harmony.h5ad")

print(f"=== Step 4 {ct} Complete ===")
