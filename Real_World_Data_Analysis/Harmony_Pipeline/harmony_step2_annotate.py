import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer_harmony"
sc.settings.figdir = out_dir

adata_path = f"{out_dir}/integrated_adata_harmony.h5ad"

print("=== Loading integrated data ===")
adata = sc.read_h5ad(adata_path)

major_markers = {
    'T cells': ['CD3D', 'CD3E', 'CD2'],
    'Macrophages': ['CD68', 'CD163', 'C1QA', 'LYZ'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN', 'ACTA2'],
    'Epithelial/Cancer': ['EPCAM', 'KRT8', 'KRT18'],
    'B cells': ['CD79A', 'MS4A1'],
    'Endothelial': ['PECAM1', 'VWF', 'ENG'],
    'NK cells': ['NKG7', 'GNLY']
}

print("=== Applying automated annotations based on markers ===")
# 对每个大类进行打分
for ct, genes in major_markers.items():
    valid_genes = [g for g in genes if g in adata.var_names]
    if valid_genes:
        sc.tl.score_genes(adata, gene_list=valid_genes, score_name=f'{ct}_score')

# 提取打分结果并为每个聚类分配最高分的细胞类型
score_cols = [f'{ct}_score' for ct in major_markers.keys()]
cluster_scores = adata.obs.groupby('leiden_harmony')[score_cols].mean()

cluster_mapping = {}
for cluster in cluster_scores.index:
    scores = cluster_scores.loc[cluster]
    if scores.max() > 0:  # 只有得分大于0才分配
        best_ct = scores.idxmax().replace('_score', '')
        cluster_mapping[cluster] = best_ct
    else:
        cluster_mapping[cluster] = 'Unknown'

print("Cluster mapping:")
for k, v in cluster_mapping.items():
    print(f"Cluster {k}: {v}")

adata.obs['CellType_Broad'] = adata.obs['leiden_harmony'].map(cluster_mapping)

print("=== Saving UMAP ===")
sc.pl.umap(adata, color='CellType_Broad', save="_03_All_Cells_CellType_Harmony.pdf", show=False)

print("=== Saving Dotplot ===")
valid_markers = {k: [g for g in v if g in adata.var_names] for k, v in major_markers.items()}
sc.pl.dotplot(adata, valid_markers, groupby='CellType_Broad', standard_scale='var', 
              save="_04_All_Cells_Marker_Harmony.pdf", show=False)

print("=== Saving annotated adata ===")
adata.write(f"{out_dir}/integrated_adata_annotated_harmony.h5ad")

# 为下游分析分别保存巨噬细胞，成纤维细胞，T细胞的子集
print("=== Saving subsets for downstream analysis ===")
for ct in ['Macrophages', 'Fibroblasts', 'T cells']:
    if ct in adata.obs['CellType_Broad'].values:
        sub_adata = adata[adata.obs['CellType_Broad'] == ct].copy()
        sub_adata.write(f"{out_dir}/05_{ct.replace(' ', '')}_adata_harmony.h5ad")
        print(f"Saved {ct} subset.")

print("=== Step 2 Complete ===")
