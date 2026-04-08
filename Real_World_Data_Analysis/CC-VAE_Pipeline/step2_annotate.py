import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer"
sc.settings.figdir = out_dir

adata_path = f"{out_dir}/integrated_adata.h5ad"

print("=== Loading integrated data ===")
adata = sc.read_h5ad(adata_path)

# 手动确定的聚类映射字典 (待填充)
cluster_mapping = {
    '0': 'Epithelial/Cancer',
    '1': 'T cells',
    '2': 'Epithelial/Cancer',
    '3': 'Macrophages',
    '4': 'T cells',
    '5': 'Macrophages',
    '6': 'Unknown',
    '7': 'Endothelial',
    '8': 'Unknown',
    '9': 'Fibroblasts',
    '10': 'Macrophages',
    '11': 'Fibroblasts',
    '12': 'B cells',
    '13': 'Unknown',
    '14': 'T cells',
    '15': 'Myeloid',
    '16': 'Epithelial/Cancer',
    '17': 'T cells',
    '18': 'T cells',
    '19': 'Macrophages',
    '20': 'Myeloid',
    '21': 'B cells',
    '22': 'Myeloid',
    '23': 'Fibroblasts',
    '24': 'T cells'
}

print("=== Applying manual annotations ===")
adata.obs['CellType_Broad'] = adata.obs['leiden_ccvae'].map(cluster_mapping)
# 针对未注释的类，保留原名
adata.obs['CellType_Broad'] = adata.obs['CellType_Broad'].fillna(adata.obs['leiden_ccvae'])

print("=== Saving UMAP ===")
sc.pl.umap(adata, color='CellType_Broad', save="_03_All_Cells_CellType.pdf", show=False)

# 经典大类标志基因字典
major_markers = {
    'T cells': ['CD3D', 'CD3E', 'CD2'],
    'Macrophages': ['CD68', 'CD163', 'C1QA', 'LYZ'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN', 'ACTA2'],
    'Epithelial/Cancer': ['EPCAM', 'KRT8', 'KRT18'],
    'B cells': ['CD79A', 'MS4A1'],
    'Endothelial': ['PECAM1', 'VWF', 'ENG'],
    'NK cells': ['NKG7', 'GNLY']
}

print("=== Saving Dotplot ===")
sc.pl.dotplot(adata, {k: [g for g in v if g in adata.var_names] for k, v in major_markers.items() if [g for g in v if g in adata.var_names]}, groupby='CellType_Broad', standard_scale='var', 
              save="_04_All_Cells_Marker.pdf", show=False)

print("=== Saving annotated adata ===")
adata.write(f"{out_dir}/integrated_adata_annotated.h5ad")
print("=== Step 2 Complete ===")
