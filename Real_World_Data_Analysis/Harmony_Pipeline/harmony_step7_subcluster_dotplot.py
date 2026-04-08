import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer_harmony"

fine_markers = {
    'Macrophages': {
        'C1QC+ TAM (Phagocytic)': ['C1QA', 'C1QB', 'C1QC', 'TREM2'],
        'SPP1+ TAM (Angiogenic/Hypoxic)': ['SPP1', 'VEGFA', 'HIF1A', 'FN1'],
        'M1-like (Pro-inflammatory)': ['CXCL9', 'CXCL10', 'IL1B', 'TNF', 'IL6'],
        'Cycling Macrophages': ['MKI67', 'TOP2A', 'CDK1']
    },
    'Fibroblasts': {
        'myCAF (Myofibroblastic)': ['ACTA2', 'TAGLN', 'MMP2', 'COL1A1'],
        'iCAF (Inflammatory)': ['IL6', 'CXCL12', 'CCL2', 'CFD'],
        'apCAF (Antigen-presenting)': ['HLA-DRA', 'HLA-DRB1', 'CD74']
    },
    'T cells': {
        'CD8+ Exhausted T': ['CD8A', 'PDCD1', 'HAVCR2', 'LAG3', 'TOX'],
        'CD8+ Effector/Memory T': ['CD8A', 'GZMK', 'GZMB', 'PRF1', 'IFNG'],
        'CD4+ Treg': ['CD4', 'FOXP3', 'IL2RA', 'CTLA4'],
        'CD4+ Naive/Memory T': ['CD4', 'TCF7', 'SELL', 'CCR7', 'IL7R']
    }
}

cell_types = ['Macrophages', 'Fibroblasts', 'T cells']

for ct in cell_types:
    print(f"\n{'='*40}")
    print(f"Generating Dotplot for {ct}")
    print(f"{'='*40}")
    
    in_file = f"{out_dir}/07_{ct.replace(' ', '')}_adata_fine_harmony.h5ad"
    if not os.path.exists(in_file):
        print(f"{in_file} not found, skipping.")
        continue
        
    adata = sc.read_h5ad(in_file)
    
    # Check if CellType_Fine has Unknown, maybe filter it out for dotplot
    if 'Unknown' in adata.obs['CellType_Fine'].values:
        adata_sub = adata[adata.obs['CellType_Fine'] != 'Unknown'].copy()
    else:
        adata_sub = adata.copy()
        
    # Get valid markers
    markers = fine_markers[ct]
    valid_markers = {}
    for subtype, genes in markers.items():
        valid_genes = [g for g in genes if g in adata_sub.var_names]
        if valid_genes:
            valid_markers[subtype] = valid_genes
            
    # Plot
    # We will use figsize argument to make sure it's wide enough
    sc.pl.dotplot(adata_sub, valid_markers, groupby='CellType_Fine', standard_scale='var', 
                  show=False, figsize=(10, 5))
    
    # Save manually to the subfolder or main folder
    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_{ct.replace(' ', '')}_Subcluster_Markers_Dotplot.pdf", bbox_inches='tight')
    plt.close()

print("=== Step 7 Dotplot Complete ===")
