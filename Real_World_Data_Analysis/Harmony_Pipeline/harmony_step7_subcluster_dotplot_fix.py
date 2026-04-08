import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer_harmony"

# Markers used in CC-VAE
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
    
    # DO NOT filter out 'Unknown'
    adata_sub = adata.copy()
        
    # Get valid markers
    markers = fine_markers[ct]
    valid_markers = {}
    for subtype, genes in markers.items():
        valid_genes = [g for g in genes if g in adata_sub.var_names]
        if valid_genes:
            valid_markers[subtype] = valid_genes
            
    # Convert 'CellType_Fine' to categorical to ensure 'Unknown' shows up even if it's rare
    if 'Unknown' in adata_sub.obs['CellType_Fine'].unique():
        categories = list(markers.keys()) + ['Unknown']
        # Filter to only categories that exist in the data
        categories = [c for c in categories if c in adata_sub.obs['CellType_Fine'].unique()]
        adata_sub.obs['CellType_Fine'] = pd.Categorical(adata_sub.obs['CellType_Fine'], categories=categories, ordered=True)
            
    # Plot
    sc.pl.dotplot(adata_sub, valid_markers, groupby='CellType_Fine', standard_scale='var', 
                  show=False, figsize=(12, 6))
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_{ct.replace(' ', '')}_Subcluster_Markers_Dotplot.pdf", bbox_inches='tight')
    plt.close()

print("=== Step 7 Dotplot Fixed Complete ===")
