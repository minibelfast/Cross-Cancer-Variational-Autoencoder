import scanpy as sc
import scTour as sct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

out_dir = "results_new_cross_cancer_harmony"

cell_types = ['T cells', 'Macrophages', 'Fibroblasts']

# Function to run scTour with subsampling if dataset is too large
def run_sctour(adata, ct_name):
    print(f"\n{'='*40}")
    print(f"Running scTour for {ct_name}")
    print(f"{'='*40}")
    
    # Preprocess specifically for scTour
    adata_sct = adata.copy()
    sc.pp.filter_genes(adata_sct, min_cells=3)
    sc.pp.highly_variable_genes(adata_sct, n_top_genes=2000, flavor='cell_ranger', subset=True, layer='counts')
    
    # scTour expects counts, and it's better to provide raw counts or just X
    # but since our X is log1p normalized, we can try to use raw counts if available
    if 'counts' in adata_sct.layers:
        adata_sct.X = adata_sct.layers['counts'].copy()
        
    print(f"Data shape for scTour: {adata_sct.shape}")
    
    # Subsample if memory is an issue
    if adata_sct.shape[0] > 10000:
        print("Dataset > 10000 cells, subsampling for scTour model training...")
        sc.pp.subsample(adata_sct, n_obs=10000, random_state=42)
        
    print("Training scTour model...")
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = sct.train.Trainer(adata_sct, loss_mode='nb', alpha_recon_lec=0.5, alpha_recon_lode=0.5)
        model.train(device=device)
    except Exception as e:
        print(f"Failed to train on {device}. Error: {e}. Falling back to CPU...")
        model = sct.train.Trainer(adata_sct, loss_mode='nb', alpha_recon_lec=0.5, alpha_recon_lode=0.5)
        model.train(device='cpu')
        
    print("Predicting pseudotime and vector field for full dataset...")
    # For inference, use the highly variable genes only
    adata_infer = adata[:, adata_sct.var_names].copy()
    if 'counts' in adata_infer.layers:
        adata_infer.X = adata_infer.layers['counts'].copy()
        
    pred_res = model.get_vector_field(adata_infer.X)
    
    # Add results back to full adata
    adata.obs['scTour_pseudotime'] = pred_res['ptime']
    
    # Save plots
    print("Generating scTour plots...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color='CellType_Fine', show=False, ax=axs[0])
    sc.pl.umap(adata, color='scTour_pseudotime', color_map='viridis', show=False, ax=axs[1])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/09_{ct_name.replace(' ', '')}_Pseudotime_UMAP.pdf", bbox_inches='tight')
    plt.close()
    
    # Save the updated anndata
    print("Saving updated AnnData...")
    adata.write(f"{out_dir}/10_{ct_name.replace(' ', '')}_adata_sctour_harmony.h5ad")

for ct in cell_types:
    in_file = f"{out_dir}/08_{ct.replace(' ', '')}_adata_fine_harmony.h5ad"
    if os.path.exists(in_file):
        adata = sc.read_h5ad(in_file)
        run_sctour(adata, ct)
    else:
        print(f"File {in_file} not found. Skipping {ct}.")

print("=== Step 6 scTour Complete ===")
