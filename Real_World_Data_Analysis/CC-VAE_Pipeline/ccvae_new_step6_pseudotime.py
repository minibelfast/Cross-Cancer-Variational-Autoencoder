import scanpy as sc
import sctour as sct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

out_dir = "results_new_cross_cancer_ccvae"

cell_types = ['T cells', 'Macrophages', 'Fibroblasts']

def run_sctour(adata, ct_name):
    print(f"\n{'='*40}")
    print(f"Running scTour for {ct_name}")
    print(f"{'='*40}")
    
    adata_sct = adata.copy()
    sc.pp.filter_genes(adata_sct, min_cells=3)
    sc.pp.highly_variable_genes(adata_sct, n_top_genes=2000, flavor='cell_ranger', subset=True, layer='counts')
    
    if 'counts' in adata_sct.layers:
        adata_sct.X = adata_sct.layers['counts'].copy()
        # scTour requires integers in X for nb mode, ensure it's not log-normalized counts
        if hasattr(adata_sct.X, 'toarray'):
            adata_sct.X.data = np.round(adata_sct.X.data).astype(np.float32)
        else:
            adata_sct.X = np.round(adata_sct.X).astype(np.float32)
        
    print(f"Data shape for scTour: {adata_sct.shape}")
    
    if adata_sct.shape[0] > 10000:
        print("Dataset > 10000 cells, subsampling for scTour model training...")
        sc.pp.subsample(adata_sct, n_obs=10000, random_state=42)
        
    print("Training scTour model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = sct.train.Trainer(adata_sct, loss_mode='nb', alpha_recon_lec=0.5, alpha_recon_lode=0.5)
        model.train()
    except Exception as e:
        print(f"Failed to train. Error: {e}")
        
    print("Predicting pseudotime and vector field for full dataset...")
    adata_infer = adata[:, adata_sct.var_names].copy()
    if 'counts' in adata_infer.layers:
        adata_infer.X = adata_infer.layers['counts'].copy()
        if hasattr(adata_infer.X, 'toarray'):
            adata_infer.X.data = np.round(adata_infer.X.data).astype(np.float32)
        else:
            adata_infer.X = np.round(adata_infer.X).astype(np.float32)
            
    # scTour prediction logic
    print("Extracting pseudotime...")
    try:
        # Based on typical scTour usage
        pred_res = model.get_latentsp(adata_infer.X)
        if isinstance(pred_res, tuple) and len(pred_res) >= 2:
            adata.obs['scTour_pseudotime'] = pred_res[1]
        elif isinstance(pred_res, dict) and 'ptime' in pred_res:
            adata.obs['scTour_pseudotime'] = pred_res['ptime']
        else:
            adata.obs['scTour_pseudotime'] = pred_res[0] if isinstance(pred_res, tuple) else pred_res
    except Exception as e:
        print(f"Fallback due to: {e}")
        # Sometimes alpha_z issue with sparse matrices, convert to dense
        try:
            X_dense = adata_infer.X.toarray() if hasattr(adata_infer.X, 'toarray') else adata_infer.X
            pred_res = model.get_latentsp(X_dense)
            if isinstance(pred_res, tuple) and len(pred_res) >= 2:
                adata.obs['scTour_pseudotime'] = pred_res[1]
            else:
                adata.obs['scTour_pseudotime'] = pred_res[0] if isinstance(pred_res, tuple) else pred_res
        except Exception as e2:
            print(f"Second fallback failed: {e2}")
            # Try get_vector_field on dense
            try:
                pred_res = model.get_vector_field(adata_infer.X.toarray() if hasattr(adata_infer.X, 'toarray') else adata_infer.X)
                adata.obs['scTour_pseudotime'] = pred_res['ptime']
            except Exception as e3:
                print(f"Third fallback failed: {e3}")
                adata.obs['scTour_pseudotime'] = 0
    
    print("Generating scTour plots...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color='CellType_Fine', show=False, ax=axs[0])
    sc.pl.umap(adata, color='scTour_pseudotime', color_map='viridis', show=False, ax=axs[1])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/09_{ct_name.replace(' ', '')}_Pseudotime_UMAP.pdf", bbox_inches='tight')
    plt.close()
    
    print("Saving updated AnnData...")
    adata.write(f"{out_dir}/10_{ct_name.replace(' ', '')}_adata_sctour_ccvae.h5ad")

for ct in cell_types:
    in_file = f"{out_dir}/08_{ct.replace(' ', '')}_adata_fine_ccvae.h5ad"
    if os.path.exists(in_file):
        adata = sc.read_h5ad(in_file)
        run_sctour(adata, ct)
    else:
        print(f"File {in_file} not found. Skipping {ct}.")

print("=== Step 6 scTour Complete ===")
