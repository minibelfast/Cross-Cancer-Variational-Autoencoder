import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sctour as sct

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"
sc.settings.figdir = out_dir

cell_types = ['Macrophages', 'Fibroblasts', 'T cells']

for ct in cell_types:
    print(f"\n{'='*40}")
    print(f"Running scTour for {ct}")
    print(f"{'='*40}")
    
    in_file = f"{out_dir}/07_{ct.replace(' ', '')}_adata_fine_harmony.h5ad"
    if not os.path.exists(in_file):
        print(f"{in_file} not found, skipping.")
        continue
        
    adata = sc.read_h5ad(in_file)
    
    if 'highly_variable' in adata.var:
        adata_sct = adata[:, adata.var['highly_variable']].copy()
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=False, layer='counts')
        adata_sct = adata[:, adata.var['highly_variable']].copy()
        
    if adata_sct.shape[0] > 10000:
        sc.pp.subsample(adata_sct, n_obs=10000, random_state=42)
        print(f"Subsampled to {adata_sct.shape[0]} cells for scTour training.")
        
    print("Training scTour model...")
    tn = sct.train.Trainer(adata_sct, loss_mode='mse', alpha_recon_lec=0.5, alpha_recon_lode=0.5, nepoch=20)
    tn.train()
        
    print("Extracting pseudotime and vector field...")
    adata_sct.obs['ptime'] = tn.get_time()
    
    try:
        mix_zs, zs, pred_zs = tn.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
        adata_sct.obsm['X_T'] = mix_zs
        vf = tn.get_vector_field(adata_sct.obs['ptime'].values, mix_zs)
    except TypeError:
        pass
        
    if 'X_T' not in adata_sct.obsm:
        if 'X_T' not in adata_sct.obsm:
            adata_sct.obsm['X_T'] = tn.get_latentsp()
        vf = tn.get_vector_field(adata_sct.obs['ptime'].values, adata_sct.obsm['X_T'])

    adata_sct.obsm['X_VF'] = vf
    
    adata_sct.obsm['X_VF'] = np.nan_to_num(adata_sct.obsm['X_VF'])
    adata_sct.obsm['X_T'] = np.nan_to_num(adata_sct.obsm['X_T'])
    
    adata.obs['ptime'] = np.nan
    adata.obs.loc[adata_sct.obs_names, 'ptime'] = adata_sct.obs['ptime']
    
    sc.pl.umap(adata_sct, color=['CellType_Fine', 'ptime'], save=f"_{ct.replace(' ', '')}_scTour_ptime.png", show=False)
    
    try:
        sct.vf.plot_vector_field(adata_sct, zs_key='X_T', vf_key='X_VF', use_rep_neigh='X_T', color='CellType_Fine', 
                                 show=False, save=True, outdir=out_dir, title=f"{ct} Trajectory")
    except Exception as e:
        print(f"Error plotting vector field as PDF, trying PNG: {e}")
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            sct.vf.plot_vector_field(adata_sct, zs_key='X_T', vf_key='X_VF', use_rep_neigh='X_T', color='CellType_Fine', 
                                     show=False, ax=ax, title=f"{ct} Trajectory")
            plt.savefig(f"{out_dir}/{ct}_Trajectory.png", bbox_inches='tight')
            plt.close()
        except Exception as e2:
            print(f"Failed to plot trajectory: {e2}")
            
    adata.write(f"{out_dir}/08_{ct.replace(' ', '')}_adata_sctour_harmony.h5ad")

print("=== Step 4 Complete ===")
