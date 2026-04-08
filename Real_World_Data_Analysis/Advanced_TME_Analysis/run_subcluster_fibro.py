import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model import CrossCancerVAE
from integration_benchmarks import run_benchmarks

def subcluster_fibroblasts(n_organs, fc):
    print(f"\n--- Subclustering Fibroblast/CAF for Organs={n_organs}, FC={fc} ---")
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    # 1. Load Data
    meta_path = os.path.join(data_dir, f"{prefix}_metadata.csv")
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    
    obs = pd.read_csv(meta_path, index_col=0)
    expr_df = pd.read_csv(expr_path, index_col=0)
    
    # We need the raw counts for scVI/CC-VAE
    counts_path = os.path.join(data_dir, f"{prefix}_counts.csv")
    counts_df = pd.read_csv(counts_path, index_col=0)
    
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    adata.layers['counts'] = counts_df.values
    
    # Subset to Fibroblast lineage
    adata_fibro = adata[adata.obs['BroadType'] == 'Fibroblast'].copy()
    print(f"Total Fibroblast/CAF cells: {adata_fibro.n_obs}")
    
    # Create explicit Fibroblast vs CAF label
    adata_fibro.obs['Fibroblast_vs_CAF'] = adata_fibro.obs['State'].apply(lambda x: 'CAF' if x == 'Tumor' else 'Fibroblast')
    
    # Re-run HVG for the subset to focus on fibro-specific variation
    sc.pp.highly_variable_genes(adata_fibro, n_top_genes=500, flavor='cell_ranger')
    
    # 3. Run Methods
    embeddings = {}
    
    # CC-VAE
    print("Running CC-VAE...")
    try:
        cc_vae = CrossCancerVAE(adata_fibro, batch_key='Patient_Batch')
        cc_vae.train(max_epochs=100)
        embeddings['CC-VAE'] = cc_vae.get_latent_representation()
    except Exception as e:
        print(f"CC-VAE failed: {e}")
        
    # Benchmarks
    print("Running Benchmarks...")
    try:
        bench_res = run_benchmarks(adata_fibro, methods=['Harmony', 'scVI', 'BBKNN'], batch_key='Patient_Batch')
        for m in ['Uncorrected', 'Harmony', 'scVI', 'BBKNN']:
            if m in bench_res:
                embeddings[m] = bench_res[m]['emb']
    except Exception as e:
        print(f"Benchmarks failed: {e}")
        
    # 4. Cluster and Plot
    methods = ['Uncorrected', 'Harmony', 'BBKNN', 'scVI', 'CC-VAE']
    
    out_dir = os.path.join(data_dir, "Fibroblast_Subcluster_UMAPs")
    os.makedirs(out_dir, exist_ok=True)
    
    for method in methods:
        if method not in embeddings:
            continue
            
        print(f"Processing {method}...")
        temp_adata = adata_fibro.copy()
        temp_adata.obsm['X_emb'] = embeddings[method]
        
        # Build graph
        sc.pp.neighbors(temp_adata, use_rep='X_emb', n_neighbors=15)
        sc.tl.umap(temp_adata)
        
        # Tune Leiden resolution to get exactly or close to 8 clusters
        target_clusters = 8
        best_res = 1.0
        best_diff = float('inf')
        best_clusters = None
        
        # Binary search for resolution
        res_min = 0.01
        res_max = 3.0
        for _ in range(15):
            res = (res_min + res_max) / 2
            sc.tl.leiden(temp_adata, resolution=res, key_added='leiden')
            n_clusters = len(temp_adata.obs['leiden'].unique())
            
            diff = abs(n_clusters - target_clusters)
            if diff < best_diff:
                best_diff = diff
                best_res = res
                best_clusters = temp_adata.obs['leiden'].copy()
                
            if n_clusters == target_clusters:
                break
            elif n_clusters < target_clusters:
                res_min = res
            else:
                res_max = res
                
        # Apply the best clustering found
        temp_adata.obs['leiden'] = best_clusters
        adata_fibro.obs[f'Subcluster_{method}'] = best_clusters
        print(f"  -> {method} clustered into {len(temp_adata.obs['leiden'].unique())} clusters (Target: 8, Res: {best_res:.3f})")
        
        # Plot UMAP
        sc.settings.set_figure_params(dpi=150, figsize=(5, 5))
        
        # We need to plot 4 labels: leiden, Fibroblast_vs_CAF, State, Organ
        # We will create a 2x2 grid of UMAPs for each method
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        labels_to_plot = ['leiden', 'Fibroblast_vs_CAF', 'State', 'Organ']
        titles = [f'{method} - Subclusters', f'{method} - Fibroblast vs CAF', f'{method} - State', f'{method} - Organ']
        
        for i, (label, title) in enumerate(zip(labels_to_plot, titles)):
            sc.pl.umap(temp_adata, color=label, ax=axs[i], show=False, title=title, s=50)
            
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"UMAP_Fibroblast_{method}_{prefix}.pdf")
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        
    # Save the subclustering labels
    out_meta = os.path.join(data_dir, f"{prefix}_Fibroblast_Subcluster_Metadata.csv")
    adata_fibro.obs.to_csv(out_meta)
    print(f"Saved subclustering metadata to {out_meta}")
        
if __name__ == "__main__":
    subcluster_fibroblasts(4, 1.5)
    subcluster_fibroblasts(4, 2.0)
