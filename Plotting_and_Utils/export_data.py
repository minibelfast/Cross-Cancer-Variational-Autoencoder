import os
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from simulation import generate_simulation_data
from model import CrossCancerVAE
from integration_benchmarks import run_benchmarks

def export_condition(n_organs, fc, output_dir):
    print(f"Exporting data for Organs={n_organs}, FC={fc}...")
    
    # 1. Generate Data
    # To ensure reproducibility for this export, we can set a seed, but here we just generate one instance.
    np.random.seed(42)
    adata = generate_simulation_data(n_cells=2000, n_organs=n_organs, fc=fc)
    
    # Preprocess
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    
    # Run Methods to get clusters
    embeddings = {}
    
    # CC-VAE
    cc_vae = CrossCancerVAE(adata, batch_key='Patient_Batch')
    cc_vae.train(max_epochs=100) # Enough to converge
    embeddings['CC-VAE'] = cc_vae.get_latent_representation()
    
    # Benchmarks
    bench_res = run_benchmarks(adata, methods=['Harmony', 'scVI', 'BBKNN'], batch_key='Patient_Batch')
    
    # Collect results
    obs_export = adata.obs[['BroadType', 'Detailed_Subtype', 'State', 'Organ', 'Patient_Batch']].copy()
    
    for method in ['Uncorrected', 'Harmony', 'scVI', 'BBKNN']:
        if method in bench_res:
            embeddings[method] = bench_res[method]['emb']
            
    # Calculate clusters for each method
    for method, emb in embeddings.items():
        temp_adata = adata.copy()
        temp_adata.obsm['X_emb'] = emb
        sc.pp.neighbors(temp_adata, use_rep='X_emb')
        sc.tl.leiden(temp_adata, key_added=f'Cluster_{method}')
        obs_export[f'Cluster_{method}'] = temp_adata.obs[f'Cluster_{method}']
        
    # Save Data
    prefix = f"Organs{n_organs}_FC{fc}"
    
    # 1. Save Metadata (Cell Info + Clusters)
    obs_file = os.path.join(output_dir, f"{prefix}_metadata.csv")
    obs_export.to_csv(obs_file)
    
    # 2. Save Expression Matrix (Normalized Log Counts)
    # To save space and keep it readable, we save the dense matrix to CSV
    expr_file = os.path.join(output_dir, f"{prefix}_expression.csv")
    expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    expr_df.to_csv(expr_file)
    
    # 3. Save raw counts as well just in case
    counts_file = os.path.join(output_dir, f"{prefix}_counts.csv")
    counts_df = pd.DataFrame(adata.layers['counts'], index=adata.obs_names, columns=adata.var_names)
    counts_df.to_csv(counts_file)
    
    print(f"Saved {prefix} to {output_dir}")

if __name__ == "__main__":
    out_dir = "./exported_simulation_data"
    os.makedirs(out_dir, exist_ok=True)
    export_condition(4, 1.5, out_dir)
    export_condition(4, 2.0, out_dir)
