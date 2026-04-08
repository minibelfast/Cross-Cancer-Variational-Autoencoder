import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from simulation import generate_simulation_data
from model import CrossCancerVAE
from integration_benchmarks import run_benchmarks
from evaluation_advanced import evaluate_integration_rigorous, save_fair_data
from visualization import FigureGenerator

def run_simulation_experiment(output_dir):
    """
    Run the full simulation benchmark as described in the paper.
    Parameters:
    - Clusters: 2, 3, 4
    - Fold Change: 1.2, 1.5, 2.0
    Methods:
    - CC-VAE (Proxy for DscSTAR in this reproduction)
    - Harmony
    - scVI
    - BBKNN
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    viz = FigureGenerator(output_dir)
    
    # Simulation Parameters
    cluster_counts = [2, 3, 4]
    fold_changes = [1.2, 1.5, 2.0]
    
    # Store aggregate results
    all_metrics = pd.DataFrame()
    
    for n_clust in cluster_counts:
        for fc in fold_changes:
            sim_id = f"Clust{n_clust}_FC{fc}"
            print(f"\n--- Running Simulation: {sim_id} ---")
            
            # 1. Generate Data
            adata = generate_simulation_data(n_cells=2000, n_clusters=n_clust, fc=fc)
            
            # Preprocess
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=1000)
            adata.raw = adata
            
            # Store counts layer for scVI/CC-VAE
            # Note: Simulation generates raw counts, but we normalized above. 
            # Re-generate or copy raw before norm? 
            # Let's fix: generate -> copy -> normalize
            # (Simulation code returns raw counts)
            # Reload to be safe/clean
            adata = generate_simulation_data(n_cells=2000, n_clusters=n_clust, fc=fc)
            adata.layers['counts'] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=False) # Mark HVGs
            
            # 2. Run Methods
            embeddings = {}
            
            # CC-VAE (Our DscSTAR proxy)
            print("  Running CC-VAE...")
            cc_vae = CrossCancerVAE(adata, batch_key='batch', n_latent=10)
            cc_vae.train(max_epochs=50) # Fast training for sim
            embeddings['DscSTAR (CC-VAE)'] = cc_vae.get_latent_representation()
            
            # Benchmarks
            print("  Running Benchmarks...")
            bench_res = run_benchmarks(adata, methods=['Harmony', 'scVI', 'BBKNN'], batch_key='batch')
            embeddings.update(bench_res)
            
            # 3. Evaluate
            for method, emb in embeddings.items():
                temp_adata = adata.copy()
                if method == 'BBKNN':
                    temp_adata.obsm['X_emb'] = emb # BBKNN returns UMAP/Graph, handled in eval
                else:
                    temp_adata.obsm['X_emb'] = emb
                    sc.pp.neighbors(temp_adata, use_rep='X_emb')
                    sc.tl.umap(temp_adata)
                
                # Metrics: ARI, ASW (Silhouette)
                metrics = evaluate_integration_rigorous(temp_adata, 'X_emb', label_key='cluster', batch_key='batch')
                
                # Add Simulation Meta
                metrics['Method'] = method
                metrics['N_Clusters'] = n_clust
                metrics['Fold_Change'] = fc
                
                all_metrics = pd.concat([all_metrics, pd.DataFrame([metrics])], ignore_index=True)
                
    # 4. Save and Plot
    all_metrics.to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)
    
    # Plot ARI vs Fold Change (Line Plot)
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=all_metrics, x="Fold_Change", y="ARI", hue="Method", style="N_Clusters", markers=True)
    plt.title("Clustering Accuracy (ARI) vs Signal Strength")
    plt.savefig(os.path.join(output_dir, "Sim_ARI_vs_FC.pdf"), bbox_inches='tight')
    plt.close()
    
    # Plot ASW vs Fold Change
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=all_metrics, x="Fold_Change", y="Silhouette_Bio", hue="Method", style="N_Clusters", markers=True)
    plt.title("Cluster Separation (ASW) vs Signal Strength")
    plt.savefig(os.path.join(output_dir, "Sim_ASW_vs_FC.pdf"), bbox_inches='tight')
    plt.close()
    
    print("Simulation experiment complete.")

if __name__ == "__main__":
    run_simulation_experiment("./results_simulation")
