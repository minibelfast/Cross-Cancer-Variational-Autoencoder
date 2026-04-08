import argparse
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, preprocess_data
from model import CrossCancerVAE
from integration_benchmarks import run_benchmarks
from evaluation_advanced import evaluate_integration_rigorous, save_fair_data
from visualization import FigureGenerator

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Initialize FigureGenerator
    viz = FigureGenerator(args.output_dir)

    # 1. Load and Preprocess Data
    print("Step 1: Loading Data...")
    adata = load_data(args.data_dir, use_synthetic=args.use_synthetic, subsample_n=args.subsample_n)
    adata = preprocess_data(adata)
    
    # Store raw counts for DE analysis later if needed
    adata.raw = adata
    
    # 2. Run CC-VAE (Our Method)
    print("Step 2: Running CC-VAE...")
    cc_vae = CrossCancerVAE(adata, batch_key='batch', n_latent=args.n_latent)
    cc_vae.train(max_epochs=args.max_epochs)
    
    # Get latent representation
    latent_cc_vae = cc_vae.get_latent_representation()
    
    # 3. Run Benchmarks
    print("Step 3: Running Benchmarks...")
    benchmark_methods = ['Harmony', 'scVI', 'BBKNN'] # Add others if implemented
    embeddings = run_benchmarks(adata, methods=benchmark_methods, batch_key='batch')
    
    # Add CC-VAE to embeddings
    embeddings['CC-VAE'] = latent_cc_vae
    
    # 4. Evaluation (Nature Methods Standards)
    print("Step 4: Evaluating Performance (Nature Methods Standards)...")
    
    label_key = 'labels' if 'labels' in adata.obs else 'cell_type'
    if label_key not in adata.obs:
        print(f"Warning: Label key '{label_key}' not found. Using 'batch' as proxy (Metrics will be limited).")
        label_key = 'batch'
        
    metrics_df = pd.DataFrame()
    adata_dict = {}
    
    for method, emb in embeddings.items():
        print(f"Processing {method}...")
        temp_adata = adata.copy()
        
        # specific handling for BBKNN which returns UMAP directly
        if method == 'BBKNN':
            temp_adata.obsm['X_umap'] = emb
            temp_adata.obsm['X_emb'] = emb 
        else:
            temp_adata.obsm['X_emb'] = emb
            sc.pp.neighbors(temp_adata, use_rep='X_emb')
            sc.tl.umap(temp_adata)
            
        # Rigorous Metrics
        metrics = evaluate_integration_rigorous(temp_adata, 'X_emb', label_key=label_key, batch_key='batch')
        metrics['Method'] = method
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        
        adata_dict[method] = temp_adata
        
        # FAIR Data Output for each method
        save_fair_data(temp_adata, os.path.join(args.output_dir, "data"), prefix=method)
        
    # Save metrics
    metrics_df.to_csv(os.path.join(args.output_dir, "evaluation_metrics_nature.csv"), index=False)
    print(metrics_df)
    
    # 5. Visualization (Nature Methods Style)
    print("Step 5: Generating Visualizations...")
    
    # Figure 2: Global Summary
    viz.plot_figure_2_metrics_summary(metrics_df)
    
    # Figure 3: Detailed Metrics
    viz.plot_figure_3_detailed_metrics(metrics_df)
    
    # Figure 4: UMAPs
    viz.plot_figure_4_umaps(adata_dict, color_keys=['batch', label_key])
    
    # Figure 5: Confusion Matrix for CC-VAE
    sc.tl.leiden(adata_dict['CC-VAE'], key_added='leiden_pred')
    viz.plot_figure_5_confusion_matrix(adata_dict['CC-VAE'], label_key, 'leiden_pred', 'CC-VAE')
    
    # Marker heatmap for CC-VAE (using the integrated clustering)
    # plot_marker_heatmap(adata_dict['CC-VAE'], 'CC-VAE', args.output_dir, groupby='leiden_pred')

    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Cancer scRNA-seq Integration Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for results")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data if real data missing")
    parser.add_argument("--subsample_n", type=int, default=None, help="Number of cells to subsample for testing")
    parser.add_argument("--n_latent", type=int, default=30, help="Latent dimension for CC-VAE")
    parser.add_argument("--max_epochs", type=int, default=100, help="Training epochs")
    
    args = parser.parse_args()
    main(args)
