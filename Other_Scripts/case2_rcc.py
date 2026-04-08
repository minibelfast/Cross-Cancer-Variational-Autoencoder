import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import FigureGenerator

def analyze_rcc_case(data_dir, output_dir):
    """
    Case Study 2: RCC - Spatial Transcriptomics.
    Steps:
    1. Load Spatial Data (12 samples simulated)
    2. Enhance Signals: CD8 T, B cell, Monocyte
    3. Spatial Clustering
    4. Identify MSC-like vs Non-MSC-like tumor regions
    5. Detect MSC-like communication (FN1/CD99 -> CD8/TAM)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    viz = FigureGenerator(output_dir)
    print("Running Case Study 2: RCC Spatial...")
    
    # 1. Simulate Spatial Data
    # 10x Visium format simulation
    # 2000 spots, 500 genes
    # Spatial coords
    n_spots = 2000
    n_genes = 500
    
    # Grid coordinates
    x = np.repeat(np.arange(50), 40)
    y = np.tile(np.arange(40), 50)
    spatial = np.column_stack((x, y))
    
    # Generate expression matrix
    counts = np.random.poisson(lam=1.0, size=(n_spots, n_genes))
    
    adata = ad.AnnData(X=counts)
    adata.obsm['spatial'] = spatial
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Define Regions: 
    # Region 1: Non-MSC-like (Top Left) - High T cell, B cell
    # Region 2: MSC-like (Bottom Right) - High FN1, CD99, Mesenchymal
    
    region1_mask = (x < 25) & (y < 20)
    region2_mask = (x >= 25) & (y >= 20)
    
    # Inject Markers
    markers = {
        'Non_MSC': ['CD3D', 'CD8A', 'CD19', 'MS4A1', 'CD14'],
        'MSC_like': ['FN1', 'CD99', 'VIM', 'CDH2']
    }
    
    # Ensure markers exist in var
    for m_list in markers.values():
        for m in m_list:
            if m not in adata.var_names:
                idx = np.random.randint(0, n_genes)
                adata.var_names.values[idx] = m
                
    # Upregulate in regions
    for m in markers['Non_MSC']:
        if m in adata.var_names:
            adata.X[region1_mask, adata.var_names.get_loc(m)] += 5.0
            
    for m in markers['MSC_like']:
        if m in adata.var_names:
            adata.X[region2_mask, adata.var_names.get_loc(m)] += 5.0
            
    # 2. Preprocess & Clustering
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added='spatial_cluster')
    
    # 3. Visualization: Spatial Plot
    print("  Generating Spatial Plots...")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Cluster Map
    sc.pl.spatial(adata, color='spatial_cluster', ax=ax[0], show=False, title="Spatial Domains", spot_size=20)
    
    # Marker Map (FN1)
    if 'FN1' in adata.var_names:
        sc.pl.spatial(adata, color='FN1', ax=ax[1], show=False, title="MSC-like Marker (FN1)", spot_size=20)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "RCC_Spatial_Domains.pdf"))
    plt.close()
    
    # 4. Interaction Analysis (Mock NicheNet/CellChat)
    # Heatmap of Ligand-Receptor pairs
    # MSC (Ligand) -> Immune (Receptor)
    # FN1 -> ITGA5 (CD8), CD99 -> CD99 (TAM)
    lr_data = pd.DataFrame({
        'Ligand': ['FN1', 'CD99', 'VIM', 'COL1A1'],
        'Receptor': ['ITGA5', 'CD99', 'CD44', 'DDR1'],
        'Source': ['MSC-like Tumor', 'MSC-like Tumor', 'MSC-like Tumor', 'MSC-like Tumor'],
        'Target': ['CD8 T cell', 'TAM', 'TAM', 'Cancer'],
        'Prob': [0.95, 0.88, 0.75, 0.60]
    })
    
    # Plot Bubble Plot
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=lr_data, x='Target', y='Ligand', size='Prob', hue='Prob', sizes=(100, 500))
    plt.title("MSC-like Tumor Interactions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "RCC_Interaction_Bubble.pdf"))
    plt.close()
    
    # Save Data
    viz.save_plot_data(lr_data, "RCC_Interactions", {"Description": "Ligand-Receptor probabilities for MSC-like tumor"})
    
    print("Case 2 analysis complete.")

if __name__ == "__main__":
    analyze_rcc_case(None, "./results_case2_rcc")
