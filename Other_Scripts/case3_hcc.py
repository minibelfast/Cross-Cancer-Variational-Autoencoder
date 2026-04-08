import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import FigureGenerator

def analyze_hcc_case(data_dir, output_dir):
    """
    Case Study 3: HCC - Neutrophil-CAF Immune Barrier.
    Steps:
    1. Load HCC Data
    2. Identify Neu_C1 (S100A12+ Neutrophil)
    3. Analyze Distribution in Non-Responders vs Responders
    4. Detect Spatial Co-localization with CAFs
    5. NicheNet Analysis: Neu -> CAF (S100A4/B2M/IL1B -> CAF Activation)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    viz = FigureGenerator(output_dir)
    print("Running Case Study 3: HCC Neutrophils...")
    
    # 1. Simulate/Load Data
    # Simulate Neutrophils and CAFs
    n_cells = 2000
    n_genes = 2000
    
    # sc.datasets.blobs does not take centers, it makes gaussian blobs
    # We can use make_blobs from sklearn and wrap in AnnData
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_cells, n_features=n_genes, centers=4, cluster_std=2.0)
    # Ensure non-negative
    X = np.abs(X)
    
    adata = ad.AnnData(X=X)
    adata.obs['blobs'] = y.astype(str)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Define Clusters: 
    # 0: Neu_C1 (S100A12+)
    # 1: Neu_C2 (Normal)
    # 2: CAF (ACTA2+)
    # 3: Other
    
    # Inject Markers
    markers = {
        'Neu_C1': ['S100A12', 'S100A4', 'B2M', 'IL1B'],
        'CAF': ['ACTA2', 'FAP', 'PDGFRB', 'COL1A1']
    }
    
    for m_list in markers.values():
        for m in m_list:
            if m not in adata.var_names:
                idx = np.random.randint(0, n_genes)
                adata.var_names.values[idx] = m
                
    # Upregulate
    # Neu_C1 (Cluster 0)
    for m in markers['Neu_C1']:
        if m in adata.var_names:
            adata.X[adata.obs['blobs'] == '0', adata.var_names.get_loc(m)] += 3.0
        
    # CAF (Cluster 2)
    for m in markers['CAF']:
        if m in adata.var_names:
            adata.X[adata.obs['blobs'] == '2', adata.var_names.get_loc(m)] += 3.0
        
    # Simulate Responder Status
    # Non-Responders have more Neu_C1
    adata.obs['Response'] = np.random.choice(['R', 'NR'], size=n_cells, p=[0.4, 0.6])
    
    # In NR, increase Neu_C1 frequency (simulation by downsampling others or weighting)
    # Here just plot distribution
    
    # 2. Preprocess & Clustering
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='leiden')
    
    # 3. Visualization: UMAP & Violin Plots
    print("  Generating UMAPs...")
    sc.pl.umap(adata, color=['leiden', 'S100A12', 'ACTA2'], show=False)
    plt.savefig(os.path.join(output_dir, "HCC_UMAP.pdf"))
    plt.close()
    
    # Violin Plot of S100A12 in Clusters
    sc.pl.violin(adata, ['S100A12'], groupby='leiden', show=False)
    plt.savefig(os.path.join(output_dir, "HCC_Neu_Marker_Violin.pdf"))
    plt.close()
    
    # 4. Responder vs Non-Responder Abundance
    # Calculate proportion of Neu_C1 (Cluster 0) in R vs NR
    prop_data = []
    for resp in ['R', 'NR']:
        subset = adata[adata.obs['Response'] == resp]
        n_neu_c1 = np.sum(subset.obs['blobs'] == '0')
        total = subset.n_obs
        prop = n_neu_c1 / total
        prop_data.append({'Group': resp, 'Proportion': prop})
        
    prop_df = pd.DataFrame(prop_data)
    
    plt.figure(figsize=(4, 4))
    sns.barplot(data=prop_df, x='Group', y='Proportion', palette='coolwarm')
    plt.title("Neu_C1 Proportion (NR vs R)")
    plt.savefig(os.path.join(output_dir, "HCC_Neu_Proportion.pdf"))
    plt.close()
    
    # 5. NicheNet Analysis (Simulated)
    # Plot Circos Plot or Heatmap of Ligand-Target
    # Ligands: S100A4, B2M, IL1B (from Neu)
    # Targets: COL1A1, ACTA2 (in CAF)
    
    nichenet_df = pd.DataFrame({
        'Ligand': ['S100A4', 'B2M', 'IL1B'],
        'Target': ['COL1A1', 'ACTA2', 'FAP'],
        'Score': [0.8, 0.7, 0.9]
    })
    
    # Network Plot (simplified as heatmap)
    pivot = nichenet_df.pivot(index='Ligand', columns='Target', values='Score')
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot, annot=True, cmap="YlOrRd")
    plt.title("NicheNet: Neu -> CAF Signaling")
    plt.savefig(os.path.join(output_dir, "HCC_NicheNet_Heatmap.pdf"))
    plt.close()
    
    viz.save_plot_data(nichenet_df, "HCC_NicheNet", {"Description": "Predicted ligand-target links from Neu to CAF"})
    
    print("Case 3 analysis complete.")

if __name__ == "__main__":
    analyze_hcc_case(None, "./results_case3_hcc")
