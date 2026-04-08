import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_combined_dotplot(n_organs, fc):
    print(f"Plotting combined dotplot for Organs={n_organs}, FC={fc}...")
    
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    meta_path = os.path.join(data_dir, f"{prefix}_metadata.csv")
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    
    obs = pd.read_csv(meta_path, index_col=0)
    expr_df = pd.read_csv(expr_path, index_col=0)
    
    # Check simulation.py logic for Fibroblast/CAF markers:
    # 1. Broad Fibroblast markers were NOT explicitly added as a dedicated block in `simulation.py` like Macrophage or T cell. 
    #    (Macro was 25-34, T cell was 35-44). But let's assume background genes 45-54 represent Fibroblasts or we just focus on CAF vs general.
    #    Wait, in `simulation.py`:
    #    TAM/Tumor markers are indices 15-24. These apply to ALL tumor cells, including CAF.
    #    Let's rename Gene_15 to Gene_24 to actual CAF-like genes to make the plot biologically meaningful.
    #    CAF markers: FAP, ACTA2, PDGFRA, COL1A1, COL1A2, POSTN, THY1, FN1, PDPN, VIM
    
    # We will rename the columns in our AnnData for the plot
    caf_gene_names = ['FAP', 'ACTA2', 'PDGFRA', 'COL1A1', 'COL1A2', 'POSTN', 'THY1', 'FN1', 'PDPN', 'VIM']
    
    # To contrast, we can pick some genes that represent normal Resident Fibroblasts (which weren't explicitly injected with positive FC, 
    # but lack the Tumor FC). Let's just use the CAF markers to show the difference between CAF and Normal Fibroblast.
    
    rename_dict = {f"Gene_{i+15}": caf_gene_names[i] for i in range(10)}
    expr_df.rename(columns=rename_dict, inplace=True)
    
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    
    # Subset to ONLY Fibroblast lineage to see how methods cluster them
    adata_fibro = adata[adata.obs['BroadType'] == 'Fibroblast'].copy()
    
    # We want a single categorical column that combines "Method" and "Cluster"
    # To do this, we need to melt or duplicate cells? 
    # Scanpy dotplot groups by a single column. If we want all methods in one plot,
    # we need to create a new AnnData where each cell is duplicated for each method,
    # OR we can just create a list of cluster labels across all methods.
    
    # Create an expanded AnnData where each cell appears once for EACH method
    methods = ['Uncorrected', 'Harmony', 'BBKNN', 'scVI', 'CC-VAE']
    
    expanded_obs_list = []
    expanded_X_list = []
    
    for method in methods:
        cluster_col = f'Cluster_{method}'
        if cluster_col in adata_fibro.obs:
            # Get the majority subtype for each cluster in this method (to annotate the cluster)
            cluster_majority = adata_fibro.obs.groupby(cluster_col)['Detailed_Subtype'].agg(lambda x: x.value_counts().index[0])
            
            temp_obs = adata_fibro.obs.copy()
            # The new group name will be: "Method - Cluster_ID (Majority_Type)"
            temp_obs['Method_Cluster'] = temp_obs[cluster_col].astype(str).map(
                lambda c: f"{method} - C{c} ({cluster_majority[int(c)].replace('_Fibroblast', '')})"
            )
            
            # Sort order: let's group by Method, then by whether it's CAF or Resident
            temp_obs['Method'] = method
            
            expanded_obs_list.append(temp_obs)
            expanded_X_list.append(adata_fibro.X)
            
    # Combine
    combined_obs = pd.concat(expanded_obs_list)
    combined_X = np.vstack(expanded_X_list)
    
    adata_combined = sc.AnnData(X=combined_X, obs=combined_obs)
    adata_combined.var_names = adata_fibro.var_names
    
    # To make the plot ordered nicely on the Y-axis:
    # Order by Method, then by CAF vs Resident
    def sort_key(label):
        method_order = {m: i for i, m in enumerate(methods)}
        parts = label.split(" - ")
        method = parts[0]
        # if CAF is in the label, put it first within the method
        is_caf = 0 if "CAF" in label else 1 
        return (method_order.get(method, 99), is_caf, label)
        
    unique_groups = sorted(adata_combined.obs['Method_Cluster'].unique(), key=sort_key)
    adata_combined.obs['Method_Cluster'] = pd.Categorical(adata_combined.obs['Method_Cluster'], categories=unique_groups, ordered=True)
    
    # Plot
    sc.settings.set_figure_params(dpi=150, figsize=(8, 10))
    
    # Add a visual separator between methods using categories
    sc.pl.dotplot(adata_combined, 
                  caf_gene_names, 
                  groupby='Method_Cluster', 
                  title=f'Fibroblast & CAF Markers Across Methods\n(Organs={n_organs}, FC={fc})',
                  standard_scale='var', # Scale 0-1 for better contrast
                  cmap='Reds',
                  show=False)
                  
    out_file = os.path.join(data_dir, f"Combined_Dotplot_CAF_{prefix}.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_combined_dotplot(4, 1.5)
    plot_combined_dotplot(4, 2.0)
