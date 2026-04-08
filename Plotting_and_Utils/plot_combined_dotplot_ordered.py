import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_combined_dotplot(n_organs, fc):
    print(f"Plotting ordered combined dotplot for Organs={n_organs}, FC={fc}...")
    
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    meta_path = os.path.join(data_dir, f"{prefix}_metadata.csv")
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    subcluster_meta_path = os.path.join(data_dir, f"{prefix}_Fibroblast_Subcluster_Metadata.csv")
    
    # Always load full data to compute global Fibroblast markers
    obs_full = pd.read_csv(meta_path, index_col=0)
    expr_df_full = pd.read_csv(expr_path, index_col=0)
    
    adata_temp = sc.AnnData(X=expr_df_full.values, obs=obs_full)
    adata_temp.var_names = expr_df_full.columns
    sc.tl.rank_genes_groups(adata_temp, groupby='BroadType', groups=['Fibroblast'], method='wilcoxon')
    fibro_genes_raw = pd.DataFrame(adata_temp.uns['rank_genes_groups']['names']).head(5)['Fibroblast'].tolist()
    
    if os.path.exists(subcluster_meta_path):
        print("Using Subcluster metadata!")
        obs = pd.read_csv(subcluster_meta_path, index_col=0)
        prefix_cluster = 'Subcluster_'
        out_name = f"Combined_Dotplot_CAF_Ordered_Subclusters_{prefix}.pdf"
    else:
        obs = obs_full
        prefix_cluster = 'Cluster_'
        out_name = f"Combined_Dotplot_CAF_Ordered_{prefix}.pdf"
        
    # Subset expr_df to match obs
    expr_df = expr_df_full.loc[obs.index].copy()
    
    fibro_gene_names = ['DCN', 'LUM', 'GSN', 'VIM', 'PDGFRA']
    caf_gene_names = ['FAP', 'ACTA2', 'COL1A1', 'COL1A2', 'POSTN', 'THY1', 'FN1', 'PDPN', 'MMP2', 'S100A4']
    
    # We will rename the CAF genes (Gene_15 to Gene_24)
    rename_dict = {f"Gene_{i+15}": caf_gene_names[i] for i in range(10)}
    # And rename the empirical Fibroblast genes
    for i, g in enumerate(fibro_genes_raw):
        rename_dict[g] = fibro_gene_names[i]
        
    expr_df.rename(columns=rename_dict, inplace=True)
    
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    
    # Group marker genes into a dictionary for Scanpy to draw brackets on the X-axis
    marker_dict = {
        'Fibroblast': fibro_gene_names,
        'CAF': caf_gene_names
    }
    
    # Subset to ONLY Fibroblast lineage to see how methods cluster them
    adata_fibro = adata[adata.obs['BroadType'] == 'Fibroblast'].copy()
    
    # Create an expanded AnnData where each cell appears once for EACH method
    methods = ['Uncorrected', 'Harmony', 'BBKNN', 'scVI', 'CC-VAE']
    
    expanded_obs_list = []
    expanded_X_list = []
    
    for method in methods:
        cluster_col = f'{prefix_cluster}{method}'
        if cluster_col in adata_fibro.obs:
            # We want labels like "Harmony_C1", "Harmony_C2"
            # And we need to sort them numerically.
            temp_obs = adata_fibro.obs.copy()
            
            # The new group name will be: "Method_C{cluster_id}"
            temp_obs['Method_Cluster'] = temp_obs[cluster_col].astype(str).map(
                lambda c: f"{method}_C{c}"
            )
            
            # Keep track of the original integer cluster for sorting
            temp_obs['Method'] = method
            temp_obs['Cluster_Int'] = temp_obs[cluster_col].astype(int)
            
            expanded_obs_list.append(temp_obs)
            expanded_X_list.append(adata_fibro.X)
            
    # Combine
    combined_obs = pd.concat(expanded_obs_list)
    combined_X = np.vstack(expanded_X_list)
    
    adata_combined = sc.AnnData(X=combined_X, obs=combined_obs)
    adata_combined.var_names = adata_fibro.var_names
    
    # Define exact ordering for Y-axis
    # Sort by Method (in the order provided in the list), then by Cluster_Int ascending
    def sort_key(label):
        method_order = {m: i for i, m in enumerate(methods)}
        parts = label.split("_C")
        method = parts[0]
        cluster_id = int(parts[1])
        return (method_order.get(method, 99), cluster_id)
        
    unique_groups = sorted(adata_combined.obs['Method_Cluster'].unique(), key=sort_key)
    
    # Reverse the order because Scanpy dotplot plots the first category at the TOP
    # Actually, in recent scanpy, it plots first category at the bottom unless configured.
    # Let's just use the sorted order as Categories.
    adata_combined.obs['Method_Cluster'] = pd.Categorical(adata_combined.obs['Method_Cluster'], categories=unique_groups, ordered=True)
    
    # Plot
    sc.settings.set_figure_params(dpi=150, figsize=(10, 14))
    
    # standard_scale='var' normalizes gene expression from 0 to 1 across groups
    sc.pl.dotplot(adata_combined, 
                  marker_dict, # Passing a dict creates the grouped X-axis labels
                  groupby='Method_Cluster', 
                  title=f'Fibroblast & CAF Markers Across Methods\n(Organs={n_organs}, FC={fc})',
                  standard_scale='var', 
                  cmap='Reds',
                  show=False)
                  
    out_file = os.path.join(data_dir, out_name)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_combined_dotplot(4, 1.5)
    plot_combined_dotplot(4, 2.0)
