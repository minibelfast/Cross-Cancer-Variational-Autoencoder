import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_organ_dotplot(n_organs, fc):
    print(f"Plotting ordered organ dotplot for Organs={n_organs}, FC={fc}...")
    
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    meta_path = os.path.join(data_dir, f"{prefix}_metadata.csv")
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    subcluster_meta_path = os.path.join(data_dir, f"{prefix}_Fibroblast_Subcluster_Metadata.csv")
    
    # Always load full data to compute global Organ markers
    obs_full = pd.read_csv(meta_path, index_col=0)
    expr_df_full = pd.read_csv(expr_path, index_col=0)
    
    if os.path.exists(subcluster_meta_path):
        print("Using Fibroblast/CAF Subcluster metadata!")
        obs = pd.read_csv(subcluster_meta_path, index_col=0)
        prefix_cluster = 'Subcluster_'
        out_name = f"Combined_Dotplot_Organ_Ordered_Subclusters_{prefix}.pdf"
    else:
        obs = obs_full
        prefix_cluster = 'Cluster_'
        out_name = f"Combined_Dotplot_Organ_Ordered_{prefix}.pdf"
        
    expr_df = expr_df_full.loc[obs.index].copy()
    
    # We will rename the organ-specific macrophage markers (from simulation.py)
    # Liver: 0-4
    liver_genes = ['CD163', 'MARCO', 'VSIG4', 'CLEC4F', 'VVSIG4']
    # Lung: 5-9
    lung_genes = ['PPARG', 'FABP4', 'MARCO_L', 'MSR1', 'CHIT1']
    # Kidney: 10-14
    kidney_genes = ['CX3CR1', 'CCR2', 'CD14', 'FCGR1A', 'ITGAM']
    
    # What about Bladder? In simulation.py we only added explicit markers for Liver, Lung, Kidney.
    # Bladder wasn't given explicit markers, but let's find empirical ones or just plot the top 3 organs.
    # Actually, we can use the empirical rank_genes_groups to find Bladder markers if it's there.
    # We should run this on the FULL dataset to find real organ markers
    adata_temp = sc.AnnData(X=expr_df_full.values, obs=obs_full)
    adata_temp.var_names = expr_df_full.columns
    
    sc.tl.rank_genes_groups(adata_temp, groupby='Organ', method='wilcoxon')
    
    # Get top 5 empirical markers for Bladder
    if 'Bladder' in adata_temp.uns['rank_genes_groups']['names'].dtype.names:
        bladder_genes_raw = pd.DataFrame(adata_temp.uns['rank_genes_groups']['names']).head(5)['Bladder'].tolist()
        bladder_genes = ['UPK3A', 'UPK1A', 'UPK2', 'KRT20', 'FGFR3']
    else:
        bladder_genes_raw = []
        bladder_genes = []
        
    rename_dict = {}
    for i, g in enumerate(liver_genes):
        rename_dict[f"Gene_{i}"] = g
    for i, g in enumerate(lung_genes):
        rename_dict[f"Gene_{i+5}"] = g
    for i, g in enumerate(kidney_genes):
        rename_dict[f"Gene_{i+10}"] = g
        
    for i, g in enumerate(bladder_genes_raw):
        rename_dict[g] = bladder_genes[i]
        
    expr_df.rename(columns=rename_dict, inplace=True)
    
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    
    marker_dict = {
        'Liver Markers': liver_genes,
        'Lung Markers': lung_genes,
        'Kidney Markers': kidney_genes,
    }
    if bladder_genes:
        # User asked for "Bladder高表达基因" first
        marker_dict = {
            'Bladder Markers': bladder_genes,
            'Kidney Markers': kidney_genes,
            'Liver Markers': liver_genes,
            'Lung Markers': lung_genes
        }
        
    # Create an expanded AnnData where each cell appears once for EACH method
    methods = ['Uncorrected', 'Harmony', 'BBKNN', 'scVI', 'CC-VAE']
    
    expanded_obs_list = []
    expanded_X_list = []
    
    for method in methods:
        cluster_col = f'{prefix_cluster}{method}'
        if cluster_col in adata.obs:
            temp_obs = adata.obs.copy()
            
            temp_obs['Method_Cluster'] = temp_obs[cluster_col].astype(str).map(
                lambda c: f"{method}_C{c}"
            )
            
            temp_obs['Method'] = method
            temp_obs['Cluster_Int'] = temp_obs[cluster_col].astype(int)
            
            expanded_obs_list.append(temp_obs)
            expanded_X_list.append(adata.X)
            
    combined_obs = pd.concat(expanded_obs_list)
    combined_X = np.vstack(expanded_X_list)
    
    adata_combined = sc.AnnData(X=combined_X, obs=combined_obs)
    adata_combined.var_names = adata.var_names
    
    def sort_key(label):
        method_order = {m: i for i, m in enumerate(methods)}
        parts = label.split("_C")
        method = parts[0]
        cluster_id = int(parts[1])
        return (method_order.get(method, 99), cluster_id)
        
    unique_groups = sorted(adata_combined.obs['Method_Cluster'].unique(), key=sort_key)
    
    adata_combined.obs['Method_Cluster'] = pd.Categorical(adata_combined.obs['Method_Cluster'], categories=unique_groups, ordered=True)
    
    sc.settings.set_figure_params(dpi=150, figsize=(12, 14))
    
    sc.pl.dotplot(adata_combined, 
                  marker_dict,
                  groupby='Method_Cluster', 
                  title=f'Organ Specific Markers Across Methods\n(Organs={n_organs}, FC={fc})',
                  standard_scale='var', 
                  cmap='Blues',
                  show=False)
                  
    out_file = os.path.join(data_dir, out_name)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_organ_dotplot(4, 1.5)
    plot_organ_dotplot(4, 2.0)
