import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dotplot(n_organs, fc):
    print(f"Plotting all cells dotplot for Organs={n_organs}, FC={fc}...")
    
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    meta_path = os.path.join(data_dir, f"{prefix}_metadata.csv")
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    
    obs = pd.read_csv(meta_path, index_col=0)
    expr_df = pd.read_csv(expr_path, index_col=0)
    
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    
    # Tumor/CAF markers
    marker_genes = [f"Gene_{i}" for i in range(15, 25)]
    
    methods = ['CC-VAE', 'BBKNN', 'Harmony', 'scVI', 'Uncorrected']
    
    for method in methods:
        cluster_col = f'Cluster_{method}'
        if cluster_col in adata.obs:
            # Create a combined label to show the dominant cell type in each cluster
            # This makes the Y-axis easier to interpret
            cluster_majority_type = adata.obs.groupby(cluster_col)['Detailed_Subtype'].agg(lambda x: x.value_counts().index[0])
            
            # Add the majority type to the cluster name for the plot
            adata.obs[f'{cluster_col}_annotated'] = adata.obs[cluster_col].astype(str).map(
                lambda c: f"C{c} ({cluster_majority_type[int(c)]})"
            )
            
            # Make the plot larger and save
            sc.pl.dotplot(adata, marker_genes, groupby=f'{cluster_col}_annotated', 
                          title=f'{method} Clusters - Organs{n_organs} FC{fc} (Tumor/CAF Markers)',
                          show=False,
                          dendrogram=True) # Adding dendrogram to group similar clusters
                          
            out_file = os.path.join(data_dir, f"Dotplot_AllCells_CAF_Markers_{method}_{prefix}.pdf")
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_dotplot(4, 1.5)
    plot_dotplot(4, 2.0)
