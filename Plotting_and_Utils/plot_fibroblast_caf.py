import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dotplot(n_organs, fc):
    print(f"Plotting for Organs={n_organs}, FC={fc}...")
    
    # Load expression and metadata
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    # Load metadata
    meta_path = os.path.join(data_dir, f"{prefix}_metadata.csv")
    if not os.path.exists(meta_path):
        print(f"File not found: {meta_path}")
        return
        
    obs = pd.read_csv(meta_path, index_col=0)
    
    # Load expression
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    expr_df = pd.read_csv(expr_path, index_col=0)
    
    # Create AnnData
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    
    # Filter for Fibroblast and CAF (to make the plot focused)
    # We will plot the markers for all cells, but group them by cluster.
    # The markers for TAM/Tumor are Gene_15 to Gene_24.
    # However, for Fibroblast/CAF, we didn't inject specific explicit genes for CAF vs Fibroblast in the simulation script!
    # Let's check simulation.py:
    #   Tumor markers (tam_genes) are indices 15-24, applied to ALL State=="Tumor" (including CAF)
    #   Broad markers (macro, T) are 25-34, 35-44. No specific Fibroblast broad markers were injected!
    # Wait, let me just plot the Tumor markers (Gene_15 to Gene_24) for Fibroblasts, because CAFs are Tumor state.
    
    # Actually, we can just select the cells that are Fibroblasts in ground truth
    # to see how the clusters capture them.
    adata_fibro = adata[adata.obs['BroadType'] == 'Fibroblast'].copy()
    
    # Genes to plot: Tumor markers (Gene_15 to Gene_24)
    # These represent the "CAF" markers in our simulation (since CAFs are Tumor state)
    marker_genes = [f"Gene_{i}" for i in range(15, 25)]
    
    methods = ['CC-VAE', 'BBKNN', 'Harmony', 'scVI', 'Uncorrected']
    
    # We will create a dotplot for each method's clustering
    for method in methods:
        cluster_col = f'Cluster_{method}'
        if cluster_col in adata_fibro.obs:
            # We want to show how different clusters express these CAF markers
            # Filter to clusters that actually contain Fibroblasts
            clusters_with_fibro = adata_fibro.obs[cluster_col].unique().tolist()
            
            # Use the full adata but subset to the clusters that have fibroblasts, or just use adata_fibro
            # Plotting on adata_fibro shows expression within the Fibroblast ground-truth cells, split by the method's clusters
            
            sc.pl.dotplot(adata_fibro, marker_genes, groupby=cluster_col, 
                          title=f'{method} Clusters (Fibroblast Cells) - Organs{n_organs} FC{fc}',
                          show=False)
                          
            out_file = os.path.join(data_dir, f"Dotplot_CAF_Markers_{method}_{prefix}.pdf")
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_dotplot(4, 1.5)
    plot_dotplot(4, 2.0)
