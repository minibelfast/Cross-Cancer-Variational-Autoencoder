import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sctour as sct

def compute_and_plot_pseudotime(n_organs, fc):
    print(f"\n--- Processing Organs={n_organs}, FC={fc} ---")
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    meta_path = os.path.join(data_dir, f"{prefix}_Fibroblast_Subcluster_Metadata.csv")
    expr_path = os.path.join(data_dir, f"{prefix}_expression.csv")
    
    if not os.path.exists(meta_path) or not os.path.exists(expr_path):
        print(f"Data for {prefix} not found!")
        return
        
    # 1. Load data
    obs = pd.read_csv(meta_path, index_col=0)
    expr_df = pd.read_csv(expr_path, index_col=0)
    
    # Subset expression matrix to only Fibroblast/CAF cells
    expr_df = expr_df.loc[obs.index]
    
    adata = sc.AnnData(X=expr_df.values, obs=obs)
    adata.var_names = expr_df.columns
    
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes.")
    
    # scTour requires n_genes_by_counts in obs
    # Since our data is already normalized expression, we can estimate it or just calculate qc metrics
    # scTour uses this for library size adjustment internally
    adata.X = np.expm1(adata.X) # convert log1p back to counts roughly
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.X = np.log1p(adata.X) # back to log1p
    
    # 2. Define Features for Pseudotime
    # We will use Fibroblast and CAF specific marker genes to guide the trajectory
    fibro_genes = ['DCN', 'LUM', 'GSN', 'VIM', 'PDGFRA']
    caf_genes = ['FAP', 'ACTA2', 'COL1A1', 'COL1A2', 'POSTN', 'THY1', 'FN1', 'PDPN', 'MMP2', 'S100A4']
    
    # We need highly variable genes for scTour to work efficiently.
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='cell_ranger')
    
    # Combine HVGs with our known marker genes
    hvg_mask = adata.var.highly_variable.copy()
    for g in fibro_genes + caf_genes:
        if g in adata.var_names:
            hvg_mask[g] = True
            
    adata_traj = adata[:, hvg_mask].copy()
    
    # 3. Run scTour to infer pseudotime
    print("Training scTour model to infer pseudotime...")
    # Set random seed for reproducibility
    np.random.seed(42)
    tnn = sct.train.Trainer(adata_traj, loss_mode='mse')
    # scTour automatically determines epochs if not passed, but let's just run default
    tnn.train()
    
    # Get pseudotime
    pseudotime = tnn.get_time()
    adata.obs['scTour_Pseudotime'] = pseudotime
    
    # Normalize pseudotime to [0, 1] for easier interpretation
    pt_min = adata.obs['scTour_Pseudotime'].min()
    pt_max = adata.obs['scTour_Pseudotime'].max()
    adata.obs['scTour_Pseudotime_Norm'] = (adata.obs['scTour_Pseudotime'] - pt_min) / (pt_max - pt_min)
    
    # We want to see if the direction makes sense (Resident -> CAF)
    # Let's check the mean pseudotime for State (Tumor vs Resident/Normal)
    mean_pt = adata.obs.groupby('State')['scTour_Pseudotime_Norm'].mean()
    print("Mean Pseudotime by State:")
    print(mean_pt)
    
    # If Tumor has a lower pseudotime than Normal/Resident, we might want to invert it 
    # so that the trajectory goes from Normal -> Tumor
    if 'Tumor' in mean_pt and 'Resident' in mean_pt:
        if mean_pt['Tumor'] < mean_pt['Resident']:
            print("Inverting pseudotime so Tumor is late in trajectory.")
            adata.obs['scTour_Pseudotime_Norm'] = 1.0 - adata.obs['scTour_Pseudotime_Norm']
    elif 'Tumor' in mean_pt and 'Normal' in mean_pt:
        if mean_pt['Tumor'] < mean_pt['Normal']:
            print("Inverting pseudotime so Tumor is late in trajectory.")
            adata.obs['scTour_Pseudotime_Norm'] = 1.0 - adata.obs['scTour_Pseudotime_Norm']
            
    # 4. Plot Violin plots for CC-VAE and Harmony
    methods_to_plot = ['Harmony', 'CC-VAE']
    
    # Set style
    sns.set(style="whitegrid")
    
    # Define a rich color palette for the 8 clusters
    cluster_palette = sns.color_palette("husl", 8)
    
    for method in methods_to_plot:
        cluster_col = f'Subcluster_{method}'
        if cluster_col not in adata.obs.columns:
            print(f"Column {cluster_col} not found!")
            continue
            
        # Create a new column with prefixed cluster names (e.g. CC-VAE_C0)
        plot_col = f'{method}_Cluster'
        adata.obs[plot_col] = adata.obs[cluster_col].apply(lambda x: f"{method}_C{x}")
        
        # Sort clusters C0 to C7
        unique_clusters = sorted(adata.obs[plot_col].unique(), key=lambda x: int(x.split('_C')[1]))
        
        plt.figure(figsize=(10, 6))
        
        # Map each cluster to a unique color
        palette = {c: cluster_palette[i] for i, c in enumerate(unique_clusters)}
        
        # Plot Violin (colored by cluster) without inner elements
        ax = sns.violinplot(
            data=adata.obs, 
            x=plot_col, 
            y='scTour_Pseudotime_Norm', 
            order=unique_clusters,
            palette=palette,
            inner=None, # Remove default inner
            linewidth=0, # Remove border to make it look cleaner before adding boxplot
            alpha=0.6,
            scale="width"
        )
        
        # Add black and white Boxplot inside the violin
        sns.boxplot(
            data=adata.obs, 
            x=plot_col, 
            y='scTour_Pseudotime_Norm', 
            order=unique_clusters,
            width=0.2,
            boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 2},
            whiskerprops={'color': 'black', 'zorder': 2},
            capprops={'color': 'black', 'zorder': 2},
            medianprops={'color': 'black', 'zorder': 2},
            showfliers=False, # Hide outliers for cleaner look
            ax=ax
        )
        
        plt.title(f'{method} Subclusters Pseudotime (scTour)', fontsize=14, fontweight='bold')
        plt.xlabel(f'{method} Subclusters', fontsize=12)
        plt.ylabel('Normalized scTour Pseudotime', fontsize=12)
        plt.xticks(rotation=45)
        
        out_file = os.path.join(data_dir, f"Violin_Pseudotime_{method}_{prefix}.pdf")
        plt.tight_layout()
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_file}")

if __name__ == "__main__":
    # Suppress sctour warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore')
    
    compute_and_plot_pseudotime(4, 1.5)
    compute_and_plot_pseudotime(4, 2.0)
