import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

out_dir = "results_new_cross_cancer_ccvae"
os.makedirs(f"{out_dir}/downstream", exist_ok=True)

cell_types = ['Macrophages', 'Fibroblasts', 'T cells']

print("=== 1. Calculating Cell Type Proportions across Cancers ===")
for ct in cell_types:
    in_file = f"{out_dir}/08_{ct.replace(' ', '')}_adata_fine_ccvae.h5ad"
    if not os.path.exists(in_file):
        continue
    
    adata = sc.read_h5ad(in_file)
    
    props = adata.obs.groupby(['Cancer_Type', 'CellType_Fine']).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0) * 100
    
    props.to_csv(f"{out_dir}/downstream/{ct}_Proportions_by_Cancer.csv")
    
    ax = props.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='tab20')
    plt.title(f"{ct} Subtype Proportions by Cancer Type (CC-VAE)")
    plt.ylabel("Percentage (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/downstream/{ct}_Proportions_Barplot.pdf")
    plt.close()

print("=== 2. Common Mechanism Discovery (Pseudotime & Pathways) ===")
pathways = {
    'Angiogenesis': ['VEGFA', 'VEGFB', 'KDR', 'PECAM1', 'FLT1'],
    'Hypoxia': ['HIF1A', 'EPAS1', 'ARNT', 'SLC2A1'],
    'Immunosuppression': ['IL10', 'TGFB1', 'CD274', 'IDO1', 'PDCD1LG2'],
    'T_cell_exhaustion': ['PDCD1', 'LAG3', 'HAVCR2', 'CTLA4', 'TIGIT']
}

for ct in cell_types:
    in_file = f"{out_dir}/10_{ct.replace(' ', '')}_adata_sctour_ccvae_fixed.h5ad"
    if not os.path.exists(in_file):
        print(f"File {in_file} not found, skipping downstream...")
        continue
        
    adata = sc.read_h5ad(in_file)
    
    for path_name, genes in pathways.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=path_name)
            
    path_cols = [p for p in pathways.keys() if p in adata.obs.columns]
    if path_cols:
        sc.pl.dotplot(adata, path_cols, groupby='CellType_Fine', 
                      save=f"_downstream_{ct}_Pathways.pdf", show=False)
                      
    # We used 'ptime' in the corrected sctour script
    if 'ptime' in adata.obs.columns:
        print(f"Finding genes correlated with pseudotime in {ct}...")
        sc.pp.highly_variable_genes(adata, n_top_genes=1000)
        adata_hv = adata[:, adata.var['highly_variable']].copy()
        
        pt = adata_hv.obs['ptime'].values
        exprs = adata_hv.X.toarray() if hasattr(adata_hv.X, 'toarray') else adata_hv.X
        
        corrs = []
        for i in range(exprs.shape[1]):
            gene_expr = exprs[:, i]
            if np.std(gene_expr) > 0 and np.std(pt) > 0:
                corr = np.corrcoef(gene_expr, pt)[0, 1]
            else:
                corr = 0
            corrs.append(corr)
            
        corr_df = pd.DataFrame({
            'Gene': adata_hv.var_names,
            'Correlation': corrs
        }).sort_values('Correlation', ascending=False)
        
        corr_df.to_csv(f"{out_dir}/downstream/{ct}_Pseudotime_Correlated_Genes.csv", index=False)
        
        top_genes = corr_df.head(5)['Gene'].tolist() + corr_df.tail(5)['Gene'].tolist()
        
        sc.pl.matrixplot(adata, top_genes, groupby='CellType_Fine', 
                         save=f"_downstream_{ct}_Top_Pseudotime_Genes.pdf", show=False)

print("=== Downstream Analysis Complete ===")
