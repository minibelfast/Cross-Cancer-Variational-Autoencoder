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

out_dir = "results_real_cross_cancer/fibroblast/tf_analysis"
os.makedirs(out_dir, exist_ok=True)

print("Loading Fibroblast data...")
# Load annotated data for labels
adata_fine = sc.read_h5ad("results_real_cross_cancer/06_Fibroblast_adata_fine.h5ad")
# Load sctour data for pseudotime
adata_sct = sc.read_h5ad("results_real_cross_cancer/06_Fibroblast_sct_adata.h5ad")

# Transfer pseudotime to fine data
adata_fine.obs['pseudotime'] = adata_sct.obs['pseudotime']

adata = adata_fine.copy()

# Filter out Unknown
if 'Unknown' in adata.obs['CellType_Fine'].values:
    adata = adata[adata.obs['CellType_Fine'] != 'Unknown'].copy()

# Known TFs
putative_tfs = [
    'SMAD2', 'SMAD3', 'TWIST1', 'SNAI1', 'ZEB1', 'YAP1', 'MYC', 'SRF', # myCAF
    'NFKB1', 'RELA', 'STAT3', 'CEBPB', 'HIF1A', 'FOSL1', 'JUNB',       # iCAF
    'STAT1', 'IRF1', 'CIITA', 'RFX5', 'IRF8', 'NLRC5'                  # apCAF
]

valid_tfs = [tf for tf in putative_tfs if tf in adata.var_names]
print(f"Found {len(valid_tfs)} curated TFs in the dataset.")

print("Calculating DEGs to expand TF list...")
sc.tl.rank_genes_groups(adata, groupby='CellType_Fine', method='wilcoxon')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

top_genes = set()
for group in groups:
    top_genes.update(result['names'][group][:200])

possible_tf_families = ('STAT', 'IRF', 'SMAD', 'FOX', 'HOX', 'SOX', 'TCF', 'LEF', 'GATA', 'E2F', 'MYC', 'NFKB', 'REL', 'CEBP', 'KLF', 'HIF', 'JUN', 'FOS', 'ETS', 'ZNF', 'RUNX', 'EGR', 'ATF')
for g in top_genes:
    if any(g.startswith(fam) for fam in possible_tf_families):
        if g not in valid_tfs and g in adata.var_names:
            valid_tfs.append(g)

valid_tfs = list(set(valid_tfs))
print(f"Expanded to {len(valid_tfs)} potential TFs.")

print("Analyzing pseudotime dynamics...")
ptime = adata.obs['pseudotime'].values
cell_types = adata.obs['CellType_Fine'].values

df_dyn = pd.DataFrame(index=adata.obs_names)
df_dyn['Pseudotime'] = ptime
df_dyn['CellType'] = cell_types

if hasattr(adata.X, "toarray"):
    expr_matrix = adata[:, valid_tfs].X.toarray()
else:
    expr_matrix = adata[:, valid_tfs].X

expr_df = pd.DataFrame(expr_matrix, columns=valid_tfs, index=adata.obs_names)
df_dyn = pd.concat([df_dyn, expr_df], axis=1)

df_dyn = df_dyn.sort_values('Pseudotime')

mean_ptime = df_dyn.groupby('CellType')['Pseudotime'].mean().sort_values()
print("Mean pseudotime per subtype:")
print(mean_ptime)

bins = pd.cut(df_dyn['Pseudotime'], bins=20)
binned_expr = df_dyn.groupby(bins)[valid_tfs].mean()

binned_expr = binned_expr.loc[:, binned_expr.max() > 0.1]
tf_variance = binned_expr.var().sort_values(ascending=False)
top_dynamic_tfs = tf_variance.head(12).index.tolist()

print(f"Top dynamic TFs: {top_dynamic_tfs}")

plt.figure(figsize=(15, 10))
for i, tf in enumerate(top_dynamic_tfs):
    plt.subplot(3, 4, i+1)
    sns.regplot(data=df_dyn, x='Pseudotime', y=tf, scatter_kws={'s': 1, 'alpha': 0.1, 'color': 'gray'}, line_kws={'color': 'red'}, lowess=True)
    
    plt.title(tf)
    plt.xlabel('scTour Pseudotime')
    plt.ylabel('Expression')

plt.tight_layout()
plt.savefig(f"{out_dir}/Top_Dynamic_TFs_Pseudotime.pdf", bbox_inches='tight')
plt.close()

sc.pl.matrixplot(adata, valid_tfs, groupby='CellType_Fine', standard_scale='var', 
                 cmap='viridis', show=False, figsize=(12, 6))
plt.title('TF Expression Across Fibroblast Subtypes')
plt.savefig(f"{out_dir}/TF_Expression_Matrixplot.pdf", bbox_inches='tight')
plt.close()

corr_matrix = expr_df[top_dynamic_tfs].corr()
corr_matrix.to_csv(f"{out_dir}/Top_TFs_Correlation_Matrix.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Correlation Network of Key TFs in Transition')
plt.savefig(f"{out_dir}/Top_TFs_Correlation_Heatmap.pdf", bbox_inches='tight')
plt.close()

summary_df = pd.DataFrame({'TF': top_dynamic_tfs})
summary_df.to_csv(f"{out_dir}/Key_Transition_TFs.csv", index=False)

print("=== TF Analysis Complete ===")
