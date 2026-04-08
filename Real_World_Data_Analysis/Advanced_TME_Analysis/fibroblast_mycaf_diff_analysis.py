import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gseapy as gp
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer/fibroblast/myCAF_BLCA_vs_HCC"
os.makedirs(out_dir, exist_ok=True)

print("Loading CC-VAE Fibroblast data...")
adata = sc.read_h5ad("results_real_cross_cancer/06_Fibroblast_adata_fine.h5ad")

print("Subsetting to myCAF in Bladder and HCC...")
# Select myCAF
adata_mycaf = adata[adata.obs['CellType_Fine'].str.contains('myCAF', na=False)].copy()
# Select Bladder and HCC
adata_sub = adata_mycaf[adata_mycaf.obs['Cancer_Type'].isin(['Bladder', 'HCC'])].copy()

# Remove genes with 0 counts in this subset to avoid division by zero
sc.pp.filter_genes(adata_sub, min_cells=3)

print("Running Differential Expression (Bladder vs HCC within myCAF)...")
# We rank Bladder against HCC
sc.tl.rank_genes_groups(adata_sub, groupby='Cancer_Type', groups=['Bladder'], reference='HCC', method='wilcoxon')

result = adata_sub.uns['rank_genes_groups']
df_volcano = pd.DataFrame({
    'Gene': result['names']['Bladder'],
    'Log2FC': result['logfoldchanges']['Bladder'],
    'Pval': result['pvals']['Bladder'],
    'Pval_adj': result['pvals_adj']['Bladder']
})

df_volcano['-log10(Pval_adj)'] = -np.log10(df_volcano['Pval_adj'] + 1e-300)
df_volcano['Significance'] = 'Not Sig'
df_volcano.loc[(df_volcano['Log2FC'] > 1) & (df_volcano['Pval_adj'] < 0.05), 'Significance'] = 'Up in Bladder myCAF'
df_volcano.loc[(df_volcano['Log2FC'] < -1) & (df_volcano['Pval_adj'] < 0.05), 'Significance'] = 'Up in HCC myCAF'

df_volcano.to_csv(f"{out_dir}/myCAF_BLCA_vs_HCC_DEGs.csv", index=False)

print("Generating Volcano Plot...")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_volcano, x='Log2FC', y='-log10(Pval_adj)', hue='Significance', 
                palette={'Not Sig': 'lightgrey', 'Up in Bladder myCAF': 'red', 'Up in HCC myCAF': 'blue'}, 
                s=20, edgecolor=None, alpha=0.8)

# Add text for top genes
top_bladder = df_volcano[df_volcano['Significance'] == 'Up in Bladder myCAF'].nlargest(15, 'Log2FC')
top_hcc = df_volcano[df_volcano['Significance'] == 'Up in HCC myCAF'].nsmallest(15, 'Log2FC')

import matplotlib.patheffects as pe
for i, row in pd.concat([top_bladder, top_hcc]).iterrows():
    plt.text(row['Log2FC'], row['-log10(Pval_adj)'], row['Gene'], fontsize=8,
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])

plt.title('Differential Expression: myCAF (Bladder vs HCC)')
plt.axvline(1, color='k', linestyle='--', alpha=0.5)
plt.axvline(-1, color='k', linestyle='--', alpha=0.5)
plt.axhline(-np.log10(0.05), color='k', linestyle='--', alpha=0.5)
plt.xlabel('Log2 Fold Change (Bladder / HCC)')
plt.ylabel('-log10(Adjusted P-value)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{out_dir}/myCAF_BLCA_vs_HCC_Volcano.pdf", bbox_inches='tight')
plt.close()

print("Running GSEAPY Enrichment...")
up_bladder_genes = df_volcano[df_volcano['Significance'] == 'Up in Bladder myCAF']['Gene'].tolist()
up_hcc_genes = df_volcano[df_volcano['Significance'] == 'Up in HCC myCAF']['Gene'].tolist()

gene_sets = ['GO_Biological_Process_2021', 'KEGG_2021_Human']

def plot_enrichment(genes, title, color, filename):
    if len(genes) > 0:
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism='human', outdir=None)
            res = enr.results
            res = res[res['Adjusted P-value'] < 0.05]
            if not res.empty:
                res.to_csv(f"{out_dir}/{filename}.csv", index=False)
                # Sort by adjusted P-value and take top 15
                res = res.sort_values('Adjusted P-value').head(15)
                res['-log10(P)'] = -np.log10(res['Adjusted P-value'])
                
                # Clean up GO terms for display
                res['Term_Clean'] = res['Term'].apply(lambda x: x.split(' (GO:')[0] if ' (GO:' in x else x)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=res, x='-log10(P)', y='Term_Clean', color=color)
                plt.title(title)
                plt.xlabel('-log10(Adjusted P-value)')
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(f"{out_dir}/{filename}_Barplot.pdf", bbox_inches='tight')
                plt.close()
                return True
        except Exception as e:
            print(f"Enrichment failed for {title}: {e}")
    return False

plot_enrichment(up_bladder_genes, 'Top Enriched Pathways in Bladder myCAF', 'red', 'myCAF_Bladder_Enrichment')
plot_enrichment(up_hcc_genes, 'Top Enriched Pathways in HCC myCAF', 'blue', 'myCAF_HCC_Enrichment')

# Plot Dotplot of top DEGs across these two myCAFs
# Get top 20 for each
top_genes = top_bladder['Gene'].tolist() + top_hcc['Gene'].tolist()
sc.pl.dotplot(adata_sub, top_genes, groupby='Cancer_Type', standard_scale='var', 
              cmap='viridis', show=False, figsize=(15, 4))
plt.title('Top DEGs Expression in myCAF (Bladder vs HCC)')
plt.savefig(f"{out_dir}/myCAF_TopDEGs_Dotplot.pdf", bbox_inches='tight')
plt.close()

print("=== myCAF Comparative Analysis Complete ===")
