import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_ccvae/nature_medicine"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

print("=== Starting High-Level Nature Medicine Style TME Analysis ===")

print("Loading finely annotated subclusters...")
adata_mac = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Macrophages_adata_sctour_ccvae_fixed.h5ad")
adata_fib = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Fibroblasts_adata_sctour_ccvae_fixed.h5ad")
adata_t = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Tcells_adata_sctour_ccvae_fixed.h5ad")

# --- 1. Metabolic Reprogramming along Pseudotime ---
print("1. Profiling Metabolic Reprogramming along Differentiation Trajectory...")
metabolism_genes = {
    'Glycolysis': ['ENO1', 'HK2', 'PGK1', 'LDHA', 'ALDOA', 'GAPDH', 'PKM'],
    'Lipid_Metabolism': ['CD36', 'FASN', 'SREBF1', 'SCD', 'ACACA', 'PPARG', 'APOE'],
    'OXPHOS': ['COX5A', 'COX6C', 'NDUFA4', 'ATP5F1A', 'UQCRB']
}

for adata, name in zip([adata_mac, adata_t], ['Macrophages', 'Tcells']):
    for path_name, genes in metabolism_genes.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=path_name)
    
    if 'ptime' in adata.obs.columns:
        # Create pseudotime bins
        adata.obs['Pseudotime_Bin'] = pd.qcut(adata.obs['ptime'], q=5, labels=['Early', 'Early-Mid', 'Mid', 'Late-Mid', 'Late'])
        
        metab_cols = [p for p in metabolism_genes.keys() if p in adata.obs.columns]
        if metab_cols:
            df = adata.obs[['Pseudotime_Bin', 'CellType_Fine'] + metab_cols].dropna()
            df_melt = df.melt(id_vars=['Pseudotime_Bin', 'CellType_Fine'], value_vars=metab_cols, var_name='Pathway', value_name='Score')
            
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df_melt, x='Pseudotime_Bin', y='Score', hue='Pathway', palette='Set2', showfliers=False)
            plt.title(f"{name}: Metabolic Reprogramming during Cell Evolution")
            plt.xlabel("Pseudotime Stage (scTour)")
            plt.ylabel("Pathway Enrichment Score")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/NM_Fig1_Metabolic_Trajectory_{name}.pdf")
            plt.close()

# --- 2. Stromal-Immune Interactome (Receptor-Ligand Network) ---
print("2. Mapping Pan-Cancer Stromal-Immune Interactome (Receptor-Ligand)...")
rl_pairs = [
    ('SPP1', 'CD44'), ('CD274', 'PDCD1'), ('PDCD1LG2', 'PDCD1'), 
    ('NECTIN2', 'TIGIT'), ('PVR', 'TIGIT'), ('LGALS9', 'HAVCR2'),
    ('TGFB1', 'TGFBR1'), ('CXCL12', 'CXCR4'), ('IL10', 'IL10RA')
]

def get_avg_expr(adata, genes):
    valid = [g for g in genes if g in adata.var_names]
    if not valid: return pd.DataFrame()
    X = adata[:, valid].X
    if hasattr(X, 'toarray'): X = X.toarray()
    df = pd.DataFrame(X, columns=valid, index=adata.obs_names)
    df['Cancer_Type'] = adata.obs['Cancer_Type'].values
    return df.groupby('Cancer_Type').mean()

ligands = list(set([p[0] for p in rl_pairs]))
receptors = list(set([p[1] for p in rl_pairs]))

mac_expr = get_avg_expr(adata_mac, ligands)
fib_expr = get_avg_expr(adata_fib, ligands)
t_expr = get_avg_expr(adata_t, receptors)

cancer_types = set(adata_mac.obs['Cancer_Type'].unique()) & set(adata_t.obs['Cancer_Type'].unique())
interaction_data = []

for ct in cancer_types:
    for lig, rec in rl_pairs:
        if lig in mac_expr.columns and rec in t_expr.columns and ct in mac_expr.index and ct in t_expr.index:
            score = mac_expr.loc[ct, lig] * t_expr.loc[ct, rec]
            interaction_data.append({'Cancer': ct, 'Source': 'TAM', 'Pair': f"{lig}(Mac)-{rec}(T)", 'Score': score})
        if lig in fib_expr.columns and rec in t_expr.columns and ct in fib_expr.index and ct in t_expr.index:
            score = fib_expr.loc[ct, lig] * t_expr.loc[ct, rec]
            interaction_data.append({'Cancer': ct, 'Source': 'CAF', 'Pair': f"{lig}(Fib)-{rec}(T)", 'Score': score})

if interaction_data:
    int_df = pd.DataFrame(interaction_data)
    int_pivot = int_df.pivot_table(index='Pair', columns='Cancer', values='Score').fillna(0)
    
    # Normalize by row for better visualization of cancer specificity
    int_pivot_norm = int_pivot.div(int_pivot.max(axis=1), axis=0)
    
    plt.figure(figsize=(8, 10))
    sns.heatmap(int_pivot_norm, cmap='Reds', annot=int_pivot.round(2), fmt=".2f", linewidths=.5)
    plt.title("Stromal-Immune Checkpoint Interaction Potential")
    plt.xlabel("Cancer Type")
    plt.ylabel("Ligand(Source) - Receptor(T cell)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/NM_Fig2_Interactome_Heatmap.pdf")
    plt.close()

# --- 3. Co-abundance of Pathogenic TME Niche ---
print("3. Discovering Conserved Pathogenic TME Niche Co-abundance...")
# Use 'batch' (dataset/sample proxy) to calculate correlation of proportions
batch_col = 'batch' if 'batch' in adata_mac.obs.columns else 'Cancer_Type'

def get_proportions(adata, prefix):
    counts = adata.obs.groupby([batch_col, 'CellType_Fine']).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    props.columns = [f"{prefix}_{c}" for c in props.columns]
    return props

prop_mac = get_proportions(adata_mac, 'Mac')
prop_fib = get_proportions(adata_fib, 'Fib')
prop_t = get_proportions(adata_t, 'T')

merged_props = pd.concat([prop_mac, prop_fib, prop_t], axis=1).fillna(0)
corr_matrix = merged_props.corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='vlag', center=0, vmin=-1, vmax=1, 
            annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("TME Niche Co-abundance Network (Spearman Correlation across samples)")
plt.tight_layout()
plt.savefig(f"{out_dir}/NM_Fig3_Pathogenic_Niche_CoAbundance.pdf")
plt.close()

# --- 4. Master Transcription Factor (TF) Target Signatures ---
print("4. Scoring Master Transcription Factor Activities...")
tf_targets = {
    'HIF1A_Targets (Hypoxia)': ['VEGFA', 'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'ALDOA'],
    'STAT3_Targets (Inflammation)': ['IL10', 'SOCS3', 'MCL1', 'BCL2', 'MYC', 'VEGFA'],
    'MYC_Targets (Proliferation)': ['CDK4', 'CCND1', 'E2F1', 'ODC1', 'NCL']
}

for adata, name in zip([adata_mac, adata_fib, adata_t], ['Macrophages', 'Fibroblasts', 'Tcells']):
    for tf, genes in tf_targets.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=tf)
            
    tf_cols = [t for t in tf_targets.keys() if t in adata.obs.columns]
    if tf_cols:
        sc.pl.matrixplot(adata, tf_cols, groupby='CellType_Fine', cmap='magma', standard_scale='var',
                         save=f"_NM_Fig4_MasterRegulators_{name}.pdf", show=False)

print("=== Analysis Complete. Check 'results_new_cross_cancer_ccvae/nature_medicine/' ===")
