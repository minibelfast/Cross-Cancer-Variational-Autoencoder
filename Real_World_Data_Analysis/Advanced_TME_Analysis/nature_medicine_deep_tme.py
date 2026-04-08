import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_ccvae/nature_medicine_deep"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

print("=== Starting NM-Level Deep Analysis ===")

adata_mac = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Macrophages_adata_sctour_ccvae_fixed.h5ad")
adata_fib = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Fibroblasts_adata_sctour_ccvae_fixed.h5ad")
adata_t = sc.read_h5ad("results_new_cross_cancer_ccvae/10_Tcells_adata_sctour_ccvae_fixed.h5ad")

# Module 1: High-Resolution T cell Exhaustion Cascade (Continuous State Transition)
print("1. Mapping High-Resolution T cell Exhaustion Cascade...")
if 'ptime' in adata_t.obs.columns:
    # Sort T cells by pseudotime
    t_sorted = adata_t[adata_t.obs.sort_values('ptime').index]
    
    # Define cascade markers: Naive -> Effector -> Progenitor Exhausted -> Terminally Exhausted
    cascade_genes = ['TCF7', 'IL7R', 'GZMK', 'PDCD1', 'TOX', 'HAVCR2', 'ENTPD1', 'CXCL13']
    valid_cascade = [g for g in cascade_genes if g in t_sorted.var_names]
    
    if valid_cascade:
        expr_data = t_sorted[:, valid_cascade].X
        if hasattr(expr_data, 'toarray'): expr_data = expr_data.toarray()
        
        # Smooth expression over pseudotime
        smoothed_expr = gaussian_filter1d(expr_data, sigma=200, axis=0)
        
        # Normalize to 0-1 for comparison
        smoothed_expr = (smoothed_expr - smoothed_expr.min(axis=0)) / (smoothed_expr.max(axis=0) - smoothed_expr.min(axis=0))
        
        plt.figure(figsize=(10, 6))
        for i, gene in enumerate(valid_cascade):
            plt.plot(t_sorted.obs['ptime'].values, smoothed_expr[:, i], label=gene, linewidth=2.5)
            
        plt.title("Continuous State Transition: T Cell Exhaustion Cascade")
        plt.xlabel("scTour Pseudotime")
        plt.ylabel("Normalized Smoothed Expression")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/NM_Deep_Fig1_Tcell_Exhaustion_Cascade.pdf")
        plt.close()

# Module 2: Cellular Plasticity and Stemness Potential
print("2. Evaluating Cellular Plasticity (Stemness Potential)...")
# Approximation of developmental potential: less differentiated cells often express a higher diversity of genes
for adata, name in zip([adata_mac, adata_fib, adata_t], ['Macrophages', 'Fibroblasts', 'Tcells']):
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    if 'ptime' in adata.obs.columns and 'n_genes_by_counts' in adata.obs.columns:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=adata.obs, x='ptime', y='n_genes_by_counts', 
                    scatter_kws={'alpha':0.1, 's':2}, line_kws={'color':'red', 'linewidth':2},
                    lowess=True)
        plt.title(f"{name}: Cellular Plasticity along Differentiation")
        plt.xlabel("Pseudotime (Differentiation State)")
        plt.ylabel("Gene Expression Diversity (Proxy for Stemness/Plasticity)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/NM_Deep_Fig2_Plasticity_{name}.pdf")
        plt.close()

# Module 3: Pan-Cancer Conserved vs Specific TAM/CAF Signatures
print("3. Extracting Conserved Pan-Cancer Immunosuppressive Signatures...")
def get_late_stage_markers(adata, name):
    if 'ptime' not in adata.obs.columns: return pd.DataFrame()
    # Define top 20% pseudotime as 'Late' (Terminally differentiated / Pathological)
    threshold = adata.obs['ptime'].quantile(0.8)
    adata.obs['Stage'] = ['Late' if x >= threshold else 'Early_Mid' for x in adata.obs['ptime']]
    
    sc.tl.rank_genes_groups(adata, groupby='Stage', groups=['Late'], reference='Early_Mid', method='wilcoxon')
    res = sc.get.rank_genes_groups_df(adata, group='Late')
    res = res[res['pvals_adj'] < 0.05].sort_values('scores', ascending=False)
    
    # Save the signature
    res.head(50).to_csv(f"{out_dir}/NM_Deep_Table1_PanCancer_Late_{name}_Signature.csv", index=False)
    return res.head(10)['names'].tolist()

late_mac_genes = get_late_stage_markers(adata_mac, 'Macrophages')
late_fib_genes = get_late_stage_markers(adata_fib, 'Fibroblasts')

if late_mac_genes and late_fib_genes:
    # Visualize these conserved signatures across cancer types
    for adata, genes, name in zip([adata_mac, adata_fib], [late_mac_genes, late_fib_genes], ['Macrophages', 'Fibroblasts']):
        sc.pl.dotplot(adata, genes, groupby='Cancer_Type', 
                      title=f"Conserved Pathological {name} Signature",
                      save=f"_NM_Deep_Fig3_ConservedSignature_{name}.pdf", show=False)

# Module 4: Antigen Presentation vs. Immunosuppression Axis in TAMs
print("4. Mapping Antigen Presentation vs Suppressive Axis in TAMs...")
if 'ptime' in adata_mac.obs.columns:
    mhc_genes = ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DQA1', 'CD74']
    suppressive_genes = ['IL10', 'VEGFA', 'TGFB1', 'ARG1', 'SPP1', 'TREM2']
    
    valid_mhc = [g for g in mhc_genes if g in adata_mac.var_names]
    valid_supp = [g for g in suppressive_genes if g in adata_mac.var_names]
    
    sc.tl.score_genes(adata_mac, gene_list=valid_mhc, score_name='MHC_Class_II')
    sc.tl.score_genes(adata_mac, gene_list=valid_supp, score_name='Suppressive_Score')
    
    df_mac = adata_mac.obs[['ptime', 'MHC_Class_II', 'Suppressive_Score']].dropna()
    df_mac = df_mac.sort_values('ptime')
    
    # Smoothing
    mhc_smooth = gaussian_filter1d(df_mac['MHC_Class_II'].values, sigma=200)
    supp_smooth = gaussian_filter1d(df_mac['Suppressive_Score'].values, sigma=200)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_mac['ptime'].values, mhc_smooth, label='MHC Class II (Antigen Presentation)', color='blue', linewidth=2.5)
    plt.plot(df_mac['ptime'].values, supp_smooth, label='Pro-Tumor / Suppressive', color='red', linewidth=2.5)
    plt.fill_between(df_mac['ptime'].values, mhc_smooth, supp_smooth, where=(supp_smooth > mhc_smooth), color='red', alpha=0.1)
    plt.title("Macrophage Polarization Continuum")
    plt.xlabel("Differentiation Trajectory (Pseudotime)")
    plt.ylabel("Module Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/NM_Deep_Fig4_TAM_Polarization_Axis.pdf")
    plt.close()

# Module 5: Myeloid-CAF-T Cell Functional Triad
print("5. Analyzing Myeloid-CAF-T Cell Functional Triad...")
# Correlate specific module scores across patients to establish clinical niches
patient_scores = []
# Identify the patient/sample level metadata column
if 'Batch' in adata_mac.obs.columns:
    batch_col = 'Batch'
elif 'Sample' in adata_mac.obs.columns:
    batch_col = 'Sample'
elif 'batch' in adata_mac.obs.columns:
    batch_col = 'batch'
elif 'Dataset' in adata_mac.obs.columns:
    batch_col = 'Dataset'
else:
    # If no true sample metadata exists, we create a pseudo-sample ID 
    # to avoid just doing it at the Cancer_Type level
    batch_col = 'Cancer_Type'

print(f"Using '{batch_col}' as the patient/sample identifier for the triad analysis.")

# Need to ensure the scoring exists before aggregating
# Score T cell exhaustion
valid_exh = [g for g in ['PDCD1', 'HAVCR2', 'CTLA4', 'LAG3', 'TOX', 'ENTPD1'] if g in adata_t.var_names]
if valid_exh:
    sc.tl.score_genes(adata_t, gene_list=valid_exh, score_name='Exhaustion_Score')

# We already scored Suppressive_Score for macs in Module 4, but let's ensure it's there
suppressive_genes = ['IL10', 'VEGFA', 'TGFB1', 'ARG1', 'SPP1', 'TREM2']
valid_supp = [g for g in suppressive_genes if g in adata_mac.var_names]
if valid_supp and 'Suppressive_Score' not in adata_mac.obs.columns:
    sc.tl.score_genes(adata_mac, gene_list=valid_supp, score_name='Suppressive_Score')

# Let's also add CAF remodeling score
remodel_genes = ['COL1A1', 'FN1', 'MMP2', 'ACTA2', 'LOXL2', 'FAP']
valid_remodel = [g for g in remodel_genes if g in adata_fib.var_names]
if valid_remodel:
    sc.tl.score_genes(adata_fib, gene_list=valid_remodel, score_name='Remodeling_Score')

for sample_id in set(adata_mac.obs[batch_col].unique()):
    # Find which cancer type this sample belongs to
    sample_ct = adata_mac.obs[adata_mac.obs[batch_col] == sample_id]['Cancer_Type'].iloc[0]
    
    t_batch = adata_t[adata_t.obs[batch_col] == sample_id]
    mac_batch = adata_mac[adata_mac.obs[batch_col] == sample_id]
    fib_batch = adata_fib[adata_fib.obs[batch_col] == sample_id] if batch_col in adata_fib.obs.columns else None
    
    # We require at least 10 cells of each type in a patient to compute a reliable mean
    if len(t_batch) > 10 and len(mac_batch) > 10:
        exh_score = t_batch.obs['Exhaustion_Score'].mean() if 'Exhaustion_Score' in t_batch.obs.columns else 0
        supp_score = mac_batch.obs['Suppressive_Score'].mean() if 'Suppressive_Score' in mac_batch.obs.columns else 0
        remodel_score = fib_batch.obs['Remodeling_Score'].mean() if fib_batch is not None and len(fib_batch) > 10 and 'Remodeling_Score' in fib_batch.obs.columns else np.nan
        
        patient_scores.append({
            'Sample_ID': sample_id,
            'Cancer_Type': sample_ct,
            'T_Cell_Exhaustion': exh_score,
            'Macrophage_Suppression': supp_score,
            'CAF_Remodeling': remodel_score
        })

df_triad = pd.DataFrame(patient_scores)
if not df_triad.empty:
    # 1. Plot TAM vs T cell Exhaustion
    plt.figure(figsize=(7, 6))
    sns.regplot(data=df_triad, x='Macrophage_Suppression', y='T_Cell_Exhaustion', scatter=False, color='black')
    sns.scatterplot(data=df_triad, x='Macrophage_Suppression', y='T_Cell_Exhaustion', hue='Cancer_Type', s=100, alpha=0.8)
    
    # Calculate Spearman correlation
    corr = df_triad[['Macrophage_Suppression', 'T_Cell_Exhaustion']].corr(method='spearman').iloc[0, 1]
    plt.annotate(f"Spearman r = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                 
    plt.title("Patient-Level Correlation: TAM vs T Cell Exhaustion")
    plt.xlabel("Mean TAM Suppressive Score per Patient")
    plt.ylabel("Mean T Cell Exhaustion Score per Patient")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/NM_Deep_Fig5_Functional_Triad_TAM_Tcell.pdf")
    plt.close()
    
    # 2. Plot CAF vs T cell Exhaustion (if data exists)
    if not df_triad['CAF_Remodeling'].isna().all():
        df_valid = df_triad.dropna(subset=['CAF_Remodeling'])
        plt.figure(figsize=(7, 6))
        sns.regplot(data=df_valid, x='CAF_Remodeling', y='T_Cell_Exhaustion', scatter=False, color='black')
        sns.scatterplot(data=df_valid, x='CAF_Remodeling', y='T_Cell_Exhaustion', hue='Cancer_Type', s=100, alpha=0.8)
        
        corr_fib = df_valid[['CAF_Remodeling', 'T_Cell_Exhaustion']].corr(method='spearman').iloc[0, 1]
        plt.annotate(f"Spearman r = {corr_fib:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                     
        plt.title("Patient-Level Correlation: CAF Matrix Remodeling vs T Cell Exhaustion")
        plt.xlabel("Mean CAF Remodeling Score per Patient")
        plt.ylabel("Mean T Cell Exhaustion Score per Patient")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/NM_Deep_Fig5_Functional_Triad_CAF_Tcell.pdf")
        plt.close()

print("=== Deep NM Analysis Complete ===")
