import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"

pathways_dict = {
    'Macrophages': {
        'M1: TNFa Signaling via NFkB': ['TNF', 'NFKB1', 'REL', 'RELA', 'CCL2', 'CXCL8', 'CXCL10', 'IL1B'],
        'M1: Inflammatory Response': ['IL1A', 'IL1B', 'IL6', 'CXCL8', 'PTGS2', 'CCL5', 'CXCL9'],
        'M1: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'IDO1', 'GBP1'],
        'M1: Antigen Processing and Presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'B2M', 'TAP1', 'TAP2', 'CD74'],
        'M2: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'SMAD2', 'SMAD3', 'TGFBI', 'THBS1'],
        'M2: IL4 and IL13 Signaling': ['IL4R', 'IL13RA1', 'STAT6', 'ARG1', 'CCL18', 'CD209', 'MRC1'],
        'M2: Tissue Remodeling/Wound Healing': ['MMP9', 'MMP14', 'TIMP1', 'FN1', 'COL1A1', 'VIM'],
        'SPP1: Hypoxia Response': ['HIF1A', 'EPAS1', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'ALDOA'],
        'SPP1: Angiogenesis': ['VEGFA', 'KDR', 'PECAM1', 'CD34', 'TEK', 'FLT1', 'ANGPT1', 'ANGPT2'],
        'SPP1: Glycolysis': ['HK2', 'PFKP', 'PKM', 'GAPDH', 'ENO1', 'LDHA', 'ALDOA', 'PGK1'],
        'C1QC: Complement Activation': ['C1QA', 'C1QB', 'C1QC', 'C3', 'CFB', 'C4A', 'CR1'],
        'C1QC: Phagocytosis/Efferocytosis': ['CD36', 'MERTK', 'MARCO', 'CD68', 'FCGR3A', 'SIRPA', 'AXL'],
        'C1QC: Lipid Metabolism': ['APOE', 'APOC1', 'ABCA1', 'ABCG1', 'LPL', 'TREM2'],
        'Cycling: G2M Checkpoint': ['TOP2A', 'CDK1', 'BIRC5', 'UBE2C', 'CENPF', 'MKI67', 'PTTG1', 'AURKA'],
        'Cycling: E2F Targets': ['MCM2', 'MCM4', 'MCM6', 'PCNA', 'TYMS', 'FEN1', 'RFC4'],
        'Cycling: Cell Cycle': ['CCNB1', 'CCNA2', 'CCNE1', 'CDK2', 'CDK4', 'CHEK1']
    },
    'Fibroblasts': {
        'myCAF: ECM Organization/Collagen Formation': ['COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'LUM', 'DCN', 'BGN', 'POSTN'],
        'myCAF: Smooth Muscle Contraction': ['ACTA2', 'TAGLN', 'MYLK', 'TPM1', 'MYL9', 'CNN1'],
        'myCAF: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'THBS1'],
        'myCAF: Focal Adhesion': ['ITGB1', 'ITGA5', 'ITGA1', 'CAV1', 'VCL', 'FLNA'],
        'myCAF: Angiogenesis Regulation': ['VEGFA', 'FGF2', 'PDGFA', 'PDGFB', 'HGF'],
        'iCAF: Inflammatory Response / Cytokine': ['IL6', 'CXCL12', 'CCL2', 'CXCL1', 'CXCL2', 'CXCL8', 'IL1B'],
        'iCAF: Complement and Coagulation': ['C3', 'CFB', 'CFD', 'C1R', 'C1S', 'SERPING1'],
        'iCAF: TNF Signaling via NFkB': ['TNF', 'NFKB1', 'RELB', 'PTGS2', 'LIF'],
        'iCAF: IL6_JAK_STAT3 Signaling': ['IL6', 'JAK1', 'STAT3', 'SOCS3', 'PIM1'],
        'apCAF: MHC Class II Antigen Presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'CD74', 'B2M'],
        'apCAF: T Cell Activation / Costimulation': ['CD80', 'CD86', 'CD40', 'ICOS', 'TNFSF4'],
        'apCAF: Interferon Gamma Response': ['STAT1', 'IRF1', 'CIITA', 'RFX5']
    },
    'T cells': {
        'Exhausted: T Cell Exhaustion/Coinhibition': ['PDCD1', 'HAVCR2', 'LAG3', 'CTLA4', 'TIGIT', 'TOX', 'ENTPD1'],
        'Exhausted: Chronic Antigen Stimulation': ['BATF', 'IRF4', 'NR4A1', 'EGR2', 'NFATC1'],
        'Exhausted: Apoptosis': ['FAS', 'CASP3', 'CASP8', 'BAX', 'BAK1', 'BIM'],
        'Effector: Cytotoxicity/Granzyme Pathway': ['GZMA', 'GZMB', 'GZMK', 'GZMH', 'PRF1', 'GNLY', 'FASLG'],
        'Effector: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'TBX21'],
        'Effector: T Cell Activation': ['CD69', 'IL2RA', 'CD38', 'CD40', 'HLA-DRA'],
        'Effector: Chemokine Signaling': ['CCL5', 'CCL4', 'CCL3', 'CXCR3', 'CCR5'],
        'Treg: Regulatory T Cell Suppression': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TIGIT', 'TNFRSF18', 'ENTPD1'],
        'Treg: TGF-beta Production': ['TGFB1', 'LRRC32', 'GARP', 'SMAD3'],
        'Naive: Lymphocyte Homing / Migration': ['SELL', 'CCR7', 'CXCR4', 'S1PR1', 'LEF1'],
        'Naive: TCF7 / Wnt Signaling': ['TCF7', 'LEF1', 'CTNNB1', 'MYC', 'APC'],
        'Naive: IL-7 Signaling (Survival)': ['IL7R', 'JAK1', 'JAK3', 'STAT5A', 'BCL2']
    }
}

fine_col = 'CellType_Fine'
cancer_col = 'Cancer_Type'

for ct in pathways_dict.keys():
    print(f"\n{'='*40}")
    print(f"Downstream Analysis for {ct}")
    print(f"{'='*40}")
    
    in_file = f"{out_dir}/08_{ct.replace(' ', '')}_adata_sctour_harmony.h5ad"
    if not os.path.exists(in_file):
        print(f"{in_file} not found, skipping.")
        continue
        
    adata = sc.read_h5ad(in_file)
    sub_dir = f"{out_dir}/{ct.replace(' ', '').lower()}"
    os.makedirs(sub_dir, exist_ok=True)
    
    # 1. Proportions
    count_df = adata.obs.groupby([cancer_col, fine_col]).size().unstack(fill_value=0)
    count_df = count_df.loc[:, count_df.sum(axis=0) > 0]
    prop_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
    count_df.to_csv(f"{sub_dir}/{ct}_Counts_by_Cancer.csv")
    prop_df.to_csv(f"{sub_dir}/{ct}_Proportions_by_Cancer.csv")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    prop_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='black')
    plt.title(f'Proportion of {ct} Subtypes Across Cancers')
    plt.xlabel('Cancer Type')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{sub_dir}/{ct}_Proportions_StackedBar.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. Markers
    sc.tl.rank_genes_groups(adata, groupby=fine_col, method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    marker_dict = {}
    for group in groups:
        top50_genes = result['names'][group][:50]
        marker_dict[group] = top50_genes
    marker_df = pd.DataFrame(marker_dict)
    marker_df.to_csv(f"{sub_dir}/{ct}_Subtypes_Top50_Markers.csv", index=False)
    
    # 3. Pathway Dotplot with P-value and OR
    from scipy.stats import mannwhitneyu
    pathways_of_interest = pathways_dict[ct]
    valid_pathways = {}
    for name, genes in pathways_of_interest.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if len(valid_genes) > 0:
            valid_pathways[name] = valid_genes
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=name)
            
    pathway_names = list(valid_pathways.keys())
    adata_sub = adata[adata.obs[fine_col] != 'Unknown'].copy()
    subtypes = adata_sub.obs[fine_col].unique()
    
    results = []
    for pathway in pathway_names:
        scores = adata_sub.obs[pathway].values
        global_mean = np.mean(scores)
        for subtype in subtypes:
            mask = (adata_sub.obs[fine_col] == subtype).values
            scores_in = scores[mask]
            scores_out = scores[~mask]
            
            if len(scores_in) > 0 and len(scores_out) > 0:
                stat, pval = mannwhitneyu(scores_in, scores_out, alternative='two-sided')
            else:
                pval = 1.0
                
            high_in = np.sum(scores_in > global_mean)
            low_in = len(scores_in) - high_in
            high_out = np.sum(scores_out > global_mean)
            low_out = len(scores_out) - high_out
            
            OR = ((high_in + 0.5) / (low_in + 0.5)) / ((high_out + 0.5) / (low_out + 0.5))
            log2_OR = np.log2(OR)
            nlog10_pval = -np.log10(pval + 1e-300)
            
            results.append({
                'Pathway': pathway,
                'Subtype': subtype,
                'P_value': pval,
                '-log10(P)': nlog10_pval,
                'OR': OR,
                'Log2_OR': log2_OR
            })
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{sub_dir}/{ct}_Pathways_Pval_OR.csv", index=False)
    
    max_size = 50
    df_res['-log10(P)_capped'] = df_res['-log10(P)'].clip(upper=max_size)
    df_res['Pathway'] = pd.Categorical(df_res['Pathway'], categories=pathway_names[::-1], ordered=True)
    
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df_res, x='Subtype', y='Pathway', 
        size='-log10(P)_capped', hue='Log2_OR', 
        sizes=(20, 600), palette='RdYlBu_r', edgecolor='gray'
    )
    plt.title(f'Pathway Enrichment (Size: P-value, Color: OR) - {ct}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('')
    plt.xlabel('Subtype')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax = plt.gca()
    x_limits = ax.get_xlim()
    ax.set_xlim(x_limits[0] - 0.5, x_limits[1] + 0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{sub_dir}/{ct}_Specific_Pathways_Dotplot.pdf", bbox_inches='tight')
    plt.close()

print("=== Step 5 Complete ===")
