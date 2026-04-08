import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer_harmony"

pathways_dict = {
    'Macrophages': {
        'M1: TNFa Signaling via NFkB': ['TNF', 'NFKB1', 'REL', 'RELA', 'CCL2', 'CXCL8', 'CXCL10', 'IL1B'],
        'M1: Inflammatory Response': ['IL1A', 'IL1B', 'IL6', 'CXCL8', 'PTGS2', 'CCL5', 'CXCL9'],
        'M1: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'IDO1', 'GBP1'],
        'M1: Antigen Processing and Presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'B2M', 'TAP1', 'TAP2', 'CD74'],
        'M1: IL6_JAK_STAT3 Signaling': ['IL6', 'JAK1', 'STAT3', 'SOCS3', 'PIM1', 'IL6R'],
        'M2: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'SMAD2', 'SMAD3', 'TGFBI', 'THBS1'],
        'M2: IL4 and IL13 Signaling': ['IL4R', 'IL13RA1', 'STAT6', 'ARG1', 'CCL18', 'CD209', 'MRC1'],
        'M2: Tissue Remodeling/Wound Healing': ['MMP9', 'MMP14', 'TIMP1', 'FN1', 'COL1A1', 'VIM'],
        'M2: PI3K_AKT_MTOR Signaling': ['PIK3CA', 'AKT1', 'MTOR', 'PTEN', 'GSK3B'],
        'M2: Oxidative Phosphorylation': ['COX5B', 'ATP5F1A', 'NDUFA4', 'SDHA', 'UQCRB', 'CYCS'],
        'SPP1: Hypoxia Response': ['HIF1A', 'EPAS1', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'ALDOA'],
        'SPP1: Angiogenesis': ['VEGFA', 'KDR', 'PECAM1', 'CD34', 'TEK', 'FLT1', 'ANGPT1', 'ANGPT2'],
        'SPP1: Glycolysis': ['HK2', 'PFKP', 'PKM', 'GAPDH', 'ENO1', 'LDHA', 'ALDOA', 'PGK1'],
        'SPP1: Epithelial Mesenchymal Transition': ['VIM', 'SNAI1', 'SNAI2', 'TWIST1', 'CDH2', 'FN1', 'MMP9'],
        'SPP1: ECM Receptor Interaction': ['SPP1', 'ITGB1', 'ITGA5', 'CD44', 'LAMC1', 'COL4A1'],
        'C1QC: Complement Activation': ['C1QA', 'C1QB', 'C1QC', 'C3', 'CFB', 'C4A', 'CR1'],
        'C1QC: Phagocytosis/Efferocytosis': ['CD36', 'MERTK', 'MARCO', 'CD68', 'FCGR3A', 'SIRPA', 'AXL'],
        'C1QC: Lipid Metabolism': ['APOE', 'APOC1', 'ABCA1', 'ABCG1', 'LPL', 'TREM2'],
        'C1QC: Lysosome Pathway': ['CTSB', 'CTSD', 'CTSS', 'LAMP1', 'LAMP2', 'LGMN'],
        'C1QC: Reactive Oxygen Species': ['NCF1', 'NCF2', 'CYBA', 'CYBB', 'RAC1']
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
        'iCAF: Response to Interferon Gamma': ['IFNG', 'STAT1', 'IRF1', 'OAS1', 'ISG15', 'CXCL10'],
        'apCAF: MHC Class II Antigen Presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'CD74', 'B2M'],
        'apCAF: T Cell Activation / Costimulation': ['CD80', 'CD86', 'CD40', 'ICOS', 'TNFSF4'],
        'apCAF: Phagosome/Lysosome': ['CTSS', 'CTSB', 'LAMP1', 'LAMP2', 'LGMN'],
        'apCAF: Interferon Gamma Response': ['STAT1', 'IRF1', 'CIITA', 'RFX5'],
        'apCAF: Immune Response Regulation': ['PTPRC', 'HAVCR2', 'LGALS9', 'CD274']
    },
    'T cells': {
        'Exhausted: T Cell Exhaustion/Coinhibition': ['PDCD1', 'HAVCR2', 'LAG3', 'CTLA4', 'TIGIT', 'TOX', 'ENTPD1'],
        'Exhausted: Chronic Antigen Stimulation': ['BATF', 'IRF4', 'NR4A1', 'EGR2', 'NFATC1'],
        'Exhausted: Apoptosis': ['FAS', 'CASP3', 'CASP8', 'BAX', 'BAK1', 'BIM'],
        'Exhausted: Hypoxia Response': ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1'],
        'Exhausted: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4'],
        'Effector: Cytotoxicity/Granzyme Pathway': ['GZMA', 'GZMB', 'GZMK', 'GZMH', 'PRF1', 'GNLY', 'FASLG'],
        'Effector: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'TBX21'],
        'Effector: T Cell Activation': ['CD69', 'IL2RA', 'CD38', 'CD40', 'HLA-DRA'],
        'Effector: Chemokine Signaling': ['CCL5', 'CCL4', 'CCL3', 'CXCR3', 'CCR5'],
        'Effector: IL-2/STAT5 Signaling': ['IL2', 'IL2RB', 'IL2RG', 'STAT5A', 'STAT5B'],
        'Treg: Regulatory T Cell Suppression': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TIGIT', 'TNFRSF18', 'ENTPD1'],
        'Treg: IL-2 Receptor Signaling': ['IL2RA', 'IL2RB', 'IL2RG', 'STAT5A', 'STAT5B', 'JAK1', 'JAK3'],
        'Treg: TGF-beta Production': ['TGFB1', 'LRRC32', 'GARP', 'SMAD3'],
        'Treg: TNFa Signaling via NFkB': ['TNF', 'NFKB1', 'REL', 'RELA', 'TRAF2', 'BIRC3'],
        'Treg: Oxidative Phosphorylation': ['COX5B', 'ATP5F1A', 'NDUFA4', 'SDHA', 'UQCRB'],
        'Naive: Lymphocyte Homing / Migration': ['SELL', 'CCR7', 'CXCR4', 'S1PR1', 'LEF1'],
        'Naive: TCF7 / Wnt Signaling': ['TCF7', 'LEF1', 'CTNNB1', 'MYC', 'APC'],
        'Naive: IL-7 Signaling (Survival)': ['IL7R', 'JAK1', 'JAK3', 'STAT5A', 'BCL2'],
        'Naive: T Cell Receptor Signaling Pathway': ['CD3D', 'CD3E', 'CD3G', 'ZAP70', 'LCK', 'LAT'],
        'Naive: Notch Signaling': ['NOTCH1', 'NOTCH2', 'HES1', 'HEY1', 'JAG1']
    }
}

fine_col = 'CellType_Fine'

for ct in pathways_dict.keys():
    print(f"\n{'='*40}")
    print(f"Calculating P-value and OR for {ct}")
    print(f"{'='*40}")
    
    in_file = f"{out_dir}/08_{ct.replace(' ', '')}_adata_sctour_harmony.h5ad"
    if not os.path.exists(in_file):
        print(f"{in_file} not found, skipping.")
        continue
        
    adata = sc.read_h5ad(in_file)
    sub_dir = f"{out_dir}/{ct.replace(' ', '').lower()}"
    os.makedirs(sub_dir, exist_ok=True)
    
    pathways_of_interest = pathways_dict[ct]
    valid_pathways = {}
    
    # Calculate pathway scores first
    for name, genes in pathways_of_interest.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if len(valid_genes) > 0:
            valid_pathways[name] = valid_genes
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=name)
            
    pathway_names = list(valid_pathways.keys())
    
    # Exclude Unknown
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
            
            # P-value using Mann-Whitney U test
            if len(scores_in) > 0 and len(scores_out) > 0:
                stat, pval = mannwhitneyu(scores_in, scores_out, alternative='two-sided')
            else:
                pval = 1.0
                
            # Odds Ratio (OR) Approximation
            # For continuous scores, we can calculate the difference in means or Cliff's Delta.
            # To simulate OR, let's use the ratio of cells above global mean
            high_in = np.sum(scores_in > global_mean)
            low_in = len(scores_in) - high_in
            high_out = np.sum(scores_out > global_mean)
            low_out = len(scores_out) - high_out
            
            # Add pseudocount
            OR = ((high_in + 0.5) / (low_in + 0.5)) / ((high_out + 0.5) / (low_out + 0.5))
            
            # Alternatively, let's use standard effect size (Cohen's d) mapped to OR if user prefers, 
            # but the above binary thresholding gives a literal Odds Ratio.
            # Let's use Log2 OR for symmetric color scaling
            log2_OR = np.log2(OR)
            
            # -log10 pval
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
    
    # Cap -log10(P) for size to avoid giant dots
    max_size = 50  # adjust capping if needed
    df_res['-log10(P)_capped'] = df_res['-log10(P)'].clip(upper=max_size)
    
    df_res['Pathway'] = pd.Categorical(df_res['Pathway'], categories=pathway_names[::-1], ordered=True)
    
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=df_res, 
        x='Subtype', 
        y='Pathway', 
        size='-log10(P)_capped', 
        hue='Log2_OR', 
        sizes=(20, 600), 
        palette='RdYlBu_r', 
        edgecolor='gray'
    )
    
    plt.title(f'Pathway Enrichment (Size: P-value, Color: OR) - {ct}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('')
    plt.xlabel('Subtype')
    
    # Customize legend
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    ax = plt.gca()
    x_limits = ax.get_xlim()
    ax.set_xlim(x_limits[0] - 0.5, x_limits[1] + 0.5)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{sub_dir}/{ct}_Specific_Pathways_Dotplot.pdf", bbox_inches='tight')
    plt.close()

print("=== Re-plot Complete ===")
