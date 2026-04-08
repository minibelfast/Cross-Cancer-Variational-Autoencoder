import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

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
    
    # 1. 比例
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
    
    # 2. Top50 Markers
    sc.tl.rank_genes_groups(adata, groupby=fine_col, method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    marker_dict = {}
    for group in groups:
        top50_genes = result['names'][group][:50]
        marker_dict[group] = top50_genes
    marker_df = pd.DataFrame(marker_dict)
    marker_df.to_csv(f"{sub_dir}/{ct}_Subtypes_Top50_Markers.csv", index=False)
    
    # 3. Pathways Dotplot
    pathways_of_interest = pathways_dict[ct]
    valid_pathways = {}
    for name, genes in pathways_of_interest.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if len(valid_genes) > 0:
            valid_pathways[name] = valid_genes
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=name)
            
    pathway_names = list(valid_pathways.keys())
    pathway_df = adata.obs[[fine_col] + pathway_names].copy()
    mean_scores = pathway_df.groupby(fine_col).mean()
    mean_scores.to_csv(f"{sub_dir}/{ct}_Specific_Pathways_MeanScores.csv")
    
    mean_scores_melted = mean_scores.reset_index().melt(id_vars=fine_col, var_name='Pathway', value_name='Score')
    mean_scores_melted['Pathway'] = pd.Categorical(mean_scores_melted['Pathway'], categories=pathway_names, ordered=True)
    
    mean_scores_melted = mean_scores_melted[mean_scores_melted[fine_col] != 'Unknown']
    
    plt.figure(figsize=(16, 8))
    sns.scatterplot(
        data=mean_scores_melted, 
        x=fine_col, 
        y='Pathway', 
        size='Score', 
        hue='Score', 
        sizes=(50, 700), 
        palette='RdYlBu_r', 
        edgecolor='gray'
    )
    plt.title(f'Top 5 Specific Pathway Enrichment Scores Across {ct} Subtypes')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('')
    plt.xlabel('Subtype')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Enrichment Score")
    
    ax = plt.gca()
    x_limits = ax.get_xlim()
    ax.set_xlim(x_limits[0] - 0.5, x_limits[1] + 0.5)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{sub_dir}/{ct}_Specific_Pathways_Dotplot.pdf", bbox_inches='tight')
    plt.close()
    
print("=== Step 5 Complete ===")
