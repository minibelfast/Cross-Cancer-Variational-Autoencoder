import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gseapy as gp

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

in_file = "results_real_cross_cancer/05_Macrophage_adata_fine.h5ad"
out_dir = "results_real_cross_cancer/macrophage"
sc.settings.figdir = out_dir

print("=== 1. Loading Macrophage data ===")
adata = sc.read_h5ad(in_file)

# 确保目标列存在
fine_col = 'CellType_Fine'
cancer_col = 'Cancer_Type'

# ==========================================
# Task 1: 比例分析与堆叠柱状图
# ==========================================
print("=== 2. Calculating proportions ===")
# 计算各癌种中不同巨噬细胞类型的细胞数
count_df = adata.obs.groupby([cancer_col, fine_col]).size().unstack(fill_value=0)
# 计算比例
prop_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

# 保存数据
count_df.to_csv(f"{out_dir}/Macrophage_Counts_by_Cancer.csv")
prop_df.to_csv(f"{out_dir}/Macrophage_Proportions_by_Cancer.csv")

# 绘制堆叠柱状图
fig, ax = plt.subplots(figsize=(8, 6))
prop_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='black')
plt.title('Proportion of Macrophage Subtypes Across Cancers')
plt.xlabel('Cancer Type')
plt.ylabel('Percentage (%)')
plt.legend(title='Macrophage Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{out_dir}/Macrophage_Proportions_StackedBar.pdf", bbox_inches='tight')
plt.close()

# ==========================================
# Task 3: 输出 Top50 标志性基因
# ==========================================
print("=== 3. Calculating Top50 Marker Genes ===")
# 我们已经在前面跑过了 marker 基因，但为了确保拿到最新的 fine-grained 分群 marker，这里重新算一次
sc.tl.rank_genes_groups(adata, groupby=fine_col, method='wilcoxon')

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

marker_dict = {}
for group in groups:
    # 提取每个亚群的 top 50 基因
    top50_genes = result['names'][group][:50]
    marker_dict[group] = top50_genes

marker_df = pd.DataFrame(marker_dict)
marker_df.to_csv(f"{out_dir}/Macrophage_Subtypes_Top50_Markers.csv", index=False)


# ==========================================
# Task 2: 分析主要激活通路 (基于每个细胞类型Top5通路打分)
# ==========================================
print("=== 4. Analyzing Specific Pathways ===")

# 基于最新文献 (例如 Cell, Nature 系列关于肿瘤相关巨噬细胞 TAM 的单细胞分型)
# 我们为这四种细胞分别设定其生物学背景下的 Top 5 代表性通路

pathways_of_interest = {
    # 1. M1-like (Pro-inflammatory)
    # 背景：经典活化，促炎，抗原呈递，分泌炎性细胞因子
    'M1: TNFa Signaling via NFkB': ['TNF', 'NFKB1', 'REL', 'RELA', 'CCL2', 'CXCL8', 'CXCL10', 'IL1B'],
    'M1: Inflammatory Response': ['IL1A', 'IL1B', 'IL6', 'CXCL8', 'PTGS2', 'CCL5', 'CXCL9'],
    'M1: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'IDO1', 'GBP1'],
    'M1: Antigen Processing and Presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'B2M', 'TAP1', 'TAP2', 'CD74'],
    'M1: IL6_JAK_STAT3 Signaling': ['IL6', 'JAK1', 'STAT3', 'SOCS3', 'PIM1', 'IL6R'],

    # 2. M2-like (Anti-inflammatory/TAM)
    # 背景：替代活化，免疫抑制，组织修复，响应 IL-4/IL-13/TGF-b
    'M2: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'SMAD2', 'SMAD3', 'TGFBI', 'THBS1'],
    'M2: IL4 and IL13 Signaling': ['IL4R', 'IL13RA1', 'STAT6', 'ARG1', 'CCL18', 'CD209', 'MRC1'],
    'M2: Tissue Remodeling/Wound Healing': ['MMP9', 'MMP14', 'TIMP1', 'FN1', 'COL1A1', 'VIM'],
    'M2: PI3K_AKT_MTOR Signaling': ['PIK3CA', 'AKT1', 'MTOR', 'PTEN', 'GSK3B'],
    'M2: Oxidative Phosphorylation': ['COX5B', 'ATP5F1A', 'NDUFA4', 'SDHA', 'UQCRB', 'CYCS'],

    # 3. SPP1+ TAM (Angiogenic/Hypoxic)
    # 背景：定位于肿瘤缺氧核心区，促血管生成，糖酵解代谢重编程，强促癌
    'SPP1: Hypoxia Response': ['HIF1A', 'EPAS1', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'ALDOA'],
    'SPP1: Angiogenesis': ['VEGFA', 'KDR', 'PECAM1', 'CD34', 'TEK', 'FLT1', 'ANGPT1', 'ANGPT2'],
    'SPP1: Glycolysis': ['HK2', 'PFKP', 'PKM', 'GAPDH', 'ENO1', 'LDHA', 'ALDOA', 'PGK1'],
    'SPP1: Epithelial Mesenchymal Transition (EMT)': ['VIM', 'SNAI1', 'SNAI2', 'TWIST1', 'CDH2', 'FN1', 'MMP9'],
    'SPP1: ECM Receptor Interaction': ['SPP1', 'ITGB1', 'ITGA5', 'CD44', 'LAMC1', 'COL4A1'],

    # 4. C1QC+ TAM (Phagocytic)
    # 背景：高表达补体系统，较强的吞噬能力（Efferocytosis），脂质代谢，突触修剪/细胞碎片清除
    'C1QC: Complement Activation': ['C1QA', 'C1QB', 'C1QC', 'C3', 'CFB', 'C4A', 'CR1'],
    'C1QC: Phagocytosis/Efferocytosis': ['CD36', 'MERTK', 'MARCO', 'CD68', 'FCGR3A', 'SIRPA', 'AXL'],
    'C1QC: Lipid Metabolism/Cholesterol Efflux': ['APOE', 'APOC1', 'ABCA1', 'ABCG1', 'LPL', 'TREM2'],
    'C1QC: Lysosome Pathway': ['CTSB', 'CTSD', 'CTSS', 'LAMP1', 'LAMP2', 'LGMN'],
    'C1QC: Reactive Oxygen Species Pathway': ['NCF1', 'NCF2', 'CYBA', 'CYBB', 'RAC1']
}

# Cycling Macrophages (附加：作为对照，主要就是细胞周期相关)
# 由于要求每种四种主群的top5，Cycling主要是增殖状态，我们将其合并或单独列出2条通路作为对照
pathways_of_interest.update({
    'Cycling: G2M Checkpoint': ['MKI67', 'TOP2A', 'CDK1', 'CCNB1', 'BIRC5', 'CENPF'],
    'Cycling: E2F Targets': ['PCNA', 'MCM2', 'MCM6', 'E2F1', 'TYMS', 'FEN1']
})

# 过滤掉不在 var_names 中的基因
valid_pathways = {}
for name, genes in pathways_of_interest.items():
    valid_genes = [g for g in genes if g in adata.var_names]
    if len(valid_genes) > 0:
        valid_pathways[name] = valid_genes

# 为每个细胞计算通路得分
for pathway_name, genes in valid_pathways.items():
    sc.tl.score_genes(adata, gene_list=genes, score_name=pathway_name)

# 构建一个 DataFrame 来汇总得分
pathway_names = list(valid_pathways.keys())
pathway_df = adata.obs[[fine_col] + pathway_names].copy()

# 计算均值作为富集分数
mean_scores = pathway_df.groupby(fine_col).mean()
mean_scores.to_csv(f"{out_dir}/Macrophage_Specific_Pathways_MeanScores.csv")

# 准备画图数据
mean_scores_melted = mean_scores.reset_index().melt(id_vars=fine_col, var_name='Pathway', value_name='Score')

# 为了图表美观，我们对Pathway进行排序，让它们按照我们定义的顺序排列
mean_scores_melted['Pathway'] = pd.Categorical(mean_scores_melted['Pathway'], categories=pathway_names, ordered=True)

# 绘制点图
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=mean_scores_melted, 
    x=fine_col, 
    y='Pathway', 
    size='Score', 
    hue='Score', 
    sizes=(50, 600), 
    palette='RdYlBu_r', # 用红蓝渐变色
    edgecolor='gray'
)

plt.title('Top 5 Specific Pathway Enrichment Scores Across Macrophage Subtypes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('')
plt.xlabel('Macrophage Subtype')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Enrichment Score")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/Macrophage_Specific_Pathways_Dotplot.pdf", bbox_inches='tight')
plt.close()

print("=== Finished successfully! ===")
