import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

in_file = "results_real_cross_cancer/06_Fibroblast_adata_fine.h5ad"
out_dir = "results_real_cross_cancer/fibroblast"
sc.settings.figdir = out_dir

print("=== 1. Loading Fibroblast data ===")
adata = sc.read_h5ad(in_file)

fine_col = 'CellType_Fine'
cancer_col = 'Cancer_Type'

# ==========================================
# Task 1: 比例分析与堆叠柱状图
# ==========================================
print("=== 2. Calculating proportions ===")
# 计算各癌种中不同成纤维细胞类型的细胞数
count_df = adata.obs.groupby([cancer_col, fine_col]).size().unstack(fill_value=0)
# 排除可能没有细胞的类别
count_df = count_df.loc[:, count_df.sum(axis=0) > 0]
# 计算比例
prop_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

# 保存数据
count_df.to_csv(f"{out_dir}/Fibroblast_Counts_by_Cancer.csv")
prop_df.to_csv(f"{out_dir}/Fibroblast_Proportions_by_Cancer.csv")

# 绘制堆叠柱状图
fig, ax = plt.subplots(figsize=(8, 6))
prop_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='black')
plt.title('Proportion of Fibroblast Subtypes Across Cancers')
plt.xlabel('Cancer Type')
plt.ylabel('Percentage (%)')
plt.legend(title='Fibroblast Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{out_dir}/Fibroblast_Proportions_StackedBar.pdf", bbox_inches='tight')
plt.close()

# ==========================================
# Task 3: 输出 Top50 标志性基因
# ==========================================
print("=== 3. Calculating Top50 Marker Genes ===")
sc.tl.rank_genes_groups(adata, groupby=fine_col, method='wilcoxon')

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names

marker_dict = {}
for group in groups:
    # 提取每个亚群的 top 50 基因
    top50_genes = result['names'][group][:50]
    marker_dict[group] = top50_genes

marker_df = pd.DataFrame(marker_dict)
marker_df.to_csv(f"{out_dir}/Fibroblast_Subtypes_Top50_Markers.csv", index=False)


# ==========================================
# Task 2: 分析主要激活通路 (基于每个细胞类型Top5通路打分)
# ==========================================
print("=== 4. Analyzing Specific Pathways ===")

# 基于肿瘤相关成纤维细胞(CAF)的经典文献(如 Elyada et al., Cancer Discovery; Biffi et al., Cancer Discovery)
# 我们为这三种主要的 CAF 亚群设定其生物学背景下的 Top 5 代表性通路

pathways_of_interest = {
    # 1. myCAF (Myofibroblastic CAF)
    # 背景：位于肿瘤实质附近，响应 TGF-beta，负责基质沉积、平滑肌收缩和物理屏障构建
    'myCAF: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'SMAD2', 'SMAD3', 'TGFBI', 'THBS1'],
    'myCAF: Extracellular Matrix (ECM) Organization': ['COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'LAMA1', 'VCAN'],
    'myCAF: Smooth Muscle Contraction': ['ACTA2', 'MYL9', 'TAGLN', 'TPM1', 'CNN1', 'CALD1'],
    'myCAF: Focal Adhesion': ['ITGB1', 'ITGA5', 'CAV1', 'PTK2', 'TLN1', 'VCL'],
    'myCAF: Wound Healing / Tissue Remodeling': ['MMP2', 'MMP14', 'TIMP1', 'TIMP2', 'POSTN', 'SPARC'],

    # 2. iCAF (Inflammatory CAF)
    # 背景：距离肿瘤实质较远，响应 IL-1 等促炎因子，分泌大量趋化因子和细胞因子，招募免疫抑制细胞
    'iCAF: IL6/JAK/STAT3 Signaling': ['IL6', 'JAK1', 'STAT3', 'SOCS3', 'PIM1', 'IL6R'],
    'iCAF: TNFa Signaling via NFkB': ['TNF', 'NFKB1', 'REL', 'RELA', 'CCL2', 'CXCL8', 'CXCL1', 'CXCL2'],
    'iCAF: Chemokine Signaling': ['CXCL12', 'CCL2', 'CCL7', 'CXCL14', 'CXCR4'],
    'iCAF: Complement Cascade': ['C3', 'CFB', 'C1S', 'C1R', 'C4A'],
    'iCAF: Inflammatory Response': ['IL1B', 'PTGS2', 'HAS1', 'LIF', 'CXCL8', 'IL1R1'],

    # 3. apCAF (Antigen-presenting CAF)
    # 背景：表达 MHC-II 类分子，具有抗原递呈能力，可能调节 CD4+ T 细胞功能（也有研究认为与特定免疫亚型相关）
    'apCAF: Antigen Processing and Presentation (MHC-II)': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'CD74'],
    'apCAF: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'IDO1', 'CIITA'],
    'apCAF: T Cell Receptor Signaling Pathway': ['CD4', 'CD3E', 'ZAP70', 'LCK', 'FYN'], # 尽管是CAF，但会高表达与T细胞相互作用的配体
    'apCAF: Th1 and Th2 Cell Differentiation': ['IL12A', 'IL12B', 'STAT4', 'TBX21'],
    'apCAF: Immune System Process': ['B2M', 'TAP1', 'TAP2', 'PSMB8', 'PSMB9']
}

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
mean_scores.to_csv(f"{out_dir}/Fibroblast_Specific_Pathways_MeanScores.csv")

# 准备画图数据
mean_scores_melted = mean_scores.reset_index().melt(id_vars=fine_col, var_name='Pathway', value_name='Score')

# 为了图表美观，我们对Pathway进行排序，让它们按照我们定义的顺序排列
mean_scores_melted['Pathway'] = pd.Categorical(mean_scores_melted['Pathway'], categories=pathway_names, ordered=True)

# 仅筛选确实属于我们三大类的（如果有 Normal/Resting，看需不需要展示，通常保留主要的）
target_subtypes = ['myCAF (Myofibroblastic)', 'iCAF (Inflammatory)', 'apCAF (Antigen-presenting)']
mean_scores_melted = mean_scores_melted[mean_scores_melted[fine_col].isin(target_subtypes)]

# 绘制点图
# 将图片调宽，避免右侧的点被遮挡，并调整边距
plt.figure(figsize=(14, 8))
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

plt.title('Top 5 Specific Pathway Enrichment Scores Across CAF Subtypes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('')
plt.xlabel('Fibroblast Subtype')
# 调整 legend 位置，避免和图形重叠，也可以通过扩大 xlim 来避免点被切边
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Enrichment Score")

# 适当扩大 x 轴的范围，防止左右边缘的点被遮挡一半
ax = plt.gca()
x_limits = ax.get_xlim()
ax.set_xlim(x_limits[0] - 0.5, x_limits[1] + 0.5)

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/Fibroblast_Specific_Pathways_Dotplot.pdf", bbox_inches='tight')
plt.close()

print("=== Finished successfully! ===")
