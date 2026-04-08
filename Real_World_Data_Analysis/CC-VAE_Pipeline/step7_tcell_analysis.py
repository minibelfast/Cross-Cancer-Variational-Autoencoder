import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

in_file = "results_real_cross_cancer/07_Tcell_adata_fine.h5ad"
out_dir = "results_real_cross_cancer/Tcell"
sc.settings.figdir = out_dir

print("=== 1. Loading T cell data ===")
adata = sc.read_h5ad(in_file)

fine_col = 'CellType_Fine'
cancer_col = 'Cancer_Type'

# ==========================================
# Task 1: 比例分析与堆叠柱状图
# ==========================================
print("=== 2. Calculating proportions ===")
# 筛选我们关注的四种 T 细胞亚群（排除 Cycling T 或是只针对主要的）
target_subtypes = ['CD4+ Naive/Memory T', 'CD4+ Treg', 'CD8+ Effector/Memory T', 'CD8+ Exhausted T']
# 我们保留所有类型做统计，但重点是这四类
count_df = adata.obs.groupby([cancer_col, fine_col]).size().unstack(fill_value=0)
# 排除可能没有细胞的类别
count_df = count_df.loc[:, count_df.sum(axis=0) > 0]
# 计算比例
prop_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

# 保存数据
count_df.to_csv(f"{out_dir}/Tcell_Counts_by_Cancer.csv")
prop_df.to_csv(f"{out_dir}/Tcell_Proportions_by_Cancer.csv")

# 绘制堆叠柱状图
fig, ax = plt.subplots(figsize=(8, 6))
prop_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='black')
plt.title('Proportion of T cell Subtypes Across Cancers')
plt.xlabel('Cancer Type')
plt.ylabel('Percentage (%)')
plt.legend(title='T cell Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{out_dir}/Tcell_Proportions_StackedBar.pdf", bbox_inches='tight')
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
marker_df.to_csv(f"{out_dir}/Tcell_Subtypes_Top50_Markers.csv", index=False)


# ==========================================
# Task 2: 分析主要激活通路 (基于每个细胞类型Top5通路打分)
# ==========================================
print("=== 4. Analyzing Specific Pathways ===")

# 基于 T 细胞耗竭、活化和分化的权威文献（如 Wherry et al., Zheng et al., Zhang et al. pan-cancer single-cell T cell landscape）
# 我们为这四种主要的 T 细胞亚群设定其生物学背景下的 Top 5 代表性通路

pathways_of_interest = {
    # 1. CD8+ Exhausted T
    # 背景：终末耗竭，高表达抑制性受体，受持续抗原刺激，TOX 驱动，丧失效应功能但可能保留一定趋化能力
    'Exhausted: T Cell Exhaustion/Coinhibition': ['PDCD1', 'HAVCR2', 'LAG3', 'CTLA4', 'TIGIT', 'TOX', 'ENTPD1'],
    'Exhausted: Chronic Antigen Stimulation': ['BATF', 'IRF4', 'NR4A1', 'EGR2', 'NFATC1'],
    'Exhausted: Apoptosis': ['FAS', 'CASP3', 'CASP8', 'BAX', 'BAK1', 'BIM'],
    'Exhausted: Hypoxia Response': ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1'],
    'Exhausted: TGF-beta Signaling': ['TGFB1', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4'],

    # 2. CD8+ Effector/Memory T
    # 背景：强大的细胞毒性杀伤功能，分泌穿孔素/颗粒酶，高 IFN-gamma 响应
    'Effector: Cytotoxicity/Granzyme Pathway': ['GZMA', 'GZMB', 'GZMK', 'GZMH', 'PRF1', 'GNLY', 'FASLG'],
    'Effector: Interferon Gamma Response': ['IFNG', 'STAT1', 'JAK2', 'IRF1', 'OAS1', 'TBX21'],
    'Effector: T Cell Activation': ['CD69', 'IL2RA', 'CD38', 'CD40', 'HLA-DRA'],
    'Effector: Chemokine Signaling': ['CCL5', 'CCL4', 'CCL3', 'CXCR3', 'CCR5'],
    'Effector: IL-2/STAT5 Signaling': ['IL2', 'IL2RB', 'IL2RG', 'STAT5A', 'STAT5B'],

    # 3. CD4+ Treg
    # 背景：免疫抑制，维持免疫耐受，FOXP3 驱动，高表达 IL2RA，消耗 IL-2
    'Treg: Regulatory T Cell Suppression': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TIGIT', 'TNFRSF18', 'ENTPD1'],
    'Treg: IL-2 Receptor Signaling': ['IL2RA', 'IL2RB', 'IL2RG', 'STAT5A', 'STAT5B', 'JAK1', 'JAK3'],
    'Treg: TGF-beta Production': ['TGFB1', 'LRRC32', 'GARP', 'SMAD3'],
    'Treg: TNFa Signaling via NFkB': ['TNF', 'NFKB1', 'REL', 'RELA', 'TRAF2', 'BIRC3'],
    'Treg: Oxidative Phosphorylation': ['COX5B', 'ATP5F1A', 'NDUFA4', 'SDHA', 'UQCRB'], # Treg 偏好 OXPHOS 而非糖酵解

    # 4. CD4+ Naive/Memory T
    # 背景：未激活或中央记忆状态，高表达淋巴结归巢受体，静息状态，细胞生存和稳态维持
    'Naive: Lymphocyte Homing / Migration': ['SELL', 'CCR7', 'CXCR4', 'S1PR1', 'LEF1'],
    'Naive: TCF7 / Wnt Signaling': ['TCF7', 'LEF1', 'CTNNB1', 'MYC', 'APC'],
    'Naive: IL-7 Signaling (Survival)': ['IL7R', 'JAK1', 'JAK3', 'STAT5A', 'BCL2'],
    'Naive: T Cell Receptor Signaling Pathway': ['CD3D', 'CD3E', 'CD3G', 'ZAP70', 'LCK', 'LAT'],
    'Naive: Notch Signaling': ['NOTCH1', 'NOTCH2', 'HES1', 'HEY1', 'JAG1']
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
mean_scores.to_csv(f"{out_dir}/Tcell_Specific_Pathways_MeanScores.csv")

# 准备画图数据
mean_scores_melted = mean_scores.reset_index().melt(id_vars=fine_col, var_name='Pathway', value_name='Score')

# 为了图表美观，我们对Pathway进行排序，让它们按照我们定义的顺序排列
mean_scores_melted['Pathway'] = pd.Categorical(mean_scores_melted['Pathway'], categories=pathway_names, ordered=True)

# 仅筛选确实属于我们四大类的（排除可能存在的 Cycling 等小众群）
mean_scores_melted = mean_scores_melted[mean_scores_melted[fine_col].isin(target_subtypes)]

# 绘制点图，设置足够宽的 figsize，并调整 xlim 防止点被遮挡
plt.figure(figsize=(16, 8))
sns.scatterplot(
    data=mean_scores_melted, 
    x=fine_col, 
    y='Pathway', 
    size='Score', 
    hue='Score', 
    sizes=(50, 700), 
    palette='RdYlBu_r', # 用红蓝渐变色
    edgecolor='gray'
)

plt.title('Top 5 Specific Pathway Enrichment Scores Across T Cell Subtypes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('')
plt.xlabel('T cell Subtype')

# 调整 legend 位置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Enrichment Score")

# 适当扩大 x 轴的范围，防止左右边缘的巨大气泡被切边或遮挡
ax = plt.gca()
x_limits = ax.get_xlim()
ax.set_xlim(x_limits[0] - 0.5, x_limits[1] + 0.5)

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/Tcell_Specific_Pathways_Dotplot.pdf", bbox_inches='tight')
plt.close()

print("=== Finished successfully! ===")
