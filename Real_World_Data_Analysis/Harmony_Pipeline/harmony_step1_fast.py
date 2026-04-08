import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scanpy.external as sce

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer_harmony"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

print("=== 加载已合并的数据集 ===")
# 使用之前CC-VAE步骤中保存的adata，跳过漫长的读取过程
adata = sc.read_h5ad("results_real_cross_cancer/integrated_adata.h5ad")
print("Adata shape:", adata.shape)

# 我们清除CC-VAE相关的列和obsm
for col in ['leiden_ccvae']:
    if col in adata.obs:
        del adata.obs[col]
if 'X_ccvae' in adata.obsm:
    del adata.obsm['X_ccvae']

print("=== PCA降维 ===")
# adata.X 已经是 log1p 后的数据，并且已经计算了 highly_variable_genes
sc.tl.pca(adata, svd_solver='arpack')

print("\n=== Harmony 整合 ===")
sce.pp.harmony_integrate(adata, 'Batch')

print("=== 提取潜变量并进行降维聚类 ===")
sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden_harmony')

print("=== 保存整合后的 UMAP 可视化 ===")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata, color='Batch', show=False, ax=axs[1], legend_loc='none')
sc.pl.umap(adata, color='leiden_harmony', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/01_All_Cells_UMAP_Harmony.pdf", bbox_inches='tight')

print("=== 计算并保存 Marker 基因 ===")
sc.tl.rank_genes_groups(adata, groupby='leiden_harmony', method='wilcoxon')
markers_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
markers_df.to_csv(f"{out_dir}/02_All_Cells_Cluster_Markers_Harmony.csv", index=False)

print("=== 保存完整 AnnData 数据对象 ===")
adata.write(f"{out_dir}/integrated_adata_harmony.h5ad")
print("=== Step 1 完全结束 ===")
