import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Imports
cells.append(nbf.v4.new_markdown_cell("# 跨癌种单细胞整合分析 (CC-VAE)\n包含：质控、整合、聚类、大类注释、亚群细分注释、拟时序分析（scTour与Monocle2导出）"))
cells.append(nbf.v4.new_code_cell("""import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys
import warnings

# 导入CC-VAE模型
sys.path.append('.') # 确保model.py在当前或指定路径
from model import CrossCancerVAE
import sctour as sct

# 设置
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
warnings.filterwarnings('ignore')

# 创建输出目录
out_dir = "results_cross_cancer"
os.makedirs(out_dir, exist_ok=True)"""))

# Cell 2: Load function
cells.append(nbf.v4.new_markdown_cell("## 1. 数据加载与质控"))
cells.append(nbf.v4.new_code_cell("""def load_dataset_10x(data_dir, cancer_type):
    adatas = []
    matrix_files = glob.glob(os.path.join(data_dir, '*matrix.mtx.gz'))
    
    for mtx_path in matrix_files:
        # 处理不同的命名规则
        base_name = os.path.basename(mtx_path).replace('_matrix.mtx.gz', '').replace('matrix.mtx.gz', '')
        
        # 寻找对应的barcodes和features文件
        barcode_path = os.path.join(data_dir, f"{base_name}_barcodes.tsv.gz")
        if not os.path.exists(barcode_path):
            barcode_path = os.path.join(data_dir, f"{base_name}barcodes.tsv.gz")
            
        feature_path = os.path.join(data_dir, f"{base_name}_features.tsv.gz")
        if not os.path.exists(feature_path):
            feature_path = os.path.join(data_dir, f"{base_name}features.tsv.gz")
        if not os.path.exists(feature_path):
            feature_path = os.path.join(data_dir, f"{base_name}_genes.tsv.gz")
            
        if os.path.exists(barcode_path) and os.path.exists(feature_path):
            print(f"Loading sample: {base_name} ({cancer_type})")
            try:
                adata = sc.read_mtx(mtx_path).T
                barcodes = pd.read_csv(barcode_path, header=None, sep='\\t')
                adata.obs_names = barcodes[0].values
                features = pd.read_csv(feature_path, header=None, sep='\\t')
                gene_col = 1 if features.shape[1] > 1 else 0
                adata.var_names = features[gene_col].values
                adata.var_names_make_unique()
                
                adata.obs['Sample'] = base_name.strip('_')
                adata.obs['Cancer_Type'] = cancer_type
                adata.obs['Batch'] = f"{cancer_type}_{base_name.strip('_')}"
                
                # 质控
                sc.pp.filter_cells(adata, min_genes=200)
                sc.pp.filter_genes(adata, min_cells=3)
                adata.var['mt'] = adata.var_names.str.startswith('MT-')
                sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                adata = adata[adata.obs.pct_counts_mt < 20, :]
                
                adatas.append(adata)
            except Exception as e:
                print(f"Error loading {base_name}: {e}")
                
    if adatas:
        return ad.concat(adatas, join='inner', label='batch_index')
    return None

# 定义数据集路径
paths = {
    "ccRCC": "/mnt/data3/STrnaseq/script/SCbatch/GSE242299 Clear cell renal cell carcinoma",
    "Bladder": "/mnt/data3/STrnaseq/script/SCbatch/GSE277524 Bladder carcinoma",
    "HCC": "/mnt/data3/STrnaseq/script/SCbatch/GSE290925 Hepatocellular carcinoma",
    "Breast": "/mnt/data3/STrnaseq/script/SCbatch/GSE292824 Breast cancer"
}

adata_list = []
for ctype, p in paths.items():
    adata_list.append(load_dataset_10x(p, ctype))

# 合并四个癌种
adata = ad.concat(adata_list, join='inner')
adata.obs_names_make_unique()
print(adata)"""))

# Cell 3: Preprocessing
cells.append(nbf.v4.new_code_cell("""# 保存原始counts用于CC-VAE和差异分析
adata.layers["counts"] = adata.X.copy()

# 归一化与高变基因选择
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=False, layer='counts')
print(f"Post-HVG shape: {adata.shape}")"""))

# Cell 4: CC-VAE Integration
cells.append(nbf.v4.new_markdown_cell("## 2. CC-VAE 跨癌种整合"))
cells.append(nbf.v4.new_code_cell("""# 初始化并训练 CC-VAE
print("Initializing CC-VAE...")
cc_vae = CrossCancerVAE(adata, batch_key='Batch')

print("Training CC-VAE...")
cc_vae.train(max_epochs=100) # 实际运行建议100-200 epochs

# 提取降维特征
adata.obsm['X_ccvae'] = cc_vae.get_latent_representation()
print("Latent representation extracted successfully.")"""))

# Cell 5: Clustering
cells.append(nbf.v4.new_markdown_cell("## 3. 降维与聚类"))
cells.append(nbf.v4.new_code_cell("""# 构建邻接图与UMAP
sc.pp.neighbors(adata, use_rep='X_ccvae', n_neighbors=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden_ccvae')

# 保存全量数据的UMAP图
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata, color='Batch', show=False, ax=axs[1], legend_loc='none')
sc.pl.umap(adata, color='leiden_ccvae', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/1_All_Cells_UMAP.pdf", bbox_inches='tight')
plt.show()"""))

# Cell 6: DEG and Broad Annotation
cells.append(nbf.v4.new_markdown_cell("## 4. 大类细胞注释 (Cell Type Annotation)"))
cells.append(nbf.v4.new_code_cell("""# 寻找每个聚类的差异基因 (Marker Genes)
sc.tl.rank_genes_groups(adata, groupby='leiden_ccvae', method='wilcoxon')
markers_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(50)
markers_df.to_csv(f"{out_dir}/2_All_Cells_Cluster_Markers.csv")

# 经典大类标志基因字典
major_markers = {
    'T cells': ['CD3D', 'CD3E', 'CD2'],
    'Macrophages': ['CD68', 'CD163', 'C1QA', 'LYZ'],
    'Fibroblasts': ['COL1A1', 'COL3A1', 'DCN', 'ACTA2'],
    'Epithelial/Cancer': ['EPCAM', 'KRT8', 'KRT18'],
    'B cells': ['CD79A', 'MS4A1'],
    'Endothelial': ['PECAM1', 'VWF', 'ENG'],
    'NK cells': ['NKG7', 'GNLY']
}

# 绘制Dotplot
sc.pl.dotplot(adata, major_markers, groupby='leiden_ccvae', standard_scale='var', 
              save=f"_{out_dir}/3_All_Cells_Marker_Dotplot.pdf")

# 自动粗注释逻辑（基于得分），实际应用中请结合上方的Dotplot和CSV手动确认！
for cell_type, genes in major_markers.items():
    valid_genes = [g for g in genes if g in adata.var_names]
    if valid_genes:
        sc.tl.score_genes(adata, gene_list=valid_genes, score_name=f'{cell_type}_score')

score_cols = [f'{ct}_score' for ctype in major_markers.keys() if f'{ct}_score' in adata.obs.columns]
# 找到每个细胞得分最高的大类
adata.obs['CellType_Broad'] = adata.obs[score_cols].idxmax(axis=1).str.replace('_score', '')

# 为了聚类级别的稳定性，我们将每个Leiden聚类的大多数细胞的类型赋给该聚类
cluster_mapping = {}
for cluster in adata.obs['leiden_ccvae'].unique():
    dominant_type = adata.obs[adata.obs['leiden_ccvae'] == cluster]['CellType_Broad'].value_counts().index[0]
    cluster_mapping[cluster] = dominant_type

adata.obs['CellType'] = adata.obs['leiden_ccvae'].map(cluster_mapping)

sc.pl.umap(adata, color='CellType', save=f"_{out_dir}/4_All_Cells_CellType_UMAP.pdf")"""))

# Cell 7: Subclustering setup
cells.append(nbf.v4.new_markdown_cell("## 5. 细分亚群分析 (巨噬细胞、成纤维细胞、T细胞)\n分别提取并重新降维、聚类、注释和轨迹分析。"))
cells.append(nbf.v4.new_code_cell("""def analyze_subpopulation(adata_full, cell_type, fine_markers, out_prefix):
    print(f"\\n--- Analyzing {cell_type} ---")
    # 提取亚群
    sub_adata = adata_full[adata_full.obs['CellType'] == cell_type].copy()
    
    # 重新降维聚类 (使用原有的CC-VAE空间进行邻接图构建以保持批次校正效果)
    sc.pp.neighbors(sub_adata, use_rep='X_ccvae', n_neighbors=15)
    sc.tl.umap(sub_adata)
    sc.tl.leiden(sub_adata, resolution=0.6, key_added=f'leiden_{cell_type}')
    
    # 差异基因
    sc.tl.rank_genes_groups(sub_adata, groupby=f'leiden_{cell_type}', method='wilcoxon')
    sub_markers_df = pd.DataFrame(sub_adata.uns['rank_genes_groups']['names']).head(50)
    sub_markers_df.to_csv(f"{out_dir}/{out_prefix}_Cluster_Markers.csv")
    
    # 细分亚群打分注释
    for fine_type, genes in fine_markers.items():
        valid_genes = [g for g in genes if g in sub_adata.var_names]
        if valid_genes:
            sc.tl.score_genes(sub_adata, gene_list=valid_genes, score_name=f'{fine_type}_score')
            
    score_cols = [f'{ct}_score' for ct in fine_markers.keys() if f'{ct}_score' in sub_adata.obs.columns]
    if score_cols:
        sub_adata.obs[f'{cell_type}_Fine'] = sub_adata.obs[score_cols].idxmax(axis=1).str.replace('_score', '')
        # 聚类级别平滑
        mapping = {}
        for cluster in sub_adata.obs[f'leiden_{cell_type}'].unique():
            dom = sub_adata.obs[sub_adata.obs[f'leiden_{cell_type}'] == cluster][f'{cell_type}_Fine'].value_counts().index[0]
            mapping[cluster] = dom
        sub_adata.obs['CellType_Fine'] = sub_adata.obs[f'leiden_{cell_type}'].map(mapping)
    else:
        sub_adata.obs['CellType_Fine'] = sub_adata.obs[f'leiden_{cell_type}']
        
    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sc.pl.umap(sub_adata, color='Cancer_Type', show=False, ax=axs[0])
    sc.pl.umap(sub_adata, color=f'leiden_{cell_type}', show=False, ax=axs[1])
    sc.pl.umap(sub_adata, color='CellType_Fine', show=False, ax=axs[2])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{out_prefix}_UMAP.pdf", bbox_inches='tight')
    plt.show()
    
    # Marker Dotplot
    sc.pl.dotplot(sub_adata, fine_markers, groupby='CellType_Fine', standard_scale='var', 
                  save=f"_{out_dir}/{out_prefix}_Marker_Dotplot.pdf")
                  
    return sub_adata"""))

# Cell 8: Macrophage analysis
cells.append(nbf.v4.new_markdown_cell("### 5.1 巨噬细胞 (Macrophages) 细分与轨迹"))
cells.append(nbf.v4.new_code_cell("""macrophage_markers = {
    'M1-like (Pro-inflammatory)': ['CXCL9', 'CXCL10', 'IL1B', 'TNF'],
    'M2-like (Anti-inflammatory/TAM)': ['CD163', 'MRC1', 'MS4A4A', 'MAF'],
    'SPP1+ TAM (Angiogenic/Hypoxic)': ['SPP1', 'VEGFA', 'HIF1A'],
    'C1QC+ TAM (Phagocytic)': ['C1QC', 'C1QB', 'C1QA', 'APOE'],
    'Cycling Macrophages': ['MKI67', 'TOP2A']
}

adata_mac = analyze_subpopulation(adata, 'Macrophages', macrophage_markers, "5_Macrophage")

# scTour 拟时序分析
tnn_mac = sct.train.Trainer(adata_mac, loss_mode='mse')
tnn_mac.train()
adata_mac.obs['pseudotime'] = tnn_mac.get_time()
adata_mac.obsm['X_TVAE'], *_ = tnn_mac.get_latentsp(alpha_z=0.5, alpha_dz=0.5)
adata_mac.obsm['X_VF'] = tnn_mac.get_vector_field(adata_mac.obsm['X_TVAE'])

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_mac, color='pseudotime', show=False, ax=axs[0])
sct.vf.plot_vector_field(adata_mac, zs_key='X_TVAE', vf_key='X_VF', use_rep_neigh='X_TVAE', 
                         color='CellType_Fine', show=False, ax=axs[1], size=80, alpha=0.2)
plt.savefig(f"{out_dir}/6_Macrophage_scTour_Trajectory.pdf", bbox_inches='tight')
plt.show()"""))

# Cell 9: Fibroblast analysis
cells.append(nbf.v4.new_markdown_cell("### 5.2 成纤维细胞 (Fibroblasts) 细分与轨迹"))
cells.append(nbf.v4.new_code_cell("""fibroblast_markers = {
    'myCAF (Myofibroblastic)': ['ACTA2', 'TAGLN', 'MYL9', 'POSTN'],
    'iCAF (Inflammatory)': ['IL6', 'CXCL12', 'CCL2', 'CFD'],
    'apCAF (Antigen-presenting)': ['HLA-DRA', 'HLA-DRB1', 'CD74'],
    'Normal/Resting Fibroblast': ['PDGFRA', 'PI16']
}

adata_fib = analyze_subpopulation(adata, 'Fibroblasts', fibroblast_markers, "7_Fibroblast")

# scTour 拟时序分析
tnn_fib = sct.train.Trainer(adata_fib, loss_mode='mse')
tnn_fib.train()
adata_fib.obs['pseudotime'] = tnn_fib.get_time()
adata_fib.obsm['X_TVAE'], *_ = tnn_fib.get_latentsp(alpha_z=0.5, alpha_dz=0.5)
adata_fib.obsm['X_VF'] = tnn_fib.get_vector_field(adata_fib.obsm['X_TVAE'])

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_fib, color='pseudotime', show=False, ax=axs[0])
sct.vf.plot_vector_field(adata_fib, zs_key='X_TVAE', vf_key='X_VF', use_rep_neigh='X_TVAE', 
                         color='CellType_Fine', show=False, ax=axs[1], size=80, alpha=0.2)
plt.savefig(f"{out_dir}/8_Fibroblast_scTour_Trajectory.pdf", bbox_inches='tight')
plt.show()"""))

# Cell 10: T cell analysis
cells.append(nbf.v4.new_markdown_cell("### 5.3 T细胞 (T cells) 细分与轨迹"))
cells.append(nbf.v4.new_code_cell("""tcell_markers = {
    'CD8+ Exhausted T': ['LAG3', 'PDCD1', 'CTLA4', 'HAVCR2', 'TOX'],
    'CD8+ Effector/Memory T': ['GZMK', 'GZMB', 'PRF1', 'IFNG', 'CCL5'],
    'CD4+ Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'BATF'],
    'CD4+ Naive/Memory T': ['IL7R', 'TCF7', 'SELL', 'CCR7'],
    'Cycling T': ['MKI67', 'STMN1']
}

adata_t = analyze_subpopulation(adata, 'T cells', tcell_markers, "9_Tcell")

# scTour 拟时序分析
tnn_t = sct.train.Trainer(adata_t, loss_mode='mse')
tnn_t.train()
adata_t.obs['pseudotime'] = tnn_t.get_time()
adata_t.obsm['X_TVAE'], *_ = tnn_t.get_latentsp(alpha_z=0.5, alpha_dz=0.5)
adata_t.obsm['X_VF'] = tnn_t.get_vector_field(adata_t.obsm['X_TVAE'])

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_t, color='pseudotime', show=False, ax=axs[0])
sct.vf.plot_vector_field(adata_t, zs_key='X_TVAE', vf_key='X_VF', use_rep_neigh='X_TVAE', 
                         color='CellType_Fine', show=False, ax=axs[1], size=80, alpha=0.2)
plt.savefig(f"{out_dir}/10_Tcell_scTour_Trajectory.pdf", bbox_inches='tight')
plt.show()"""))

# Cell 11: Export to Monocle2 (optional)
cells.append(nbf.v4.new_markdown_cell("## 6. 导出数据给Monocle2（R）使用\n如果您还需要用Monocle2进行验证，可以运行以下代码将矩阵和元数据导出。"))
cells.append(nbf.v4.new_code_cell("""import scipy.io
def export_for_monocle(sub_adata, name):
    export_dir = f"{out_dir}/monocle_export_{name}"
    os.makedirs(export_dir, exist_ok=True)
    
    # 导出metadata
    sub_adata.obs.to_csv(os.path.join(export_dir, "metadata.csv"))
    # 导出raw counts matrix
    scipy.io.mmwrite(os.path.join(export_dir, "counts_matrix.mtx"), sub_adata.layers['counts'].T)
    # 导出基因名
    pd.DataFrame(sub_adata.var_names).to_csv(os.path.join(export_dir, "genes.csv"), index=False, header=False)
    print(f"Exported {name} to {export_dir}")

# export_for_monocle(adata_mac, "Macrophage")
# export_for_monocle(adata_fib, "Fibroblast")
# export_for_monocle(adata_t, "Tcell")
"""))

# Assign cells to notebook
nb['cells'] = cells

# Write notebook
with open('/mnt/data3/STrnaseq/script/SCbatch/RealData_CrossCancer_CCVAE.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
