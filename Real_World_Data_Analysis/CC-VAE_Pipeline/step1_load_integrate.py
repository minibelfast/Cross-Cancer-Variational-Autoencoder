import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# 导入CC-VAE模型
sys.path.append('.') 
from model import CrossCancerVAE

# 设置绘图和日志
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
out_dir = "results_real_cross_cancer"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

def load_dataset_10x(data_dir, cancer_type):
    adatas = []
    matrix_files = glob.glob(os.path.join(data_dir, '*matrix.mtx.gz'))
    
    for mtx_path in matrix_files:
        # 处理不同数据集命名后缀的不规则问题
        base_name = os.path.basename(mtx_path).replace('matrix.mtx.gz', '').replace('_matrix.mtx.gz', '')
        if base_name.endswith('_'):
            base_name_clean = base_name[:-1]
        else:
            base_name_clean = base_name
        
        # 寻找对应的barcodes文件
        barcode_path = os.path.join(data_dir, f"{base_name_clean}_barcodes.tsv.gz")
        if not os.path.exists(barcode_path):
            barcode_path = os.path.join(data_dir, f"{base_name_clean}barcodes.tsv.gz")
            
        # 寻找对应的features文件
        feature_path = os.path.join(data_dir, f"{base_name_clean}_features.tsv.gz")
        if not os.path.exists(feature_path):
            feature_path = os.path.join(data_dir, f"{base_name_clean}features.tsv.gz")
        if not os.path.exists(feature_path):
            feature_path = os.path.join(data_dir, f"{base_name_clean}_genes.tsv.gz")
        if not os.path.exists(feature_path):
            feature_path = os.path.join(data_dir, f"{base_name_clean}genes.tsv.gz")
            
        if os.path.exists(barcode_path) and os.path.exists(feature_path):
            print(f"Loading sample: {base_name_clean} ({cancer_type})")
            try:
                adata = sc.read_mtx(mtx_path).T
                barcodes = pd.read_csv(barcode_path, header=None, sep='\t')
                adata.obs_names = barcodes[0].values
                features = pd.read_csv(feature_path, header=None, sep='\t')
                gene_col = 1 if features.shape[1] > 1 else 0
                adata.var_names = features[gene_col].values
                adata.var_names_make_unique()
                
                adata.obs['Sample'] = base_name_clean
                adata.obs['Cancer_Type'] = cancer_type
                adata.obs['Batch'] = f"{cancer_type}_{base_name_clean}"
                
                # 质控：过滤基因和细胞，控制线粒体比例
                sc.pp.filter_cells(adata, min_genes=200)
                sc.pp.filter_genes(adata, min_cells=3)
                adata.var['mt'] = adata.var_names.str.startswith('MT-')
                sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                adata = adata[adata.obs.pct_counts_mt < 20, :]
                
                adatas.append(adata)
            except Exception as e:
                print(f"Error loading {base_name_clean}: {e}")
        else:
            print(f"Missing barcodes or features for {base_name_clean}")
                
    if adatas:
        return ad.concat(adatas, join='inner')
    return None

# 定义数据集路径
paths = {
    "ccRCC": "/mnt/data3/STrnaseq/script/SCbatch/GSE242299 Clear cell renal cell carcinoma",
    "Bladder": "/mnt/data3/STrnaseq/script/SCbatch/GSE277524 Bladder carcinoma",
    "HCC": "/mnt/data3/STrnaseq/script/SCbatch/GSE290925 Hepatocellular carcinoma",
    "Breast": "/mnt/data3/STrnaseq/script/SCbatch/GSE292824 Breast cancer"
}

print("=== 开始加载数据 ===")
adata_list = []
for ctype, p in paths.items():
    print(f"\nProcessing {ctype} from {p}...")
    ad_tmp = load_dataset_10x(p, ctype)
    if ad_tmp is not None:
        adata_list.append(ad_tmp)

# 合并四个癌种
print("\n=== 合并四个数据集 ===")
adata = ad.concat(adata_list, join='inner')
adata.obs_names_make_unique()
print("Merged adata shape:", adata.shape)

# 预处理：保留原始counts，进行归一化和高变基因筛选
print("=== 归一化与HVG筛选 ===")
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=False, layer='counts')
print(f"Post-HVG shape: {adata.shape}")

# CC-VAE 整合
print("\n=== 初始化并训练 CC-VAE ===")
cc_vae = CrossCancerVAE(adata, batch_key='Batch')
cc_vae.train(max_epochs=50) 

print("=== 提取潜变量并进行降维聚类 ===")
adata.obsm['X_ccvae'] = cc_vae.get_latent_representation()

sc.pp.neighbors(adata, use_rep='X_ccvae', n_neighbors=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden_ccvae')

print("=== 保存整合后的 UMAP 可视化 ===")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata, color='Batch', show=False, ax=axs[1], legend_loc='none')
sc.pl.umap(adata, color='leiden_ccvae', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/01_All_Cells_UMAP.pdf", bbox_inches='tight')

print("=== 计算并保存 Marker 基因 ===")
sc.tl.rank_genes_groups(adata, groupby='leiden_ccvae', method='wilcoxon')
markers_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
markers_df.to_csv(f"{out_dir}/02_All_Cells_Cluster_Markers.csv", index=False)

print("=== 保存完整 AnnData 数据对象 ===")
adata.write(f"{out_dir}/integrated_adata.h5ad")
print("=== Step 1 完全结束 ===")
