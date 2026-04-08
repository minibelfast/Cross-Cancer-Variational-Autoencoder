import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 设置绘图和日志
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
out_dir = "results_real_cross_cancer_harmony"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

def load_dataset_10x(data_dir, cancer_type):
    adatas = []
    matrix_files = glob.glob(os.path.join(data_dir, '*matrix.mtx.gz'))
    
    for mtx_path in matrix_files:
        base_name = os.path.basename(mtx_path).replace('matrix.mtx.gz', '').replace('_matrix.mtx.gz', '')
        if base_name.endswith('_'):
            base_name_clean = base_name[:-1]
        else:
            base_name_clean = base_name
        
        barcode_path = os.path.join(data_dir, f"{base_name_clean}_barcodes.tsv.gz")
        if not os.path.exists(barcode_path):
            barcode_path = os.path.join(data_dir, f"{base_name_clean}barcodes.tsv.gz")
            
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
                
                # 质控
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

print("\n=== 合并四个数据集 ===")
adata = ad.concat(adata_list, join='inner')
adata.obs_names_make_unique()
print("Merged adata shape:", adata.shape)

print("=== 归一化与HVG筛选 ===")
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=False, layer='counts')

print("=== PCA降维 ===")
sc.tl.pca(adata, svd_solver='arpack')

print("\n=== Harmony 整合 ===")
import harmonypy as hm
import scanpy.external as sce

# 使用scanpy的external api直接运行harmony
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
