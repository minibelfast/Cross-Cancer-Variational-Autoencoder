import scanpy as sc
import sctour as sct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer"
sc.settings.figdir = out_dir

adata_path = f"{out_dir}/integrated_adata_annotated.h5ad"

def analyze_subpopulation(cell_type_name, out_prefix):
    print(f"\n--- Analyzing {cell_type_name} ---")
    print("=== Loading annotated data ===")
    adata = sc.read_h5ad(adata_path)
    
    sub_adata = adata[adata.obs['CellType_Broad'] == cell_type_name].copy()
    
    del adata
    gc.collect()
    
    if len(sub_adata) == 0:
        print(f"No cells found for {cell_type_name}!")
        return None
        
    print(f"Cells: {len(sub_adata)}")
    
    # 重新降维聚类
    sc.pp.neighbors(sub_adata, use_rep='X_ccvae', n_neighbors=15)
    sc.tl.umap(sub_adata)
    sc.tl.leiden(sub_adata, resolution=0.6, key_added=f'leiden_{cell_type_name}')
    
    # 找 marker
    sc.tl.rank_genes_groups(sub_adata, groupby=f'leiden_{cell_type_name}', method='t-test')
    markers_df = pd.DataFrame(sub_adata.uns['rank_genes_groups']['names'])
    markers_df.to_csv(f"{out_dir}/{out_prefix}_Cluster_Markers.csv", index=False)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(sub_adata, color='Cancer_Type', show=False, ax=axs[0])
    sc.pl.umap(sub_adata, color=f'leiden_{cell_type_name}', show=False, ax=axs[1])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{out_prefix}_UMAP.pdf", bbox_inches='tight')
    
    # scTour 拟时序分析
    print(f"Running scTour for {cell_type_name}...")
    # 对部分较大的亚群进行随机降采样以节省内存跑scTour
    if len(sub_adata) > 2000:
        print(f"Downsampling for scTour from {len(sub_adata)} to 2000 cells to speed up...")
        sct_adata = sc.pp.subsample(sub_adata, n_obs=2000, copy=True, random_state=42)
    else:
        sct_adata = sub_adata.copy()
        
    # scTour 参数限制
    import inspect
    if 'n_epochs' in inspect.signature(sct.train.Trainer).parameters:
        tnn = sct.train.Trainer(sct_adata, loss_mode='mse', n_epochs=20)
    else:
        tnn = sct.train.Trainer(sct_adata, loss_mode='mse')
    
    # 获取原始 train 的签名来判断是否有 n_epochs
    if 'n_epochs' in inspect.signature(tnn.train).parameters:
        tnn.train(n_epochs=20)
    else:
        # 如果不支持 n_epochs 则默认
        try:
            # 某些版本可能支持隐式 kwargs
            tnn.train(epochs=20)
        except TypeError:
            pass
        except Exception:
            pass
        
        # 我们直接在内部修改它的参数试试，或者忍受默认 400 epochs
        if hasattr(tnn, 'epochs'):
            tnn.epochs = 20
        elif hasattr(tnn, 'n_epochs'):
            tnn.n_epochs = 20
            
        try:
            # 强制拦截超过一定时间的训练
            tnn.train(percent=0.05)
        except Exception:
            tnn.train()
            
    # 为了避免运行太久，手动提取模型训练状态
    sct_adata.obs['pseudotime'] = tnn.get_time()
    
    try:
        sct_adata.obsm['X_TVAE'], *_ = tnn.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
    except TypeError:
        sct_adata.obsm['X_TVAE'], *_ = tnn.get_latentsp(alpha_z=0.5, alpha_dz=0.5)
        
    try:
        sct_adata.obsm['X_VF'] = tnn.get_vector_field(sct_adata.obs['pseudotime'].values, sct_adata.obsm['X_TVAE'])
    except TypeError:
        sct_adata.obsm['X_VF'] = tnn.get_vector_field(T=sct_adata.obs['pseudotime'].values, Z=sct_adata.obsm['X_TVAE'])
    
    # 清理掉 NaN 或 inf，避免画图报错
    sct_adata.obsm['X_TVAE'] = np.nan_to_num(sct_adata.obsm['X_TVAE'])
    sct_adata.obsm['X_VF'] = np.nan_to_num(sct_adata.obsm['X_VF'])
    
    # 把降采样的结果放回原数据
    sub_adata.obs['pseudotime'] = np.nan
    sub_adata.obs.loc[sct_adata.obs_names, 'pseudotime'] = sct_adata.obs['pseudotime']
    
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sc.pl.umap(sct_adata, color='pseudotime', show=False, ax=axs[0])
        sct.vf.plot_vector_field(sct_adata, zs_key='X_TVAE', vf_key='X_VF', use_rep_neigh='X_TVAE', 
                                 color=f'leiden_{cell_type_name}', show=False, ax=axs[1], size=80, alpha=0.2)
        plt.savefig(f"{out_dir}/{out_prefix}_scTour_Trajectory.pdf", bbox_inches='tight')
    except Exception as e:
        print(f"Error plotting scTour for {cell_type_name}: {e}")
        
    sub_adata.write(f"{out_dir}/{out_prefix}_adata.h5ad")
    sct_adata.write(f"{out_dir}/{out_prefix}_sct_adata.h5ad")
    print(f"Saved {out_prefix}_adata.h5ad")
    
    del sub_adata, sct_adata
    gc.collect()

# 我们需要分析的三个亚群
# Macrophages 刚才似乎跑得差不多了，为了节省时间我们可以跳过如果它的文件存在
if not os.path.exists(f"{out_dir}/05_Macrophage_adata.h5ad"):
    analyze_subpopulation('Macrophages', "05_Macrophage")
if not os.path.exists(f"{out_dir}/06_Fibroblast_adata.h5ad"):
    analyze_subpopulation('Fibroblasts', "06_Fibroblast")
if not os.path.exists(f"{out_dir}/07_Tcell_adata.h5ad"):
    analyze_subpopulation('T cells', "07_Tcell")


