import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sctour as sct
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_real_cross_cancer"
sc.settings.figdir = out_dir

def annotate_and_plot(cell_type_name, out_prefix, fine_markers):
    adata_path = f"{out_dir}/{out_prefix}_adata.h5ad"
    if not os.path.exists(adata_path):
        print(f"{adata_path} not found.")
        return
        
    print(f"\n=== Fine annotating {cell_type_name} ===")
    sub_adata = sc.read_h5ad(adata_path)
    
    # 细分亚群打分注释
    for fine_type, genes in fine_markers.items():
        valid_genes = [g for g in genes if g in sub_adata.var_names]
        if valid_genes:
            sc.tl.score_genes(sub_adata, gene_list=valid_genes, score_name=f'{fine_type}_score')
            
    score_cols = [f'{ct}_score' for ct in fine_markers.keys() if f'{ct}_score' in sub_adata.obs.columns]
    
    if score_cols:
        sub_adata.obs[f'{cell_type_name}_Fine'] = sub_adata.obs[score_cols].idxmax(axis=1).str.replace('_score', '')
        # 聚类级别平滑：让每个Leiden cluster统一注释为该cluster内得分最高（数量最多）的类型
        mapping = {}
        leiden_col = f'leiden_{cell_type_name}'
        for cluster in sub_adata.obs[leiden_col].unique():
            dom = sub_adata.obs[sub_adata.obs[leiden_col] == cluster][f'{cell_type_name}_Fine'].value_counts().index[0]
            mapping[cluster] = dom
        sub_adata.obs['CellType_Fine'] = sub_adata.obs[leiden_col].map(mapping)
    else:
        sub_adata.obs['CellType_Fine'] = sub_adata.obs[f'leiden_{cell_type_name}']
        
    print("=== Saving Fine Annotated UMAP ===")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sc.pl.umap(sub_adata, color='Cancer_Type', show=False, ax=axs[0])
    sc.pl.umap(sub_adata, color=f'leiden_{cell_type_name}', show=False, ax=axs[1])
    sc.pl.umap(sub_adata, color='CellType_Fine', show=False, ax=axs[2])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{out_prefix}_Fine_UMAP.pdf", bbox_inches='tight')
    
    # 过滤掉不在 var_names 中的 marker 基因，画 dotplot
    filtered_markers = {}
    for ct, genes in fine_markers.items():
        valid_genes = [g for g in genes if g in sub_adata.var_names]
        if valid_genes:
            filtered_markers[ct] = valid_genes
            
    if filtered_markers:
        sc.pl.dotplot(sub_adata, filtered_markers, groupby='CellType_Fine', standard_scale='var', 
                      save=f"_{out_prefix}_Fine_Marker.pdf", show=False)
                      
    # 重新画 scTour 图，用 CellType_Fine 填色
    sct_adata_path = f"{out_dir}/{out_prefix}_sct_adata.h5ad"
    if os.path.exists(sct_adata_path):
        sct_adata = sc.read_h5ad(sct_adata_path)
        # 将细分注释映射过去
        sct_adata.obs['CellType_Fine'] = sub_adata.obs.loc[sct_adata.obs_names, 'CellType_Fine']
        
        print("=== Saving Fine Annotated scTour Trajectory ===")
        try:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            sc.pl.umap(sct_adata, color='pseudotime', show=False, ax=axs[0])
            sct.vf.plot_vector_field(sct_adata, zs_key='X_TVAE', vf_key='X_VF', use_rep_neigh='X_TVAE', 
                                     color='CellType_Fine', show=False, ax=axs[1], size=80, alpha=0.2)
            plt.savefig(f"{out_dir}/{out_prefix}_Fine_scTour_Trajectory.pdf", bbox_inches='tight')
        except Exception as e:
            print(f"Failed to plot scTour trajectory for {out_prefix}: {e}")
            
        # 尝试保存 png
        try:
            plt.savefig(f"{out_dir}/{out_prefix}_Fine_scTour_Trajectory.png", bbox_inches='tight')
        except Exception:
            pass
        
    sub_adata.write(f"{out_dir}/{out_prefix}_adata_fine.h5ad")

# Macrophage
macrophage_markers = {
    'M1-like (Pro-inflammatory)': ['CXCL9', 'CXCL10', 'IL1B', 'TNF'],
    'M2-like (Anti-inflammatory/TAM)': ['CD163', 'MRC1', 'MS4A4A', 'MAF'],
    'SPP1+ TAM (Angiogenic/Hypoxic)': ['SPP1', 'VEGFA', 'HIF1A'],
    'C1QC+ TAM (Phagocytic)': ['C1QC', 'C1QB', 'C1QA', 'APOE'],
    'Cycling Macrophages': ['MKI67', 'TOP2A']
}

# Fibroblast
fibroblast_markers = {
    'myCAF (Myofibroblastic)': ['ACTA2', 'TAGLN', 'MYL9', 'POSTN'],
    'iCAF (Inflammatory)': ['IL6', 'CXCL12', 'CCL2', 'CFD'],
    'apCAF (Antigen-presenting)': ['HLA-DRA', 'HLA-DRB1', 'CD74'],
    'Normal/Resting Fibroblast': ['PDGFRA', 'PI16']
}

# T cells
tcell_markers = {
    'CD8+ Exhausted T': ['LAG3', 'PDCD1', 'CTLA4', 'HAVCR2', 'TOX'],
    'CD8+ Effector/Memory T': ['GZMK', 'GZMB', 'PRF1', 'IFNG', 'CCL5'],
    'CD4+ Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'BATF'],
    'CD4+ Naive/Memory T': ['IL7R', 'TCF7', 'SELL', 'CCR7'],
    'Cycling T': ['MKI67', 'STMN1']
}

annotate_and_plot('Macrophages', '05_Macrophage', macrophage_markers)
annotate_and_plot('Fibroblasts', '06_Fibroblast', fibroblast_markers)
annotate_and_plot('T cells', '07_Tcell', tcell_markers)
print("=== Step 4 Complete ===")
