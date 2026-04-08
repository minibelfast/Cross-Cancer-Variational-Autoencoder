import scanpy as sc
import pandas as pd
import glob
import os
import scanpy.external as sce
import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

datasets = {
    'Colon': '/mnt/data3/STrnaseq/script/SCbatch/GSE289314 Colon cancer',
    'NSCLC': '/mnt/data3/STrnaseq/script/SCbatch/GSE299111 Non-small cell lung cancer',
    'GBM': '/mnt/data3/STrnaseq/script/SCbatch/GSE311151 Glioblastoma',
    'PDAC': '/mnt/data3/STrnaseq/script/SCbatch/GSE311788 Pancreatic ductal adenocarcinoma'
}

adatas = []
for cancer, path in datasets.items():
    print(f"Loading {cancer}...")
    matrix_files = glob.glob(f"{path}/*_matrix.mtx.gz")
    for mat_file in matrix_files:
        prefix = mat_file.replace('_matrix.mtx.gz', '')
        feat_file = prefix + '_features.tsv.gz'
        if not os.path.exists(feat_file):
            feat_file = prefix + '_genes.tsv.gz'
            
        bar_file = prefix + '_barcodes.tsv.gz'
        batch_id = os.path.basename(prefix)
        
        print(f"Reading {batch_id}...")
        try:
            adata = sc.read_mtx(mat_file).T
            
            genes = pd.read_csv(feat_file, header=None, sep='\t')
            if genes.shape[1] >= 2:
                # Typically index 1 is gene symbol for 10x
                # But sometimes it's index 0. We saw for NSCLC, GBM, PDAC it's index 1
                adata.var_names = genes[1].values
            else:
                # For Colon, it's just 1 column (symbols)
                adata.var_names = genes[0].values
                
            barcodes = pd.read_csv(bar_file, header=None)
            adata.obs_names = barcodes[0].values
            
            adata.var_names_make_unique()
            adata.obs['Batch'] = batch_id
            adata.obs['Cancer_Type'] = cancer
            
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
            adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
            
            adatas.append(adata)
        except Exception as e:
            print(f"Error loading {batch_id}: {e}")

import gc
print("Concatenating datasets...")
import functools
# Find common genes
common_genes = list(functools.reduce(set.intersection, [set(a.var_names) for a in adatas]))
print(f"Found {len(common_genes)} common genes across all batches.")

if len(common_genes) == 0:
    print("ERROR: 0 common genes. Check gene name formatting.")
    for a in adatas:
        a.var_names = a.var_names.str.upper()
    common_genes = list(functools.reduce(set.intersection, [set(a.var_names) for a in adatas]))
    print(f"After upper(), found {len(common_genes)} common genes.")

adatas_subset = [a[:, common_genes].copy() for a in adatas]
del adatas
gc.collect()

import anndata as ad
adata_concat = ad.concat(adatas_subset, label='concat_batch')
del adatas_subset
gc.collect()

adata_concat.obs_names_make_unique()

print("Preprocessing concatenated data...")
adata_concat.layers["counts"] = adata_concat.X.copy()
sc.pp.normalize_total(adata_concat, target_sum=1e4)
sc.pp.log1p(adata_concat)

print("Highly variable genes...")
sc.pp.highly_variable_genes(adata_concat, n_top_genes=2000, flavor='cell_ranger', layer='counts')

print("PCA...")
sc.tl.pca(adata_concat, svd_solver='arpack')

import gc
gc.collect()

print("Harmony integration...")
sce.pp.harmony_integrate(adata_concat, 'Batch')

print("Clustering and UMAP...")
sc.pp.neighbors(adata_concat, use_rep='X_pca_harmony', n_neighbors=30)
sc.tl.umap(adata_concat)
sc.tl.leiden(adata_concat, resolution=0.8, key_added='leiden_harmony')

print("Saving integration plots...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_concat, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata_concat, color='Batch', show=False, ax=axs[1], legend_loc='none')
sc.pl.umap(adata_concat, color='leiden_harmony', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/01_All_Cells_UMAP_Harmony.pdf", bbox_inches='tight')

print("Calculating Markers...")
sc.tl.rank_genes_groups(adata_concat, groupby='leiden_harmony', method='wilcoxon')
markers_df = pd.DataFrame(adata_concat.uns['rank_genes_groups']['names'])
markers_df.to_csv(f"{out_dir}/02_All_Cells_Cluster_Markers_Harmony.csv", index=False)

print("Saving AnnData...")
adata_concat.write(f"{out_dir}/integrated_adata_harmony.h5ad")
print("=== Step 1 Complete ===")
