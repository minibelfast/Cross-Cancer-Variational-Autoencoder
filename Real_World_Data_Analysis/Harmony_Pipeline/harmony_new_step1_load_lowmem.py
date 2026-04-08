import scanpy as sc
import pandas as pd
import glob
import os
import scanpy.external as sce
import matplotlib.pyplot as plt
import functools
import anndata as ad
import gc

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

tmp_dir = "tmp_adatas"
os.makedirs(tmp_dir, exist_ok=True)

datasets = {
    'Colon': '/mnt/data3/STrnaseq/script/SCbatch/GSE289314 Colon cancer',
    'NSCLC': '/mnt/data3/STrnaseq/script/SCbatch/GSE299111 Non-small cell lung cancer',
    'GBM': '/mnt/data3/STrnaseq/script/SCbatch/GSE311151 Glioblastoma',
    'PDAC': '/mnt/data3/STrnaseq/script/SCbatch/GSE311788 Pancreatic ductal adenocarcinoma'
}

def clean_genes(var_names):
    cleaned = []
    prefixes = ['GRCH38_', 'HG19_', 'GRCM38_', 'MM10_']
    for name in var_names:
        name = str(name).upper()
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
        cleaned.append(name)
    return cleaned

print("=== Step 1: Finding common genes ===", flush=True)
batch_genes = []
batch_info = []

for cancer, path in datasets.items():
    matrix_files = glob.glob(f"{path}/*_matrix.mtx.gz")
    for mat_file in matrix_files:
        prefix = mat_file.replace('_matrix.mtx.gz', '')
        feat_file = prefix + '_features.tsv.gz'
        if not os.path.exists(feat_file):
            feat_file = prefix + '_genes.tsv.gz'
            
        bar_file = prefix + '_barcodes.tsv.gz'
        batch_id = os.path.basename(prefix)
        
        try:
            genes = pd.read_csv(feat_file, header=None, sep='\t')
            if genes.shape[1] >= 2:
                var_names = genes[1].values
            else:
                var_names = genes[0].values
                
            cleaned_names = clean_genes(var_names)
            batch_genes.append(set(cleaned_names))
            batch_info.append((cancer, batch_id, mat_file, feat_file, bar_file))
        except Exception as e:
            print(f"Error reading features for {batch_id}: {e}", flush=True)

common_genes = list(functools.reduce(set.intersection, batch_genes))
print(f"Found {len(common_genes)} common genes across all {len(batch_info)} batches.", flush=True)

if len(common_genes) == 0:
    print("FATAL ERROR: 0 common genes. Exiting.", flush=True)
    exit(1)

print("=== Step 2: Processing and saving individual batches ===", flush=True)
saved_files = []
for cancer, batch_id, mat_file, feat_file, bar_file in batch_info:
    tmp_file = f"{tmp_dir}/{batch_id}.h5ad"
    if os.path.exists(tmp_file):
        print(f"Skipping {batch_id}, already processed.", flush=True)
        saved_files.append((tmp_file, batch_id))
        continue
        
    print(f"Processing {batch_id}...", flush=True)
    try:
        adata = sc.read_mtx(mat_file).T
        
        genes = pd.read_csv(feat_file, header=None, sep='\t')
        if genes.shape[1] >= 2:
            raw_names = genes[1].values
        else:
            raw_names = genes[0].values
            
        adata.var_names = clean_genes(raw_names)
        
        barcodes = pd.read_csv(bar_file, header=None)
        adata.obs_names = barcodes[0].values
        adata.var_names_make_unique()
        
        # Subset to common genes immediately to save memory
        # Wait, since common_genes are unique, we just subset
        # But let's intersect first
        valid_common = [g for g in common_genes if g in adata.var_names]
        adata = adata[:, valid_common].copy()
        
        adata.obs['Batch'] = batch_id
        adata.obs['Cancer_Type'] = cancer
        
        sc.pp.filter_cells(adata, min_genes=200)
        
        if adata.n_obs > 0:
            # Calculate MT percent
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
            adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
        
        if adata.n_obs > 0:
            adata.write(tmp_file)
            saved_files.append((tmp_file, batch_id))
        else:
            print(f"Warning: {batch_id} has 0 cells after filtering.", flush=True)
        
        del adata
        gc.collect()
    except Exception as e:
        print(f"Error processing {batch_id}: {e}", flush=True)

print("=== Step 3: Concatenating all batches ===", flush=True)
adatas_list = []
for f, b in saved_files:
    adatas_list.append(sc.read_h5ad(f))

adata_concat = ad.concat(adatas_list, label='concat_batch')
del adatas_list
gc.collect()

# Ensure we only have strictly common genes across all remaining subsets
# (in case some batch had duplicated gene names that were dropped or something)
common_genes_final = list(functools.reduce(set.intersection, [set(a.var_names) for a in [sc.read_h5ad(f) for f, b in saved_files]]))
adata_concat = adata_concat[:, common_genes_final].copy()

adata_concat.obs_names_make_unique()
sc.pp.filter_genes(adata_concat, min_cells=3)

print(f"Concatenated shape: {adata_concat.shape}", flush=True)

print("Preprocessing concatenated data...", flush=True)
adata_concat.layers["counts"] = adata_concat.X.copy()
sc.pp.normalize_total(adata_concat, target_sum=1e4)
sc.pp.log1p(adata_concat)

print("Highly variable genes...", flush=True)
sc.pp.highly_variable_genes(adata_concat, n_top_genes=2000, flavor='cell_ranger', layer='counts')

print("PCA...", flush=True)
sc.tl.pca(adata_concat, svd_solver='arpack')

import gc
gc.collect()

print("Harmony integration...", flush=True)
sce.pp.harmony_integrate(adata_concat, 'Batch')

print("Clustering and UMAP...", flush=True)
sc.pp.neighbors(adata_concat, use_rep='X_pca_harmony', n_neighbors=30)
sc.tl.umap(adata_concat)
sc.tl.leiden(adata_concat, resolution=0.8, key_added='leiden_harmony')

print("Saving integration plots...", flush=True)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_concat, color='Cancer_Type', show=False, ax=axs[0])
sc.pl.umap(adata_concat, color='Batch', show=False, ax=axs[1], legend_loc='none')
sc.pl.umap(adata_concat, color='leiden_harmony', show=False, ax=axs[2])
plt.tight_layout()
plt.savefig(f"{out_dir}/01_All_Cells_UMAP_Harmony.pdf", bbox_inches='tight')

print("Calculating Markers...", flush=True)
sc.tl.rank_genes_groups(adata_concat, groupby='leiden_harmony', method='wilcoxon')
markers_df = pd.DataFrame(adata_concat.uns['rank_genes_groups']['names'])
markers_df.to_csv(f"{out_dir}/02_All_Cells_Cluster_Markers_Harmony.csv", index=False)

print("Saving AnnData...", flush=True)
adata_concat.write(f"{out_dir}/integrated_adata_harmony.h5ad")
print("=== Step 1 Complete ===", flush=True)
