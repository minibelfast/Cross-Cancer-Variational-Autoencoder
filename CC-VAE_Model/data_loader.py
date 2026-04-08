import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import glob
import scvi
from scipy.io import mmread
from scipy.sparse import csr_matrix

def load_sample_10x(directory, prefix, matrix_file=None):
    """
    Load a single 10x sample with prefixed files.
    """
    if matrix_file:
        matrix_path = matrix_file
    else:
        # Try finding matrix file
        matrix_candidates = glob.glob(os.path.join(directory, f"{prefix}*matrix.mtx.gz"))
        if not matrix_candidates:
             raise FileNotFoundError(f"Matrix file not found for prefix {prefix}")
        matrix_path = matrix_candidates[0]

    # Find features/genes file
    # Candidates: {prefix}_features.tsv.gz, {prefix}features.tsv.gz, {prefix}_genes.tsv.gz, {prefix}genes.tsv.gz
    features_candidates = [
        f"{prefix}_features.tsv.gz", f"{prefix}features.tsv.gz",
        f"{prefix}_genes.tsv.gz", f"{prefix}genes.tsv.gz"
    ]
    features_path = None
    for cand in features_candidates:
        if os.path.exists(os.path.join(directory, cand)):
            features_path = os.path.join(directory, cand)
            break
            
    if not features_path:
        raise FileNotFoundError(f"Features/genes file not found for prefix {prefix}")

    # Find barcodes file
    barcodes_candidates = [
        f"{prefix}_barcodes.tsv.gz", f"{prefix}barcodes.tsv.gz"
    ]
    barcodes_path = None
    for cand in barcodes_candidates:
        if os.path.exists(os.path.join(directory, cand)):
            barcodes_path = os.path.join(directory, cand)
            break
            
    if not barcodes_path:
        raise FileNotFoundError(f"Barcodes file not found for prefix {prefix}")
    
    # Read matrix
    X = mmread(matrix_path).T.tocsr()
    
    # Read features
    try:
        features = pd.read_csv(features_path, sep='\t', header=None, compression='gzip')
    except Exception as e:
        # Try reading without compression if gzip fails (though extension says gz)
        try:
             features = pd.read_csv(features_path, sep='\t', header=None)
        except:
             raise IOError(f"Error reading features {features_path}: {e}")

    # Assuming columns are: id, name, type (optional)
    # Standard 10x features.tsv has 3 columns: gene_id, gene_name, feature_type
    # Sometimes only 2.
    var_names = features.iloc[:, 1].values # Use gene names if unique, else ids
    if len(np.unique(var_names)) < len(var_names):
        var_names = features.iloc[:, 0].values # Fallback to IDs
        
    var = pd.DataFrame(index=var_names)
    var['gene_ids'] = features.iloc[:, 0].values
    if features.shape[1] > 2:
        var['feature_types'] = features.iloc[:, 2].values
        
    # Read barcodes
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None, compression='gzip')
    obs_names = barcodes.iloc[:, 0].values
    obs = pd.DataFrame(index=obs_names)
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata

def load_data(data_dir="./data", use_synthetic=False, subsample_n=None):
    """
    Load single-cell RNA-seq data from specific cancer directories.
    Structure:
    data_dir/
        GSE242299 Clear cell renal cell carcinoma/
            GSM..._matrix.mtx.gz
            ...
        ...
    """
    # Define mapping from folder name to cancer type ID
    cancer_folders = {
        "GSE242299 Clear cell renal cell carcinoma": "ccRCC", 
        "GSE277524 Bladder carcinoma": "BLCA", 
        "GSE290925 Hepatocellular carcinoma": "HCC", 
        "GSE292824 Breast cancer": "BRCA", 
        "GSE311151 Glioblastoma": "GBM"
    }
    
    adatas = []
    
    # Check if we are running in the directory where these folders exist
    # Based on ls output, they are in the current directory or data_dir
    # If data_dir is provided as argument, we look there.
    # If data_dir is ".", we look in current dir.
    
    found_real_data = False
    
    for folder, cancer_type in cancer_folders.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            # Try checking relative to script location if data_dir is generic
            script_dir = os.path.dirname(os.path.abspath(__file__))
            folder_path = os.path.join(script_dir, folder)
            
        if os.path.exists(folder_path):
            print(f"Loading {cancer_type} from {folder_path}...")
            # Find all matrix files (flexible pattern)
            matrix_files = glob.glob(os.path.join(folder_path, "*matrix.mtx.gz"))
            
            for mat_file in matrix_files:
                basename = os.path.basename(mat_file)
                # Determine prefix
                if basename.endswith("_matrix.mtx.gz"):
                    prefix = basename[:-14] # remove _matrix.mtx.gz
                elif basename.endswith("matrix.mtx.gz"):
                    prefix = basename[:-13] # remove matrix.mtx.gz
                else:
                    # Should not happen given glob, but safety
                    prefix = basename.replace("matrix.mtx.gz", "")
                
                try:
                    adata = load_sample_10x(folder_path, prefix, matrix_file=mat_file)
                    
                    # Memory Optimization: Filter cells immediately
                    sc.pp.filter_cells(adata, min_genes=200)
                    if adata.n_obs == 0:
                        print(f"  Skipping {prefix}: no cells passing filter")
                        continue
                        
                    adata.obs['batch'] = cancer_type # Use cancer type as batch for integration task
                    adata.obs['sample_id'] = prefix
                    adata.obs['cancer_type'] = cancer_type
                    # Make var names unique
                    adata.var_names_make_unique()
                    adatas.append(adata)
                    found_real_data = True
                    print(f"  Loaded {prefix}: {adata.n_obs} cells (filtered)")
                except Exception as e:
                    print(f"  Failed to load {prefix}: {e}")
        else:
            print(f"Folder {folder} not found.")

    if found_real_data and not use_synthetic:
        print("Concatenating all datasets...")
        # Concatenate (outer join to keep all genes, or intersection)
        # Intersection is safer for integration
        # combined_adata = ad.concat(adatas, join='inner', label="batch_key", index_unique="-")
        
        # We manually added 'batch' column to obs, so it should be preserved.
        combined_adata = ad.concat(adatas, join='inner', merge="same")
        combined_adata.obs_names_make_unique()
        print(f"Total cells: {combined_adata.n_obs}, Total genes: {combined_adata.n_vars}")
        
        if subsample_n is not None and combined_adata.n_obs > subsample_n:
            print(f"Subsampling to {subsample_n} cells for testing...")
            sc.pp.subsample(combined_adata, n_obs=subsample_n)
            
        return combined_adata
    
    # Fallback to synthetic
    if use_synthetic or not found_real_data:
        print("Generating synthetic multi-cancer dataset for demonstration...")
        # Simulate 5 batches (cancers) with shared cell types but batch effects
        cancer_types = list(cancer_folders.values())
        n_batches = len(cancer_types)
        dataset = scvi.data.synthetic_iid(n_batches=n_batches, n_labels=5, n_genes=2000, n_proteins=0)
        
        # Rename batches to cancer types
        batch_map = {str(i): cancer_types[i] for i in range(n_batches)}
        dataset.obs['batch'] = dataset.obs['batch'].map(batch_map)
        
        # Add some metadata to simulate clinical features
        dataset.obs['cancer_type'] = dataset.obs['batch']
        dataset.obs['sample_id'] = [f"Sample_{i}" for i in range(dataset.n_obs)]
        
        return dataset

def preprocess_data(adata):
    """
    Standard preprocessing pipeline: QC, normalization, log1p, HVG selection.
    """
    print("Preprocessing data...")
    dataset = adata  # Operate in-place to save memory
    
    # QC
    dataset.var['mt'] = dataset.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(dataset, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells/genes (relaxed thresholds for synthetic data)
    sc.pp.filter_cells(dataset, min_genes=200)
    sc.pp.filter_genes(dataset, min_cells=3)
    
    # Normalize
    sc.pp.normalize_total(dataset, target_sum=1e4)
    sc.pp.log1p(dataset)
    
    # Store counts in layer for scVI
    dataset.layers["counts"] = adata[dataset.obs_names, dataset.var_names].X.copy()
    
    # Highly variable genes
    sc.pp.highly_variable_genes(dataset, n_top_genes=2000, batch_key="batch", subset=True)
    
    return dataset

if __name__ == "__main__":
    adata = load_data()
    print(adata)
    adata_pp = preprocess_data(adata)
    print(adata_pp)
