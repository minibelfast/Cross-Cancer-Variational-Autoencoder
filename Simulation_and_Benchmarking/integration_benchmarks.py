import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scvi
import bbknn
import scanpy.external as sce

def run_harmony(adata, batch_key='batch'):
    """
    Run Harmony integration.
    """
    print("Running Harmony...")
    adata_harmony = adata.copy()
    # Ensure PCA is computed
    if 'X_pca' not in adata_harmony.obsm:
        sc.pp.pca(adata_harmony)
    
    sce.pp.harmony_integrate(adata_harmony, key=batch_key)
    return adata_harmony.obsm['X_pca_harmony']

def run_bbknn(adata, batch_key='batch'):
    """
    Run BBKNN integration.
    Returns the connectivity graph, not an embedding directly, 
    but we can run UMAP on it.
    """
    print("Running BBKNN...")
    adata_bbknn = adata.copy()
    if 'X_pca' not in adata_bbknn.obsm:
        sc.pp.pca(adata_bbknn)
        
    bbknn.bbknn(adata_bbknn, batch_key=batch_key)
    # BBKNN modifies the neighbors graph in-place
    # To get an embedding, we must run UMAP
    sc.tl.umap(adata_bbknn)
    return adata_bbknn.obsm['X_umap']

def run_scanorama(adata, batch_key='batch'):
    """
    Run Scanorama integration.
    """
    # Check if scanorama is installed
    try:
        import scanorama
    except ImportError:
        print("Scanorama not installed, skipping.")
        return None

    print("Running Scanorama...")
    adata_scanorama = adata.copy()
    sce.pp.scanorama_integrate(adata_scanorama, key=batch_key)
    return adata_scanorama.obsm['X_scanorama']

def run_scvi(adata, batch_key='batch'):
    """
    Run scVI integration (standard implementation).
    """
    print("Running scVI (Benchmark)...")
    adata_scvi = adata.copy()
    scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(adata_scvi)
    model.train(max_epochs=200, early_stopping=True) # fewer epochs for benchmark speed
    return {
        'emb': model.get_latent_representation(),
        'denoised': model.get_normalized_expression(library_size=1e4)
    }

def run_benchmarks(adata, methods=['Harmony', 'BBKNN', 'scVI'], batch_key='batch'):
    """
    Run multiple benchmark methods and return a dictionary of embeddings.
    """
    results = {}
    
    # Raw PCA (Baseline)
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata)
    results['Uncorrected'] = {'emb': adata.obsm['X_pca'], 'denoised': None}
    
    if 'Harmony' in methods:
        try:
            results['Harmony'] = {'emb': run_harmony(adata, batch_key), 'denoised': None}
        except Exception as e:
            print(f"Harmony failed: {e}")
            
    if 'BBKNN' in methods:
        try:
            results['BBKNN'] = {'emb': run_bbknn(adata, batch_key), 'denoised': None}
        except Exception as e:
            print(f"BBKNN failed: {e}")
            
    if 'scVI' in methods:
        try:
            results['scVI'] = run_scvi(adata, batch_key)
        except Exception as e:
            print(f"scVI failed: {e}")
            
    return results
