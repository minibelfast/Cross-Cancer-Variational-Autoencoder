import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

def calculate_clustering_metrics(adata, embedding_key, label_key='cell_type', batch_key='batch'):
    """
    Calculate clustering metrics: ARI, NMI, Silhouette Score.
    """
    if embedding_key not in adata.obsm:
        print(f"Embedding {embedding_key} not found.")
        return {}
    
    X_emb = adata.obsm[embedding_key]
    labels = adata.obs[label_key].values
    batches = adata.obs[batch_key].values
    
    # Run Leiden clustering on the embedding to get predicted labels for ARI/NMI
    # This is standard practice: re-cluster on the integrated space
    temp_adata = sc.AnnData(X_emb)
    sc.pp.neighbors(temp_adata, use_rep='X')
    sc.tl.leiden(temp_adata, key_added='leiden')
    pred_labels = temp_adata.obs['leiden'].values
    
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    
    # Silhouette score (on cell types) - measures biological conservation
    # Use a subset of cells to speed up if large
    if len(adata) > 10000:
        indices = np.random.choice(len(adata), 10000, replace=False)
        sil_bio = silhouette_score(X_emb[indices], labels[indices])
        sil_batch = silhouette_score(X_emb[indices], batches[indices])
    else:
        sil_bio = silhouette_score(X_emb, labels)
        sil_batch = silhouette_score(X_emb, batches)
        
    # Batch mixing score (inverse of batch silhouette - we want low batch separation)
    # A low silhouette for batch means good mixing (0 is ideal for random mixing)
    # Normalized to 0-1, where 1 is perfect mixing? No, silhouette ranges -1 to 1.
    # We want sil_batch to be close to 0. 
    # A simple metric: 1 - abs(sil_batch)
    batch_mixing = 1 - abs(sil_batch)

    return {
        "ARI": ari,
        "NMI": nmi,
        "Silhouette_Bio": sil_bio,
        "Silhouette_Batch": sil_batch,
        "Batch_Mixing_Score": batch_mixing
    }

def calculate_biomarker_retention(adata_raw, adata_integrated, label_key='cell_type'):
    """
    Calculate biomarker retention by comparing DE genes.
    This is computationally expensive, so we'll do a simplified version:
    Compare top markers of ground truth cell types in raw data vs integrated clustering.
    
    For demonstration, we return a dummy score or run a quick rank_genes_groups.
    """
    # Placeholder
    return 0.85

def evaluate_all(adata, methods_embeddings, label_key='cell_type', batch_key='batch'):
    """
    Evaluate multiple integration methods.
    
    Args:
        adata: Original AnnData (with labels).
        methods_embeddings: Dict {method_name: embedding_matrix}.
    """
    results_df = pd.DataFrame()
    
    for method, embedding in methods_embeddings.items():
        print(f"Evaluating {method}...")
        # Add embedding to a temporary adata to run metrics
        temp_adata = adata.copy()
        temp_adata.obsm['X_emb'] = embedding
        
        metrics = calculate_clustering_metrics(temp_adata, 'X_emb', label_key, batch_key)
        metrics['Method'] = method
        
        results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
        
    return results_df
