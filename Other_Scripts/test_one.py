import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from simulation import generate_simulation_data
from model import CrossCancerVAE
from integration_benchmarks import run_benchmarks
from evaluation_advanced import evaluate_integration_rigorous

adata = generate_simulation_data(n_cells=2000, n_organs=3, fc=2.0)
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

embeddings = {}

# CC-VAE
cc_vae = CrossCancerVAE(adata, batch_key='Patient_Batch') 
cc_vae.train(max_epochs=30) 
embeddings['CC-VAE'] = {
    'emb': cc_vae.get_latent_representation(),
    'denoised': cc_vae.get_denoised_expression()
}

# Benchmarks
bench_res = run_benchmarks(adata, methods=['Harmony', 'scVI'], batch_key='Patient_Batch')
embeddings.update(bench_res)

# Evaluate
all_metrics = []
for method, res in embeddings.items():
    temp_adata = adata.copy()
    
    emb = res['emb']
    denoised = res['denoised']
    
    if denoised is not None:
        temp_adata.X = np.log1p(denoised)
        
    temp_adata.obsm['X_emb'] = emb
    sc.pp.neighbors(temp_adata, use_rep='X_emb')
    sc.tl.leiden(temp_adata, key_added='leiden')
        
    metrics = evaluate_integration_rigorous(temp_adata, 'X_emb', label_key='Detailed_Subtype', batch_key='Patient_Batch')
    metrics['Method'] = method
    all_metrics.append(metrics)

df = pd.DataFrame(all_metrics)
print(df[['Method', 'ARI', 'iLISI', 'F1_DEG']])
