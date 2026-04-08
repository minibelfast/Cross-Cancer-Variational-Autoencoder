import scanpy as sc
import numpy as np
from simulation import generate_simulation_data
from evaluation_advanced import evaluate_integration_rigorous
from model import CrossCancerVAE

adata = generate_simulation_data(n_cells=2000, n_organs=3, fc=2.0)
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

cc_vae = CrossCancerVAE(adata, batch_key='Patient_Batch')
cc_vae.train(max_epochs=100)

emb = cc_vae.get_latent_representation()
denoised = cc_vae.get_denoised_expression()

temp_adata = adata.copy()
temp_adata.X = np.log1p(denoised)
temp_adata.obsm['X_pca'] = emb

label_key = 'Detailed_Subtype'
deg_adata = temp_adata.copy()
sc.tl.rank_genes_groups(deg_adata, groupby=label_key, method='wilcoxon', key_added='rank_genes')

cluster_true_degs = {}
for c in deg_adata.obs[label_key].unique():
    genes = []
    if "Liver_Macrophage_Kupffer" in c:
        genes.extend([f"Gene_{i}" for i in range(0, 5)])
    if "Lung_Macrophage_Alveolar" in c:
        genes.extend([f"Gene_{i}" for i in range(5, 10)])
    if "Kidney_Macrophage_Interstitial" in c:
        genes.extend([f"Gene_{i}" for i in range(10, 15)])
    if "Tumor" in c or "TAM" in c:
        genes.extend([f"Gene_{i}" for i in range(15, 25)])
    if "Macrophage" in c:
        genes.extend([f"Gene_{i}" for i in range(25, 35)])
    if "T_cell" in c or "T_" in c:
        genes.extend([f"Gene_{i}" for i in range(35, 45)])
    
    if genes:
        cluster_true_degs[c] = set(genes)

result = deg_adata.uns['rank_genes']
groups = result['names'].dtype.names

f1_scores = []
for group in groups:
    if group in cluster_true_degs:
        true_set = cluster_true_degs[group]
        n_true = len(true_set)
        pred_set = set(result['names'][group][:n_true])
        
        tp = len(pred_set.intersection(true_set))
        print(f"Group: {group}, True: {n_true}, TP: {tp}")
        f1 = tp / n_true if n_true > 0 else 0
        f1_scores.append(f1)

print("Mean F1:", np.mean(f1_scores))
