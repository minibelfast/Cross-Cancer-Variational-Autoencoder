import scanpy as sc
from simulation import generate_simulation_data
from model import CrossCancerVAE

adata = generate_simulation_data(n_organs=2, fc=1.5)
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

cc_vae = CrossCancerVAE(adata, batch_key='batch')
cc_vae.train(max_epochs=2)
emb = cc_vae.get_latent_representation()
denoised = cc_vae.get_denoised_expression()
print("Success!", emb.shape, denoised.shape)
