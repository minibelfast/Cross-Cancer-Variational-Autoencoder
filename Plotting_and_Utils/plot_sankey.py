import scanpy as sc
import anndata as ad
import pandas as pd
import plotly.graph_objects as go
import os

out_dir = "results_real_cross_cancer_harmony/comparison"
os.makedirs(out_dir, exist_ok=True)

comparisons = {
    'Macrophages': {
        'ccvae': 'results_real_cross_cancer/05_Macrophage_adata_fine.h5ad',
        'harmony': 'results_real_cross_cancer_harmony/07_Macrophages_adata_fine_harmony.h5ad'
    },
    'Fibroblasts': {
        'ccvae': 'results_real_cross_cancer/06_Fibroblast_adata_fine.h5ad',
        'harmony': 'results_real_cross_cancer_harmony/07_Fibroblasts_adata_fine_harmony.h5ad'
    },
    'T_cells': {
        'ccvae': 'results_real_cross_cancer/07_Tcell_adata_fine.h5ad',
        'harmony': 'results_real_cross_cancer_harmony/07_Tcells_adata_fine_harmony.h5ad'
    }
}

for ct, paths in comparisons.items():
    print(f"Processing {ct}...")
    
    # Load just the metadata
    adata_ccvae = ad.read_h5ad(paths['ccvae'], backed='r')
    obs_ccvae = adata_ccvae.obs[['CellType_Fine']].copy()
    obs_ccvae.columns = ['CC_VAE']
    
    adata_harmony = ad.read_h5ad(paths['harmony'], backed='r')
    obs_harmony = adata_harmony.obs[['CellType_Fine']].copy()
    obs_harmony.columns = ['Harmony']
    
    # Merge on cell index
    df = obs_ccvae.join(obs_harmony, how='inner')
    
    # Calculate transition matrix
    transition = df.groupby(['CC_VAE', 'Harmony']).size().reset_index(name='Count')
    transition = transition[transition['Count'] > 0]
    
    # Save the cross-tabulation to CSV
    pivot_df = df.pivot_table(index='CC_VAE', columns='Harmony', aggfunc='size', fill_value=0)
    pivot_df.to_csv(f"{out_dir}/{ct}_Sankey_Matrix.csv")
    
    # Build Sankey
    # Nodes: CC_VAE (source), Harmony (target)
    # We need to map labels to indices
    ccvae_labels = [f"{x} (CC-VAE)" for x in transition['CC_VAE'].unique()]
    harmony_labels = [f"{x} (Harmony)" for x in transition['Harmony'].unique()]
    
    all_labels = ccvae_labels + harmony_labels
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    sources = [label_to_idx[f"{x} (CC-VAE)"] for x in transition['CC_VAE']]
    targets = [label_to_idx[f"{x} (Harmony)"] for x in transition['Harmony']]
    values = transition['Count'].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = all_labels
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values
      ))])
    
    fig.update_layout(title_text=f"{ct} Subtypes: CC-VAE vs Harmony", font_size=10)
    
    # Save as PDF
    fig.write_image(f"{out_dir}/{ct}_Sankey.pdf", width=900, height=600)
    # Also save as HTML just in case
    fig.write_html(f"{out_dir}/{ct}_Sankey.html")
    
print("Sankey diagrams generated successfully.")
