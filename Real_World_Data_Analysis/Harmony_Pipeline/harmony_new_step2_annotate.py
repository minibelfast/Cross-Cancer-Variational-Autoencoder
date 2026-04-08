import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"
sc.settings.figdir = out_dir

adata_path = f"{out_dir}/integrated_adata_harmony.h5ad"

if not os.path.exists(adata_path):
    print(f"File {adata_path} not found.")
    exit(1)

print("=== Loading integrated data ===")
adata = sc.read_h5ad(adata_path)

cluster_mapping = {
    '0': 'Epithelial/Cancer', # Glioblastoma/Neural-like (PTPRZ1, PCDH9, NPAS3)
    '1': 'T cells', # T cells (IL7R, CD3D, CD3E)
    '2': 'Macrophages', # Macrophages/Monocytes (LYZ, IFI30, AIF1)
    '3': 'T cells', # NK/CD8 T cells (CCL5, NKG7, GZMA, CD3D)
    '4': 'Macrophages', # Macrophages (C1QB, C1QA, APOE)
    '5': 'Epithelial/Cancer', # Neural/Glial/Tumor (MT3, S100B, PTN)
    '6': 'Epithelial/Cancer', # Epithelial/Carcinoma (KRT8, KRT18, EPCAM)
    '7': 'Macrophages', # Macrophages (TYROBP, CD68, FCER1G)
    '8': 'Fibroblasts', # Fibroblasts (DCN, COL1A1, COL3A1)
    '9': 'Macrophages', # Macrophages (APOE, C1QB, C1QA)
    '10': 'B cells', # B cells (MS4A1, CD79A)
    '11': 'Plasma cells', # Plasma cells (JCHAIN, MZB1, IGHA1)
    '12': 'Cycling cells', # Proliferating/Cycling (TOP2A, MKI67, CENPF)
    '13': 'Endothelial', # Endothelial (PECAM1, VWF)
    '14': 'Macrophages', # Myeloid/Monocytes (NEAT1, CSF3R)
    '15': 'Oligodendrocytes', # Oligodendrocytes/Glial (PLP1, PTGDS)
    '16': 'Fibroblasts', # Myofibroblasts/Smooth muscle (ACTA2, TAGLN, MYL9)
    '17': 'Epithelial/Cancer', # Neural-like Tumor (SOX4, EGFR, DLL3)
    '18': 'Epithelial/Cancer', # Neural-like Tumor (CTNND2, NRXN1)
    '19': 'Erythrocytes', # Erythrocytes/RBCs (HBB, HBA2)
    '20': 'Mast cells', # Mast cells (TPSAB1, CPA3, KIT)
    '21': 'Epithelial/Cancer', # Mitochondrial high/Epithelial (MT-CYB, KRT18)
    '22': 'Epithelial/Cancer' # Neural-like Tumor (TNR, DSCAM, OPCML)
}

print("=== Applying annotations ===")
# If mapping is not complete, fill missing with 'Unknown'
for c in adata.obs['leiden_harmony'].cat.categories:
    if c not in cluster_mapping:
        cluster_mapping[c] = 'Unknown'

adata.obs['CellType_Broad'] = adata.obs['leiden_harmony'].map(cluster_mapping)

print("Cluster mapping:")
for k, v in cluster_mapping.items():
    print(f"Cluster {k}: {v}")

print("=== Saving UMAP ===")
sc.pl.umap(adata, color='CellType_Broad', save="_03_All_Cells_CellType_Harmony.pdf", show=False)

print("=== Saving annotated adata ===")
adata.write(f"{out_dir}/integrated_adata_annotated_harmony.h5ad")

# Save subsets for downstream analysis
print("=== Saving subsets for downstream analysis ===")
for ct in ['Macrophages', 'Fibroblasts', 'T cells']:
    if ct in adata.obs['CellType_Broad'].values:
        sub_adata = adata[adata.obs['CellType_Broad'] == ct].copy()
        sub_adata.write(f"{out_dir}/05_{ct.replace(' ', '')}_adata_harmony.h5ad")
        print(f"Saved {ct} subset.")

print("=== Step 2 Complete ===")
