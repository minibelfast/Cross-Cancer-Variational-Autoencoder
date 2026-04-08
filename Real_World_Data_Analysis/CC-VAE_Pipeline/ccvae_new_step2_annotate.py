import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_ccvae"
sc.settings.figdir = out_dir

adata_path = f"{out_dir}/integrated_adata_ccvae.h5ad"

if not os.path.exists(adata_path):
    print(f"File {adata_path} not found.")
    exit(1)

print("=== Loading integrated data ===")
adata = sc.read_h5ad(adata_path)

cluster_mapping = {
    '0': 'T cells', # CD3D, IL7R, CD3E
    '1': 'Epithelial/Cancer', # PTN, PTPRZ1, S100B (Glioblastoma)
    '2': 'Macrophages', # CD74, HLA-DRA, C1QA, C1QB
    '3': 'Macrophages', # TYROBP, FCER1G, IFI30
    '4': 'Epithelial/Cancer', # PCDH9, EGFR (Glioblastoma)
    '5': 'Macrophages', # LYZ, S100A9, FCN1 (Monocytes)
    '6': 'Macrophages', # CD163, CD14, AIF1
    '7': 'Fibroblasts', # DCN, COL1A1, COL3A1
    '8': 'Epithelial/Cancer', # KRT19, KRT8 (Carcinoma)
    '9': 'Cycling cells', # STMN1, CENPF, HMGB1
    '10': 'B cells', # MS4A1, CD79A
    '11': 'Epithelial/Cancer', # MT3, SEC61G, EGFR
    '12': 'Epithelial/Cancer', # GFAP, S100B (Astrocytes/Glioblastoma)
    '13': 'T cells', # CD3E, CD3D
    '14': 'T cells', # NKG7, GZMB, GZMA (NK/CD8 T)
    '15': 'Endothelial', # VWF, PECAM1
    '16': 'Plasma cells', # JCHAIN, MZB1, IGHA1
    '17': 'Epithelial/Cancer', # KRT18, KRT8
    '18': 'Epithelial/Cancer', # GFAP, EGFR
    '19': 'Fibroblasts', # ACTA2, TAGLN, MYL9 (myCAF/Smooth Muscle)
    '20': 'Oligodendrocytes', # PLP1, PTGDS
    '21': 'Macrophages', # PLXDC2 (Myeloid)
    '22': 'Epithelial/Cancer', # SOX4, EGFR
    '23': 'Cycling cells', # TUBA1B, STMN1
    '24': 'Epithelial/Cancer', # KRT8, KRT18
    '25': 'Mast cells', # TPSAB1, CPA3, KIT
    '26': 'Macrophages', # CXCL8, S100A8/9 (Neutrophils/Monocytes)
    '27': 'Erythrocytes', # HBA2, HBB
    '28': 'Epithelial/Cancer', # VIM, SPARC
    '29': 'Dendritic cells', # LAMP3, CCR7
    '30': 'Macrophages', # Myeloid/Monocytes
    '31': 'Epithelial/Cancer', # Oligodendrocyte-like tumor (OPC)
    '32': 'Plasmacytoid DCs', # GZMB, JCHAIN, IL3RA (pDC)
    '33': 'Epithelial/Cancer', # AGR2, PSCA
    '34': 'Erythrocytes' # HBA2, HBB
}

print("=== Applying annotations ===")
for c in adata.obs['leiden_ccvae'].cat.categories:
    if c not in cluster_mapping:
        cluster_mapping[c] = 'Unknown'

adata.obs['CellType_Broad'] = adata.obs['leiden_ccvae'].map(cluster_mapping)

print("Cluster mapping:")
for k, v in cluster_mapping.items():
    print(f"Cluster {k}: {v}")

print("=== Saving UMAP ===")
sc.pl.umap(adata, color='CellType_Broad', save="_03_All_Cells_CellType_CCVAE.pdf", show=False)

print("=== Saving annotated adata ===")
adata.write(f"{out_dir}/integrated_adata_annotated_ccvae.h5ad")

# Save subsets for downstream analysis
print("=== Saving subsets for downstream analysis ===")
for ct in ['Macrophages', 'Fibroblasts', 'T cells']:
    if ct in adata.obs['CellType_Broad'].values:
        sub_adata = adata[adata.obs['CellType_Broad'] == ct].copy()
        sub_adata.write(f"{out_dir}/05_{ct.replace(' ', '')}_adata_ccvae.h5ad")
        print(f"Saved {ct} subset.")

print("=== Step 2 Complete ===")
