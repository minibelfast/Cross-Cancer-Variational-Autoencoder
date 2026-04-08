import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(10, 6))

method_name = 'Harmony'
method_dir = 'results_real_cross_cancer_harmony'

print(f"\nProcessing {method_name} in {method_dir}...")

broad_markers = {
    'T cells': ['CD3D', 'CD3E', 'CD2', 'CD8A', 'CD4'],
    'Macrophages/Myeloid': ['CD68', 'CD163', 'C1QA', 'LYZ', 'CD14'],
    'Fibroblasts': ['COL1A1', 'DCN', 'LUM', 'THY1', 'ACTA2'],
    'Epithelial/Tumor': ['EPCAM', 'KRT8', 'KRT18', 'KRT19'],
    'Endothelial': ['PECAM1', 'VWF', 'ENG', 'CDH5'],
    'B cells': ['CD79A', 'MS4A1', 'CD19'],
    'Plasma cells': ['MZB1', 'SDC1', 'JCHAIN'],
    'Mast cells': ['CPA3', 'TPSAB1']
}

broad_file = f"{method_dir}/integrated_adata_annotated_harmony.h5ad"
if os.path.exists(broad_file):
    print(f"  Generating broad annotation dotplot for {method_name}...")
    adata = sc.read_h5ad(broad_file)
    groupby_col = 'CellType_Broad' if 'CellType_Broad' in adata.obs.columns else ('CellType' if 'CellType' in adata.obs.columns else None)
    
    if groupby_col:
        print(f"  Cell types found: {adata.obs[groupby_col].unique().tolist()}")
        if hasattr(adata, 'raw') and adata.raw is not None:
            adata_plot = adata.raw.to_adata()
            adata_plot.obs[groupby_col] = adata.obs[groupby_col]
        else:
            adata_plot = adata
            
        valid_broad = {k: [g for g in v if g in adata_plot.var_names] for k, v in broad_markers.items()}
        valid_broad = {k: v for k, v in valid_broad.items() if v}
        
        if valid_broad:
            sc.settings.figdir = method_dir
            sc.pl.dotplot(adata_plot, valid_broad, groupby=groupby_col, standard_scale='var', 
                          save=f"_04_Broad_Annotation_Dotplot_Fixed_{method_name}.pdf", show=False)
            
            # Make sure to rename the file to remove dotplot__ prefix
            old_name = f"{method_dir}/dotplot__04_Broad_Annotation_Dotplot_Fixed_{method_name}.pdf"
            new_name = f"{method_dir}/04_Broad_Annotation_Dotplot_Fixed_{method_name}.pdf"
            if os.path.exists(old_name):
                os.rename(old_name, new_name)

print("Done!")
