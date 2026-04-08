import scanpy as sc
import os

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(8, 6))

methods = {
    'CCVAE': 'results_new_cross_cancer_ccvae',
    'Harmony': 'results_real_cross_cancer_harmony'
}

tcell_markers = {
    'CD8+ Effector': ['GZMA', 'GZMB', 'PRF1', 'IFNG'],
    'Exhausted': ['PDCD1', 'HAVCR2', 'LAG3', 'TOX', 'CTLA4'],
    'Tregs': ['FOXP3', 'IL2RA'],
    'CD4+ Naive/Memory': ['TCF7', 'IL7R', 'SELL']
}

for method_name, method_dir in methods.items():
    fine_file = f"{method_dir}/08_Tcells_adata_fine_{method_name.lower()}.h5ad"
    if not os.path.exists(fine_file):
        fine_file = f"{method_dir}/08_Tcells_adata_sctour_harmony.h5ad"
        if not os.path.exists(fine_file):
            fine_file = f"{method_dir}/07_Tcells_adata_fine_harmony.h5ad"
            
    if os.path.exists(fine_file):
        adata = sc.read_h5ad(fine_file)
        groupby_col = 'CellType_Fine' if 'CellType_Fine' in adata.obs.columns else None
        
        if groupby_col:
            valid_markers = {k: [g for g in v if g in adata.var_names] for k, v in tcell_markers.items()}
            valid_markers = {k: v for k, v in valid_markers.items() if v}
            
            if valid_markers:
                # Set saving directly to the target folder to avoid figures/ confusion
                sc.settings.figdir = method_dir
                sc.pl.dotplot(adata, valid_markers, groupby=groupby_col, standard_scale='var', 
                              save=f"_08_Fine_Annotation_Dotplot_Tcells_{method_name}.pdf", show=False)
