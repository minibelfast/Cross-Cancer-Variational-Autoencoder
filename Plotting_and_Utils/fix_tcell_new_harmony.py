import scanpy as sc
import os

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(8, 6))

method_name = 'Harmony'
method_dir = 'results_new_cross_cancer_harmony'

tcell_markers = {
    'CD8+ Effector': ['GZMA', 'GZMB', 'PRF1', 'IFNG'],
    'Exhausted': ['PDCD1', 'HAVCR2', 'LAG3', 'TOX', 'CTLA4'],
    'Tregs': ['FOXP3', 'IL2RA'],
    'CD4+ Naive/Memory': ['TCF7', 'IL7R', 'SELL']
}

fine_files = [
    f"{method_dir}/08_Tcells_adata_fine_harmony.h5ad",
    f"{method_dir}/07_Tcells_adata_fine_harmony.h5ad",
    f"{method_dir}/05_Tcells_adata_harmony.h5ad"
]

fine_file = None
for f in fine_files:
    if os.path.exists(f):
        fine_file = f
        break

if fine_file:
    print(f"Generating T cell dotplot from {fine_file}...")
    adata = sc.read_h5ad(fine_file)
    
    groupby_col = 'CellType_Fine' if 'CellType_Fine' in adata.obs.columns else ('CellType' if 'CellType' in adata.obs.columns else 'leiden_sub')
    
    if groupby_col:
        print(f"Using {groupby_col} for grouping.")
        if hasattr(adata, 'raw') and adata.raw is not None:
            adata_plot = adata.raw.to_adata()
            adata_plot.obs[groupby_col] = adata.obs[groupby_col]
        elif 'counts' in adata.layers:
            adata_plot = adata.copy()
            adata_plot.X = adata_plot.layers['counts']
        else:
            adata_plot = adata
            
        valid_markers = {k: [g for g in v if g in adata_plot.var_names] for k, v in tcell_markers.items()}
        valid_markers = {k: v for k, v in valid_markers.items() if v}
        
        if valid_markers:
            sc.settings.figdir = method_dir
            sc.pl.dotplot(adata_plot, valid_markers, groupby=groupby_col, standard_scale='var', 
                          save=f"_08_Fine_Annotation_Dotplot_Tcells_{method_name}_Fixed.pdf", show=False)
            
            old_name = f"{method_dir}/dotplot__08_Fine_Annotation_Dotplot_Tcells_{method_name}_Fixed.pdf"
            new_name = f"{method_dir}/08_Fine_Annotation_Dotplot_Tcells_{method_name}_Fixed.pdf"
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
            print("T cell dotplot generated.")
else:
    print("T cell file not found!")
