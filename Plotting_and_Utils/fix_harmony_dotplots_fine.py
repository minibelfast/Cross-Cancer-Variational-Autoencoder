import scanpy as sc
import os

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(8, 6))

method_name = 'Harmony'
method_dir = 'results_real_cross_cancer_harmony'

fine_markers = {
    'Macrophages': {
        'M1/Pro-inflammatory': ['CXCL9', 'CXCL10', 'HLA-DRA', 'IL1B'],
        'M2/Suppressive': ['SPP1', 'VEGFA', 'IL10', 'MRC1', 'TREM2'],
        'Cycling': ['MKI67', 'TOP2A']
    },
    'Fibroblasts': {
        'myCAF': ['ACTA2', 'TAGLN', 'POSTN'],
        'iCAF': ['CXCL12', 'CFD', 'IL6'],
        'apCAF': ['HLA-DRA', 'CD74']
    },
    'T cells': {
        'CD8+ Effector': ['GZMA', 'GZMB', 'PRF1', 'IFNG'],
        'Exhausted': ['PDCD1', 'HAVCR2', 'LAG3', 'TOX', 'CTLA4'],
        'Tregs': ['FOXP3', 'IL2RA'],
        'CD4+ Naive/Memory': ['TCF7', 'IL7R', 'SELL']
    }
}

subtypes = ['Macrophages', 'Fibroblasts', 'Tcells']
file_subtypes = ['Macrophages', 'Fibroblasts', 'T cells']

for st, f_st in zip(subtypes, file_subtypes):
    fine_file = f"{method_dir}/08_{f_st.replace(' ', '')}_adata_sctour_harmony.h5ad"
    if not os.path.exists(fine_file):
        fine_file = f"{method_dir}/07_{f_st.replace(' ', '')}_adata_fine_harmony.h5ad"
            
    if os.path.exists(fine_file):
        print(f"  Generating fine annotation dotplot for {st} ({method_name})...")
        adata_sub = sc.read_h5ad(fine_file)
        
        groupby_col = 'CellType_Fine' if 'CellType_Fine' in adata_sub.obs.columns else None
        
        if groupby_col and st in fine_markers:
            valid_markers = {k: [g for g in v if g in adata_sub.var_names] for k, v in fine_markers[st].items()}
            valid_markers = {k: v for k, v in valid_markers.items() if v}
            
            if valid_markers:
                sc.settings.figdir = method_dir
                sc.pl.dotplot(adata_sub, valid_markers, groupby=groupby_col, standard_scale='var', 
                              save=f"_08_Fine_Annotation_Dotplot_{st}_{method_name}_Fixed.pdf", show=False)
                
                old_name = f"{method_dir}/dotplot__08_Fine_Annotation_Dotplot_{st}_{method_name}_Fixed.pdf"
                new_name = f"{method_dir}/08_Fine_Annotation_Dotplot_{st}_{method_name}_Fixed.pdf"
                if os.path.exists(old_name):
                    os.rename(old_name, new_name)
