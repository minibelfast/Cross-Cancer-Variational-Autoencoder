import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(8, 6))

methods = {
    'CCVAE': 'results_new_cross_cancer_ccvae',
    'Harmony': 'results_real_cross_cancer_harmony'
}

# Define classic markers used for broad and fine annotation
broad_markers = {
    'T cells': ['CD3D', 'CD3E', 'CD2'],
    'Macrophages': ['CD68', 'CD163', 'C1QA', 'MARCO'],
    'Fibroblasts': ['COL1A1', 'DCN', 'LUM', 'THY1'],
    'Epithelial/Tumor': ['EPCAM', 'KRT8', 'KRT18'],
    'Endothelial': ['PECAM1', 'VWF', 'ENG'],
    'B cells': ['CD79A', 'MS4A1']
}

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

for method_name, method_dir in methods.items():
    print(f"\nProcessing {method_name} in {method_dir}...")
    if not os.path.exists(method_dir):
        print(f"Directory {method_dir} not found. Skipping.")
        continue
        
    # 1. Broad Annotation Dotplot
    broad_file = f"{method_dir}/integrated_adata_annotated_{method_name.lower()}.h5ad"
    if method_name == 'Harmony':
        broad_file = f"{method_dir}/integrated_adata_annotated_harmony.h5ad"
        
    if os.path.exists(broad_file):
        print(f"  Generating broad annotation dotplot for {method_name}...")
        adata = sc.read_h5ad(broad_file)
        
        groupby_col = 'CellType_Broad' if 'CellType_Broad' in adata.obs.columns else ('CellType' if 'CellType' in adata.obs.columns else None)
        
        if groupby_col:
            # Filter broad markers
            valid_broad = {k: [g for g in v if g in adata.var_names] for k, v in broad_markers.items()}
            valid_broad = {k: v for k, v in valid_broad.items() if v}
            
            if valid_broad:
                sc.pl.dotplot(adata, valid_broad, groupby=groupby_col, standard_scale='var', 
                              save=f"_Broad_Annotation_{method_name}.pdf", show=False)
                if os.path.exists(f"figures/dotplot__Broad_Annotation_{method_name}.pdf"):
                    os.rename(f"figures/dotplot__Broad_Annotation_{method_name}.pdf", f"{method_dir}/04_Broad_Annotation_Dotplot_{method_name}.pdf")
    else:
        print(f"  {broad_file} not found.")

    # 2. Fine Annotation Dotplots
    subtypes = ['Macrophages', 'Fibroblasts', 'Tcells']
    file_subtypes = ['Macrophages', 'Fibroblasts', 'T cells']
    
    for st, f_st in zip(subtypes, file_subtypes):
        fine_file = f"{method_dir}/08_{st}_adata_fine_{method_name.lower()}.h5ad"
        if not os.path.exists(fine_file):
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
                    sc.pl.dotplot(adata_sub, valid_markers, groupby=groupby_col, standard_scale='var', 
                                  save=f"_Fine_Annotation_{st}_{method_name}.pdf", show=False)
                    if os.path.exists(f"figures/dotplot__Fine_Annotation_{st}_{method_name}.pdf"):
                        os.rename(f"figures/dotplot__Fine_Annotation_{st}_{method_name}.pdf", 
                                  f"{method_dir}/08_Fine_Annotation_Dotplot_{st}_{method_name}.pdf")
        else:
            print(f"  {fine_file} not found.")

print("Done!")
