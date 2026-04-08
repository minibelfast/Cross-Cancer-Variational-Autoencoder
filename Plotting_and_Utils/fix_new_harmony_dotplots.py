import scanpy as sc
import os

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(10, 6))

methods = {
    'Harmony': 'results_new_cross_cancer_harmony' # Changed to the correct directory name
}

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
    
    # 1. Broad Annotation Dotplot
    broad_file = f"{method_dir}/integrated_adata_annotated_harmony.h5ad"
    if not os.path.exists(broad_file):
        broad_file = f"{method_dir}/integrated_adata_harmony.h5ad" # Fallback if annotated doesn't exist
        
    if os.path.exists(broad_file):
        print(f"  Generating broad annotation dotplot...")
        adata = sc.read_h5ad(broad_file)
        groupby_col = 'CellType_Broad' if 'CellType_Broad' in adata.obs.columns else ('CellType' if 'CellType' in adata.obs.columns else None)
        
        if groupby_col:
            if hasattr(adata, 'raw') and adata.raw is not None:
                adata_plot = adata.raw.to_adata()
                adata_plot.obs[groupby_col] = adata.obs[groupby_col]
            elif 'counts' in adata.layers:
                adata_plot = adata.copy()
                adata_plot.X = adata_plot.layers['counts']
            else:
                adata_plot = adata
                
            valid_broad = {k: [g for g in v if g in adata_plot.var_names] for k, v in broad_markers.items()}
            valid_broad = {k: v for k, v in valid_broad.items() if v}
            
            if valid_broad:
                sc.settings.figdir = method_dir
                sc.pl.dotplot(adata_plot, valid_broad, groupby=groupby_col, standard_scale='var', 
                              save=f"_04_Broad_Annotation_Dotplot_Fixed_{method_name}.pdf", show=False)
                
                old_name = f"{method_dir}/dotplot__04_Broad_Annotation_Dotplot_Fixed_{method_name}.pdf"
                new_name = f"{method_dir}/04_Broad_Annotation_Dotplot_Fixed_{method_name}.pdf"
                if os.path.exists(old_name):
                    os.rename(old_name, new_name)
    else:
        print(f"  Broad file not found: {broad_file}")

    # 2. Fine Annotation Dotplots
    subtypes = ['Macrophages', 'Fibroblasts', 'Tcells']
    file_subtypes = ['Macrophages', 'Fibroblasts', 'T cells']
    
    for st, f_st in zip(subtypes, file_subtypes):
        # Try different naming conventions that might exist in this directory
        fine_files = [
            f"{method_dir}/08_{f_st.replace(' ', '')}_adata_fine_harmony.h5ad",
            f"{method_dir}/08_{st}_adata_fine_harmony.h5ad",
            f"{method_dir}/07_{f_st.replace(' ', '')}_adata_fine_harmony.h5ad",
            f"{method_dir}/07_{st}_adata_fine_harmony.h5ad",
            f"{method_dir}/05_{st}_adata_harmony.h5ad",
            f"{method_dir}/05_{f_st.replace(' ', '')}_adata_harmony.h5ad"
        ]
        
        fine_file = None
        for f in fine_files:
            if os.path.exists(f):
                fine_file = f
                break
                
        if fine_file:
            print(f"  Generating fine annotation dotplot for {st}...")
            adata_sub = sc.read_h5ad(fine_file)
            
            groupby_col = 'CellType_Fine' if 'CellType_Fine' in adata_sub.obs.columns else None
            
            if groupby_col and st in fine_markers:
                if hasattr(adata_sub, 'raw') and adata_sub.raw is not None:
                    adata_plot = adata_sub.raw.to_adata()
                    adata_plot.obs[groupby_col] = adata_sub.obs[groupby_col]
                else:
                    adata_plot = adata_sub
                    
                valid_markers = {k: [g for g in v if g in adata_plot.var_names] for k, v in fine_markers[st].items()}
                valid_markers = {k: v for k, v in valid_markers.items() if v}
                
                if valid_markers:
                    sc.settings.figdir = method_dir
                    sc.pl.dotplot(adata_plot, valid_markers, groupby=groupby_col, standard_scale='var', 
                                  save=f"_08_Fine_Annotation_Dotplot_{st}_{method_name}_Fixed.pdf", show=False)
                    
                    old_name = f"{method_dir}/dotplot__08_Fine_Annotation_Dotplot_{st}_{method_name}_Fixed.pdf"
                    new_name = f"{method_dir}/08_Fine_Annotation_Dotplot_{st}_{method_name}_Fixed.pdf"
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)
        else:
            print(f"  Fine file for {st} not found.")

print("Done!")
