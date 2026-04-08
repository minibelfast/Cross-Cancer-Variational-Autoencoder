import os

try:
    os.rename("results_new_cross_cancer_ccvae/dotplot__08_Fine_Annotation_Dotplot_Tcells_CCVAE.pdf", 
              "results_new_cross_cancer_ccvae/08_Fine_Annotation_Dotplot_Tcells_CCVAE.pdf")
except Exception as e:
    pass

try:
    os.rename("results_real_cross_cancer_harmony/dotplot__08_Fine_Annotation_Dotplot_Tcells_Harmony.pdf", 
              "results_real_cross_cancer_harmony/08_Fine_Annotation_Dotplot_Tcells_Harmony.pdf")
except Exception as e:
    pass
print("Done")
