import os
try:
    os.rename("results_new_cross_cancer_ccvae/dotplot__04_Broad_Annotation_Dotplot_Fixed_CCVAE.pdf", "results_new_cross_cancer_ccvae/04_Broad_Annotation_Dotplot_Fixed_CCVAE.pdf")
except Exception as e:
    pass
print("Done")
