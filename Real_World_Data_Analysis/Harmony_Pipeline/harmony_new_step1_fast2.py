import scanpy as sc
import pandas as pd
import glob
import os
import scanpy.external as sce
import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(6, 6))

out_dir = "results_new_cross_cancer_harmony"
os.makedirs(out_dir, exist_ok=True)
sc.settings.figdir = out_dir

print("Loading raw concat data...")
# Load the raw concat data if available, or try to merge again but efficiently
# It seems the data size is very large.
