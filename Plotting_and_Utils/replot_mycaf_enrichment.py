import pandas as pd
import gseapy as gp
import os

out_dir = "results_real_cross_cancer/fibroblast/myCAF_BLCA_vs_HCC"
df_volcano = pd.read_csv(f"{out_dir}/myCAF_BLCA_vs_HCC_DEGs.csv")

up_bladder = df_volcano[df_volcano['Significance'] == 'Up in Bladder myCAF']['Gene'].tolist()
up_hcc = df_volcano[df_volcano['Significance'] == 'Up in HCC myCAF']['Gene'].tolist()

# Let's run enrichr and get all results (not just adj p < 0.05) to see what's there
gene_sets = ['GO_Biological_Process_2021']

enr_bladder = gp.enrichr(gene_list=up_bladder, gene_sets=gene_sets, organism='human', outdir=None)
res_bladder = enr_bladder.results.sort_values('P-value')
res_bladder.to_csv(f"{out_dir}/myCAF_Bladder_GO_Full.csv", index=False)

enr_hcc = gp.enrichr(gene_list=up_hcc, gene_sets=gene_sets, organism='human', outdir=None)
res_hcc = enr_hcc.results.sort_values('P-value')
res_hcc.to_csv(f"{out_dir}/myCAF_HCC_GO_Full.csv", index=False)

print("Saved full GO enrichment results.")
