import pandas as pd
import glob
import os
import functools

datasets = {
    'Colon': '/mnt/data3/STrnaseq/script/SCbatch/GSE289314 Colon cancer',
    'NSCLC': '/mnt/data3/STrnaseq/script/SCbatch/GSE299111 Non-small cell lung cancer',
}

batch_genes = []
for cancer, path in datasets.items():
    matrix_files = glob.glob(f"{path}/*_matrix.mtx.gz")
    # take just 1 from each
    mat_file = matrix_files[0]
    prefix = mat_file.replace('_matrix.mtx.gz', '')
    feat_file = prefix + '_features.tsv.gz'
    genes = pd.read_csv(feat_file, header=None, sep='\t')
    if genes.shape[1] >= 2:
        var_names = genes[1].values
    else:
        var_names = genes[0].values
    var_names = [str(x).upper() for x in var_names]
    batch_genes.append(set(var_names))

print("Colon sample size:", len(batch_genes[0]))
print("NSCLC sample size:", len(batch_genes[1]))
common = batch_genes[0].intersection(batch_genes[1])
print("Common:", len(common))
if len(common) > 0:
    print("Some common:", list(common)[:10])
else:
    print("No common genes!")
    print("Colon examples:", list(batch_genes[0])[:10])
    print("NSCLC examples:", list(batch_genes[1])[:10])
