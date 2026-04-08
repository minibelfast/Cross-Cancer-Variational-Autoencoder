import pandas as pd
import glob
import os
import functools

datasets = {
    'Colon': '/mnt/data3/STrnaseq/script/SCbatch/GSE289314 Colon cancer',
    'NSCLC': '/mnt/data3/STrnaseq/script/SCbatch/GSE299111 Non-small cell lung cancer',
    'GBM': '/mnt/data3/STrnaseq/script/SCbatch/GSE311151 Glioblastoma',
    'PDAC': '/mnt/data3/STrnaseq/script/SCbatch/GSE311788 Pancreatic ductal adenocarcinoma'
}

batch_genes = []
for cancer, path in datasets.items():
    matrix_files = glob.glob(f"{path}/*_matrix.mtx.gz")
    for mat_file in matrix_files:
        prefix = mat_file.replace('_matrix.mtx.gz', '')
        feat_file = prefix + '_features.tsv.gz'
        if not os.path.exists(feat_file):
            feat_file = prefix + '_genes.tsv.gz'
        try:
            genes = pd.read_csv(feat_file, header=None, sep='\t')
            if genes.shape[1] >= 2:
                var_names = genes[1].values
            else:
                var_names = genes[0].values
            var_names = [str(x).upper() for x in var_names]
            batch_genes.append(set(var_names))
        except Exception as e:
            pass

common_genes = list(functools.reduce(set.intersection, batch_genes))
print(f"Found {len(common_genes)} common genes")
