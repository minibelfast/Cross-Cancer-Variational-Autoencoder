import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

out_dir = "results_real_cross_cancer/fibroblast/myCAF_BLCA_vs_HCC"

bladder_terms = [
    'extracellular matrix organization (GO:0030198)',
    'regulation of angiogenesis (GO:0045765)',
    'regulation of actin cytoskeleton reorganization (GO:2000249)',
    'regulation of cell adhesion (GO:0030155)',
    'signal transduction by p53 class mediator (GO:0072331)'
]

hcc_terms = [
    'regulation of transforming growth factor beta receptor signaling pathway (GO:0017015)',
    'positive regulation of extracellular matrix assembly (GO:1901203)',
    'regulation of actin cytoskeleton organization (GO:0032956)',
    'regulation of focal adhesion assembly (GO:0051893)',
    'positive regulation of smooth muscle cell migration (GO:0014911)'
]

def plot_curated(file_path, terms, title, color, out_name):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    # Filter for selected terms
    df_sub = df[df['Term'].isin(terms)].copy()
    
    if df_sub.empty:
        print(f"Warning: No terms found for {title}")
        return
        
    df_sub['Term_Clean'] = df_sub['Term'].apply(lambda x: x.split(' (GO:')[0] if ' (GO:' in x else x)
    df_sub['-log10(P)'] = -np.log10(df_sub['P-value'])  # using unadjusted P for display if adj is not significant, but we label it P-value
    df_sub = df_sub.sort_values('-log10(P)', ascending=False)
    
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_sub, x='-log10(P)', y='Term_Clean', color=color)
    plt.title(title)
    plt.xlabel('-log10(P-value)')
    plt.ylabel('')
    
    import textwrap
    ax = plt.gca()
    labels = [textwrap.fill(label.get_text(), 40) for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{out_name}.pdf", bbox_inches='tight')
    plt.close()

plot_curated(f"{out_dir}/myCAF_Bladder_GO_Full.csv", bladder_terms, 'Curated Pathways in Bladder myCAF', 'lightcoral', 'myCAF_Bladder_Curated_Barplot')
plot_curated(f"{out_dir}/myCAF_HCC_GO_Full.csv", hcc_terms, 'Curated Pathways in HCC myCAF', 'skyblue', 'myCAF_HCC_Curated_Barplot')

print("Curated enrichment plots generated.")
