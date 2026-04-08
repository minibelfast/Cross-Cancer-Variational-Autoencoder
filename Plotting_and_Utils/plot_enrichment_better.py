import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

out_dir = "results_real_cross_cancer/fibroblast/myCAF_BLCA_vs_HCC"

def plot_better_enrichment(file_path, title, color, out_name):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    if df.empty: return
    
    # Filter highly overlapping or redundant GO terms manually or just take top by P-value
    # Let's take top 10 most significant
    df = df.sort_values('Adjusted P-value').head(10)
    df['-log10(P)'] = -np.log10(df['Adjusted P-value'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='-log10(P)', y='Term', color=color)
    plt.title(title)
    plt.xlabel('-log10(Adjusted P-value)')
    plt.ylabel('')
    
    # Wrap long labels
    import textwrap
    ax = plt.gca()
    labels = [textwrap.fill(label.get_text(), 40) for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{out_name}.pdf", bbox_inches='tight')
    plt.close()

plot_better_enrichment(f"{out_dir}/myCAF_Bladder_Enrichment.csv", 'Top Enriched Pathways in Bladder myCAF', 'lightcoral', 'myCAF_Bladder_Enrichment_Barplot')
plot_better_enrichment(f"{out_dir}/myCAF_HCC_Enrichment.csv", 'Top Enriched Pathways in HCC myCAF', 'skyblue', 'myCAF_HCC_Enrichment_Barplot')

