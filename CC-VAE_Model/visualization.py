import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
import os
import datetime

class FigureGenerator:
    """
    Generates publication-quality figures following Nature Methods standards.
    """
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Nature Methods Plotting Standards
        # Width: 89 mm (3.5 in) for single column, 183 mm (7.2 in) for double column
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 7
        plt.rcParams['axes.titlesize'] = 8
        plt.rcParams['axes.labelsize'] = 7
        plt.rcParams['xtick.labelsize'] = 6
        plt.rcParams['ytick.labelsize'] = 6
        plt.rcParams['legend.fontsize'] = 6
        plt.rcParams['figure.titlesize'] = 8
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['lines.linewidth'] = 0.75
        plt.rcParams['axes.linewidth'] = 0.5
        
        sns.set_context("paper", font_scale=1.0)
        sns.set_style("whitegrid", {'axes.grid': False})
        
        self.today = datetime.datetime.now().strftime("%Y%m%d")

    def save_plot_data(self, df, filename_base, metadata=None):
        """
        Save the data used for plotting as a CSV file with metadata.
        """
        csv_path = os.path.join(self.output_dir, f"{filename_base}_data.csv")
        
        # Add metadata as comments at the top
        if metadata:
            with open(csv_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                    
        # Append data
        mode = 'a' if metadata else 'w'
        df.to_csv(csv_path, mode=mode, index=False)
        print(f"Data saved to {csv_path}")

    def plot_figure_2_metrics_summary(self, metrics_df):
        """
        Figure 2: Global Performance Summary (Heatmap/Dotplot).
        """
        filename = f"Figure_2_Global_Performance_Summary_{self.today}"
        
        # Normalize metrics for heatmap (0-1)
        df_norm = metrics_df.copy()
        metrics = [col for col in metrics_df.columns if col != "Method"]
        
        for metric in metrics:
            # Check if metric is "lower is better"
            if "Batch" in metric and "Mixing" not in metric and "Silhouette_Batch" in metric:
                # Silhouette batch: closer to 0 is better. But usually we use Batch Mixing Score (higher better)
                pass
            
            # Min-Max Normalization
            min_val = df_norm[metric].min()
            max_val = df_norm[metric].max()
            if max_val > min_val:
                df_norm[metric] = (df_norm[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[metric] = 0.5
                
        # Melt for plotting
        df_melt = df_norm.melt(id_vars=["Method"], var_name="Metric", value_name="Normalized Score")
        
        # Plot
        fig, ax = plt.subplots(figsize=(7.2, 4))
        
        # Dotplot-like heatmap
        sns.heatmap(df_norm.set_index("Method")[metrics], 
                    annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Normalized Score'}, ax=ax)
        
        ax.set_title("Global Performance Summary (Normalized)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        
        # Save Data
        meta = {
            "Description": "Normalized performance metrics for all methods",
            "Metrics": ", ".join(metrics),
            "Normalization": "Min-Max scaling per metric"
        }
        self.save_plot_data(metrics_df, filename, meta)

    def plot_figure_3_detailed_metrics(self, metrics_df):
        """
        Figure 3: Detailed Metrics Bar Plots.
        """
        filename = f"Figure_3_Detailed_Metrics_{self.today}"
        
        metrics = [col for col in metrics_df.columns if col != "Method"]
        n_metrics = len(metrics)
        
        # Use double column width
        fig, axes = plt.subplots(1, n_metrics, figsize=(7.2, 2.5))
        if n_metrics == 1:
            axes = [axes]
            
        palette = sns.color_palette("viridis", n_colors=len(metrics_df['Method'].unique()))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(data=metrics_df, x="Method", y=metric, ax=ax, palette=palette, errorbar=None)
            ax.set_title(metric, fontsize=8, fontweight='bold')
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.5)
            
            # Add values on top
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.2f}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='bottom', fontsize=5, xytext=(0, 2), 
                                textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        
        # Save Data
        meta = {
            "Description": "Raw performance metrics for all methods",
            "Metrics": ", ".join(metrics)
        }
        self.save_plot_data(metrics_df, filename, meta)

    def plot_figure_4_umaps(self, adata_dict, color_keys=['batch', 'cell_type']):
        """
        Figure 4: UMAP Visualizations.
        """
        filename = f"Figure_4_Integrated_Embeddings_{self.today}"
        
        n_methods = len(adata_dict)
        n_keys = len(color_keys)
        
        # Figure size: Double column width (7.2 in), height depends on rows
        fig_width = 7.2
        fig_height = 2.0 * n_methods
        
        fig, axes = plt.subplots(n_methods, n_keys, figsize=(fig_width, fig_height))
        
        # Handle axes array shape
        if n_methods == 1 and n_keys == 1:
            axes = np.array([[axes]])
        elif n_methods == 1:
            axes = np.array([axes])
        elif n_keys == 1:
            axes = np.array([[ax] for ax in axes])
            
        plot_data_list = []
            
        for i, (method, adata) in enumerate(adata_dict.items()):
            # Collect data for CSV
            umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
            umap_df['Method'] = method
            for key in color_keys:
                if key in adata.obs:
                    umap_df[key] = adata.obs[key].values
            plot_data_list.append(umap_df)

            for j, key in enumerate(color_keys):
                ax = axes[i, j]
                if key in adata.obs:
                    sc.pl.umap(adata, color=key, ax=ax, show=False, title=f"{method}", frameon=False, legend_fontsize=6)
                    ax.set_xlabel("UMAP1")
                    ax.set_ylabel("UMAP2")
                    if i == 0:
                        ax.set_title(f"{method}\n{key}", fontsize=8, fontweight='bold')
                    else:
                        ax.set_title(f"{method}", fontsize=8)
                else:
                    ax.text(0.5, 0.5, f"N/A: {key}", ha='center', va='center')
                    ax.axis('off')
                    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        
        # Save Data (Subsampled if too large to avoid huge CSVs)
        full_df = pd.concat(plot_data_list)
        if len(full_df) > 100000:
             # Subsample for CSV export to keep it manageable
             full_df = full_df.sample(100000)
             
        meta = {
            "Description": "UMAP coordinates and metadata for integrated embeddings",
            "Note": "Data may be subsampled to 100k rows for file size limits"
        }
        self.save_plot_data(full_df, filename, meta)

    def plot_figure_5_confusion_matrix(self, adata, true_label, pred_label, method_name):
        """
        Figure 5: Confusion Matrix.
        """
        filename = f"Figure_5_Confusion_Matrix_{method_name}_{self.today}"
        
        from sklearn.metrics import confusion_matrix
        
        if true_label not in adata.obs or pred_label not in adata.obs:
            return
            
        y_true = adata.obs[true_label]
        y_pred = adata.obs[pred_label]
        
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=sorted(y_true.unique()), yticklabels=sorted(y_true.unique()))
        plt.title(f"Confusion Matrix - {method_name}", fontweight='bold')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        plt.savefig(os.path.join(self.output_dir, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        
        # Save Data
        cm_df = pd.DataFrame(cm, index=sorted(y_true.unique()), columns=sorted(y_true.unique()))
        meta = {
            "Description": f"Normalized confusion matrix for {method_name}",
            "True Label": true_label,
            "Predicted Label": pred_label
        }
        self.save_plot_data(cm_df.reset_index().rename(columns={'index': 'True_Label'}), filename, meta)
