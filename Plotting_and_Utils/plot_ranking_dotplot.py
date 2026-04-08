import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_ranking_dotplot():
    results_dir = "./results_simulation_150"
    metrics_file = os.path.join(results_dir, "simulation_results_150.csv")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
        
    df = pd.read_csv(metrics_file)
    
    # Calculate average scores across all conditions
    methods = ['Uncorrected', 'Harmony', 'scVI', 'BBKNN', 'CC-VAE']
    # Use Silhouette_Bio as ASW
    metrics = ['ARI', 'Silhouette_Bio', 'F1_DEG']
    
    avg_scores = {m: {} for m in methods}
    
    for metric in metrics:
        # Group by method and calculate mean
        metric_means = df.groupby('Method')[metric].mean()
        for m in methods:
            if m in metric_means:
                avg_scores[m][metric] = metric_means[m]
            else:
                avg_scores[m][metric] = np.nan
                
    # Convert to DataFrame for easier ranking
    df_avg = pd.DataFrame(avg_scores).T
    
    # Calculate ranks (1 is best)
    # For ARI, ASW, F1, higher is better
    df_rank = df_avg.rank(ascending=False, method='min')
    
    # Calculate Overall score (average of the 3 ranks)
    df_rank['Overall_Score'] = df_rank[['ARI', 'Silhouette_Bio', 'F1_DEG']].mean(axis=1)
    
    # Rank the overall score (lower score = better rank)
    df_rank['Overall'] = df_rank['Overall_Score'].rank(ascending=True, method='min')
    
    # ** BUG FIX **: Get exactly the sorted methods from Best to Worst
    ordered_methods = df_rank.sort_values('Overall', ascending=True).index.tolist()
    
    # Prepare data for seaborn scatterplot
    plot_data = []
    
    # We want to plot 4 columns: ARI, ASW, F1, Overall
    plot_metrics = ['ARI', 'Silhouette_Bio', 'F1_DEG', 'Overall']
    display_names = ['ARI', 'ASW', 'F1', 'Overall']
    
    for method in ordered_methods:
        for m_idx, metric in enumerate(plot_metrics):
            rank = df_rank.loc[method, metric]
            
            # Map rank to size and color
            # Size: 1->max size, 5->min size
            # Color: 1->Red, 5->Blue
            inverted_rank = 6 - rank # 1 -> 5 (best), 5 -> 1 (worst)
            
            plot_data.append({
                'Method': method,
                'Metric': display_names[m_idx],
                'Rank': rank,
                'Plot_Size': inverted_rank * 100, # Base size multiplier
                'Plot_Color': inverted_rank # Used for cmap
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    # ** BUG FIX **: Force categorical type to strictly lock the Y-axis order in seaborn
    df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=ordered_methods, ordered=True)
    df_plot['Metric'] = pd.Categorical(df_plot['Metric'], categories=display_names, ordered=True)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    ax = sns.scatterplot(
        data=df_plot,
        x='Metric',
        y='Method',
        size='Plot_Size',
        hue='Plot_Color',
        palette='coolwarm', # Blue (low value/worst rank) to Red (high value/best rank)
        sizes=(100, 600),
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Customize axes
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Algorithm Ranking Across All Conditions', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    
    # Put the best method at the TOP of the Y-axis
    ax.invert_yaxis()
    
    # Add rank text inside the dots
    for _, row in df_plot.iterrows():
        plt.text(
            x=row['Metric'],
            y=row['Method'],
            s=f"{int(row['Rank'])}",
            color='white' if row['Plot_Color'] > 3 else 'black',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
        
    plt.tight_layout()
    out_file = os.path.join(results_dir, "Ranking_Dotplot.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    
    print(f"Ranking Dotplot saved to {out_file}")
    
    # Also save the raw averages and ranks for reference
    out_csv = os.path.join(results_dir, "Ranking_Data.csv")
    df_combined = pd.concat([df_avg, df_rank.add_suffix('_Rank')], axis=1)
    df_combined.to_csv(out_csv)
    print(f"Ranking data saved to {out_csv}")

if __name__ == "__main__":
    generate_ranking_dotplot()
