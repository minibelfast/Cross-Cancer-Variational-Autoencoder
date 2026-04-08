import pandas as pd
import plotly.graph_objects as go
import os

def plot_sankey(n_organs, fc):
    print(f"Plotting Sankey diagram for Organs={n_organs}, FC={fc}...")
    
    data_dir = "./exported_simulation_data"
    prefix = f"Organs{n_organs}_FC{fc}"
    
    # Load the subcluster metadata
    meta_path = os.path.join(data_dir, f"{prefix}_Fibroblast_Subcluster_Metadata.csv")
    if not os.path.exists(meta_path):
        print(f"Metadata file not found: {meta_path}")
        return
        
    df = pd.read_csv(meta_path, index_col=0)
    
    # We want to compare Subcluster_Harmony and Subcluster_CC-VAE
    col_harmony = 'Subcluster_Harmony'
    col_ccvae = 'Subcluster_CC-VAE'
    col_state = 'Fibroblast_vs_CAF' # Useful for color or labeling
    
    if col_harmony not in df.columns or col_ccvae not in df.columns:
        print("Required cluster columns not found in metadata.")
        return
        
    # Create nodes
    # Source nodes: Harmony clusters
    harmony_clusters = [f"Harmony_C{c}" for c in sorted(df[col_harmony].unique())]
    # Target nodes: CC-VAE clusters
    ccvae_clusters = [f"CC-VAE_C{c}" for c in sorted(df[col_ccvae].unique())]
    
    all_nodes = harmony_clusters + ccvae_clusters
    node_indices = {name: i for i, name in enumerate(all_nodes)}
    
    # Define Nature Medicine style color palettes (8 colors each)
    # Palette 1: Muted, professional tones for Harmony
    harmony_palette = [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488', 
        '#F39B7F', '#8491B4', '#91D1C2', '#DC0000'
    ]
    # Palette 2: Complementary tones for CC-VAE
    ccvae_palette = [
        '#7E6148', '#B09C85', '#6F99AD', '#FFDC91',
        '#EE4C97', '#E18727', '#20854E', '#BC3C29'
    ]
    
    # Pad palettes if there are more than 8 clusters (unlikely but safe)
    while len(harmony_palette) < len(harmony_clusters):
        harmony_palette += harmony_palette
    while len(ccvae_palette) < len(ccvae_clusters):
        ccvae_palette += ccvae_palette
        
    node_colors = harmony_palette[:len(harmony_clusters)] + ccvae_palette[:len(ccvae_clusters)]
    
    # Calculate flows
    # We will aggregate by Harmony cluster -> CC-VAE cluster
    flow_df = df.groupby([col_harmony, col_ccvae]).size().reset_index(name='count')
    
    source_indices = []
    target_indices = []
    values = []
    link_colors = []
    
    # Function to convert hex to rgba with opacity
    def hex_to_rgba(hex_color, opacity=0.4):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {opacity})'
    
    for _, row in flow_df.iterrows():
        src_name = f"Harmony_C{row[col_harmony]}"
        tgt_name = f"CC-VAE_C{row[col_ccvae]}"
        
        src_idx = node_indices[src_name]
        source_indices.append(src_idx)
        target_indices.append(node_indices[tgt_name])
        values.append(row['count'])
        
        # Use source node color for the link but with transparency
        link_colors.append(hex_to_rgba(node_colors[src_idx], opacity=0.3))
        
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = all_nodes,
          color = node_colors
        ),
        link = dict(
          source = source_indices,
          target = target_indices,
          value = values,
          color = link_colors
      ))])
    
    fig.update_layout(title_text=f"Subcluster mapping between Harmony and CC-VAE (Organs={n_organs}, FC={fc})", font_size=10)
    
    out_file = os.path.join(data_dir, f"Sankey_Harmony_vs_CCVAE_{prefix}.html")
    fig.write_html(out_file)
    print(f"Saved interactive Sankey diagram to {out_file}")
    
    # Try to save as PDF if kaleido is installed
    try:
        pdf_file = os.path.join(data_dir, f"Sankey_Harmony_vs_CCVAE_{prefix}.pdf")
        fig.write_image(pdf_file)
        print(f"Saved PDF to {pdf_file}")
    except Exception as e:
        print("Could not save as PDF. Install 'kaleido' via pip to enable PDF export for Plotly.")
        
if __name__ == "__main__":
    plot_sankey(4, 1.5)
    plot_sankey(4, 2.0)
