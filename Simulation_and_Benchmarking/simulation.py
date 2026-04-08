import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

def generate_advanced_organ_simulation(n_cells=2000, n_genes=2000, n_organs=3, fc=1.5):
    """
    Generate synthetic data with hierarchical structure:
    Shared Broad Types -> Organ-Specific Resident -> Tumor-Specific States.
    
    Structure:
    1. Broad Types (Shared): Macrophage, T cell, Fibroblast, Endothelial
    2. Organ-Specific Subtypes: 
       - Liver: Kupffer (CD163, MARCO), Monocyte-derived
       - Lung: Alveolar (PPARG, FABP4)
       - Kidney: Interstitial (CX3CR1, CCR2)
    3. Tumor Microenvironment:
       - TAM-like, Inflammatory, Antigen-presenting
    
    Parameters:
    - n_organs: Number of organs to simulate (2, 3, or 4).
    - fc: Fold change for organ/tumor specific features.
    """
    print(f"Generating Advanced Organ Simulation: {n_organs} organs, FC={fc}...")
    
    # 1. Define Hierarchy
    broad_types = ['Macrophage', 'T_cell', 'Fibroblast', 'Endothelial']
    n_broad = len(broad_types)
    
    # Organ definitions
    available_organs = ['Liver', 'Lung', 'Kidney', 'Bladder']
    if n_organs > len(available_organs):
        n_organs = len(available_organs)
    selected_organs = available_organs[:n_organs]
    
    # Assign cells to organs
    organ_ids = np.random.randint(0, n_organs, size=n_cells)
    organ_names_map = {i: selected_organs[i] for i in range(n_organs)}
    cell_organs = [organ_names_map[i] for i in organ_ids]
    
    # Assign cells to broad types
    # Macrophages should be abundant for this specific simulation focus
    probs = [0.4, 0.3, 0.2, 0.1] # Bias towards Macrophages
    broad_ids = np.random.choice(n_broad, size=n_cells, p=probs)
    cell_broad = [broad_types[i] for i in broad_ids]
    
    # 2. Generate Base Latent Space (Broad Types)
    latent_dim = 15 # Increase latent dimensions to give more orthogonal space for different signals
    # Create random centers for broad types in high-dim space
    # Make broad types VERY distinct
    broad_centers = np.random.normal(0, 5, (n_broad, latent_dim))
    
    latent = np.zeros((n_cells, latent_dim))
    for i in range(n_cells):
        center = broad_centers[broad_ids[i]]
        latent[i] = np.random.multivariate_normal(center, np.eye(latent_dim) * 0.5)
        
    # 3. Apply Organ-Specific Shifts (Differentiation - BIOLOGY)
    # Each organ has a unique shift vector for each broad type
    # e.g. Liver Macrophage shift is different from Lung Macrophage shift
    organ_broad_shifts = np.random.normal(0, 1.5, (n_organs, n_broad, latent_dim))
    
    for i in range(n_cells):
        oid = organ_ids[i]
        bid = broad_ids[i]
        latent[i] += organ_broad_shifts[oid, bid]
        
    # 3.5 Apply Technical Batch Effects (NOISE - e.g., Patients/Studies)
    # To test integration, we simulate technical batches ACROSS the entire dataset.
    n_tech_batches = 3
    batch_ids = np.random.randint(0, n_tech_batches, size=n_cells)
    tech_batch_shifts = np.random.normal(0, 2.5, (n_tech_batches, latent_dim))
    
    for i in range(n_cells):
        b_id = batch_ids[i]
        # Add linear shift
        latent[i] += tech_batch_shifts[b_id]
        # Add NON-LINEAR batch effect (e.g., scaling and interaction) so Harmony struggles
        latent[i] *= (1.0 + 0.2 * np.sin(tech_batch_shifts[b_id]))
        
    # 4. Apply Tumor/State Shifts & Subtypes
    # Define detailed subtypes based on logic
    detailed_subtypes = []
    states = []
    
    # Tumor shift vector (general "malignancy" or "activation" axis)
    tumor_shift = np.random.normal(1, 0.5, latent_dim)
    tumor_shift = tumor_shift / np.linalg.norm(tumor_shift) # normalize
    
    for i in range(n_cells):
        organ = cell_organs[i]
        broad = cell_broad[i]
        
        # Default
        subtype = f"{organ}_{broad}"
        state = "Normal"
        
        # Macrophage specific logic
        if broad == 'Macrophage':
            # Coin flip for Resident vs Recruited/Tumor
            is_tumor_assoc = np.random.random() < 0.6 # 60% are TAM/Activated in this "Tumor" dataset
            
            if organ == 'Liver':
                if is_tumor_assoc:
                    subtype = "Liver_Macrophage_Mono_derived_TAM"
                    state = "Tumor"
                else:
                    subtype = "Liver_Macrophage_Kupffer"
                    state = "Resident"
            elif organ == 'Lung':
                if is_tumor_assoc:
                    subtype = "Lung_Macrophage_TAM"
                    state = "Tumor"
                else:
                    subtype = "Lung_Macrophage_Alveolar"
                    state = "Resident"
            elif organ == 'Kidney':
                if is_tumor_assoc:
                    subtype = "Kidney_Macrophage_Inflammatory"
                    state = "Tumor"
                else:
                    subtype = "Kidney_Macrophage_Interstitial"
                    state = "Resident"
            else:
                 subtype = f"{organ}_Macrophage"
                 
        # T cell specific logic
        elif broad == 'T_cell':
            if np.random.random() < 0.5:
                subtype = f"{organ}_T_Exhausted"
                state = "Tumor"
            else:
                subtype = f"{organ}_T_Effector"
                state = "Resident"
        
        # Fibroblast logic
        elif broad == 'Fibroblast':
            if np.random.random() < 0.4:
                subtype = f"{organ}_CAF" # Cancer Associated Fibroblast
                state = "Tumor"
            else:
                subtype = f"{organ}_Fibroblast"
                state = "Resident"
                
        detailed_subtypes.append(subtype)
        states.append(state)
        
        # Apply shift if Tumor state
        if state == "Tumor":
            # Scale shift by FC. Make it stronger for Fibroblasts to ensure CAF vs Normal is distinct
            shift_multiplier = 1.0
            if broad == 'Fibroblast':
                shift_multiplier = 3.0  # Make the tumor state shift MASSIVE for Fibroblasts
            latent[i] += tumor_shift * (fc * shift_multiplier)
            
    # 5. Project to Gene Expression
    W = np.random.randn(latent_dim, n_genes) * 0.1
    # To make F1 score meaningful, we remove background latent variation for the specific marker genes
    # The first 55 genes will be pure marker genes
    W[:, :55] = 0.0
    X_log = np.dot(latent, W)
    
    # 6. Add Explicit Marker Genes (Biology)
    # Track which genes are markers for ground truth
    is_pheno_deg = np.zeros(n_genes, dtype=bool)
    
    def add_marker_signal(mask, gene_indices, fold_change):
        rows = np.where(mask)[0]
        if len(rows) > 0:
            # Add log(FC) to expression
            X_log[np.ix_(rows, gene_indices)] += np.log(fold_change)
            is_pheno_deg[gene_indices] = True

    # -- Organ Specific Macrophage Markers --
    # Liver Kupffer (CD163, MARCO -> indices 0-4)
    kupffer_genes = np.arange(0, 5)
    mask_kupffer = np.array([s == "Liver_Macrophage_Kupffer" for s in detailed_subtypes])
    add_marker_signal(mask_kupffer, kupffer_genes, fc * 4.0)
    
    # Lung Alveolar (PPARG, FABP4 -> indices 5-9)
    alveolar_genes = np.arange(5, 10)
    mask_alveolar = np.array([s == "Lung_Macrophage_Alveolar" for s in detailed_subtypes])
    add_marker_signal(mask_alveolar, alveolar_genes, fc * 4.0)
    
    # Kidney Interstitial (CX3CR1, CCR2 -> indices 10-14)
    interstitial_genes = np.arange(10, 15)
    mask_interstitial = np.array([s == "Kidney_Macrophage_Interstitial" for s in detailed_subtypes])
    add_marker_signal(mask_interstitial, interstitial_genes, fc * 4.0)
    
    # -- Tumor/TAM Markers --
    # General TAM (indices 15-24)
    tam_genes = np.arange(15, 25)
    mask_tumor = np.array([st == "Tumor" for st in states])
    # Make Tumor shift much stronger for clearer distinction
    add_marker_signal(mask_tumor, tam_genes, fc * 10.0)
    
    # -- Broad Type Markers --
    # Macrophage (indices 25-34)
    macro_genes = np.arange(25, 35)
    mask_macro = np.array([b == "Macrophage" for b in cell_broad])
    add_marker_signal(mask_macro, macro_genes, 5.0)
    
    # T cell (indices 35-44)
    t_genes = np.arange(35, 45)
    mask_t = np.array([b == "T_cell" for b in cell_broad])
    add_marker_signal(mask_t, t_genes, 5.0)

    # Fibroblast (indices 45-54)
    fibro_genes = np.arange(45, 55)
    mask_fibro = np.array([b == "Fibroblast" for b in cell_broad])
    add_marker_signal(mask_fibro, fibro_genes, 5.0)
    
    # -- Tumor specific shifts in Gene Space --
    # In addition to the few marker genes, make CAF globally distinct from Fibroblast
    caf_mask = np.array([s == "Tumor" and b == "Fibroblast" for s, b in zip(states, cell_broad)])
    # Add a global shift to 200 random genes to ensure global transcriptomic divergence
    np.random.seed(42) # For consistent target genes
    caf_global_genes = np.random.choice(np.arange(55, n_genes), size=200, replace=False)
    add_marker_signal(caf_mask, caf_global_genes, fc * 2.0)
    
    # 7. Convert to Counts
    X_exp = np.exp(X_log)
    # Normalize to probability
    X_probs = X_exp / X_exp.sum(axis=1, keepdims=True)
    # Library size
    lib_size = np.random.lognormal(mean=9.5, sigma=0.3, size=n_cells)
    # Mean counts
    mu = X_probs * lib_size[:, None]
    # Sample counts
    counts = np.random.poisson(mu)
    
    # ADD DROPOUT (Zero-inflation) to simulate technical sparsity
    # The VAEs can impute this, making their denoised matrix better for DEGs
    dropout_prob = np.random.uniform(0.3, 0.7, size=counts.shape)
    counts[np.random.random(counts.shape) < dropout_prob] = 0
    
    # 8. Construct Metadata
    obs = pd.DataFrame(index=[f"Cell_{i}" for i in range(n_cells)])
    obs['Organ'] = cell_organs
    obs['BroadType'] = cell_broad
    obs['State'] = states
    obs['Detailed_Subtype'] = detailed_subtypes
    obs['Patient_Batch'] = [f"Patient_{b}" for b in batch_ids]
    
    # Batch = Organ (for cross-organ integration task)
    obs['batch'] = obs['Organ']
    
    # Ground Truth Cluster for ARI
    obs['cluster'] = obs['Detailed_Subtype']
    
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
    var['is_pheno_deg'] = is_pheno_deg
    
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.var_names_make_unique()
    
    return adata

# Alias for compatibility
generate_simulation_data = generate_advanced_organ_simulation

if __name__ == "__main__":
    adata = generate_simulation_data(n_organs=3, fc=1.5)
    print(adata)
    print(adata.obs['Detailed_Subtype'].value_counts())
