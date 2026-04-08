# CC-VAE: Cross-Cancer Variational Autoencoder for Pan-Cancer Single-Cell Integration

This repository contains the official implementation of **CC-VAE**, a generative deep learning framework designed to integrate cross-cancer single-cell RNA sequencing (scRNA-seq) data while preserving cancer-specific biological heterogeneity. The repository also provides the complete analytical pipeline used to uncover the pan-cancer tumor microenvironment (TME) pathogenic axes, including TAM polarization, CAF plasticity, and T cell exhaustion.

## 📁 Repository Structure

The codebase is logically organized into the following directories:

*   **`CC-VAE_Model/`**: Core implementation of the CC-VAE framework.
    *   `model.py`: PyTorch implementation of the CC-VAE neural network architecture.
    *   `main_pipeline.py` & `data_loader.py`: High-level APIs for data loading, model training, and latent space extraction.
    *   `evaluation.py` & `evaluation_advanced.py`: Metrics for evaluating batch correction and biological preservation.
*   **`Simulation_and_Benchmarking/`**: Scripts for benchmarking CC-VAE against other integration tools.
    *   `simulation.py` & `run_simulation*.py`: Generates hierarchical synthetic datasets with known ground-truth labels.
    *   `integration_benchmarks.py`: Evaluates ARI, ASW, and DEG recovery scores.
*   **`Real_World_Data_Analysis/`**: Scripts for reproducing the pan-cancer scRNA-seq analysis.
    *   `CC-VAE_Pipeline/`: Step-by-step pipeline utilizing CC-VAE (Integration -> Annotation -> Subclustering).
    *   `Harmony_Pipeline/`: Baseline integration pipeline using Harmony for comparative analysis.
    *   `Advanced_TME_Analysis/`: Deep TME profiling (e.g., `nature_medicine_deep_tme.py`, `fibroblast_tf_analysis.py`).
*   **`Plotting_and_Utils/`**: Utility scripts to generate all main and supplementary figures (dotplots, heatmaps, sankey diagrams).
*   **`Other_Scripts/`**: Miscellaneous test scripts and case studies.

## 💾 Datasets

This study re-analyzes publicly available datasets. Raw data files are not included in this repository due to size limits, but can be downloaded from the respective databases using the accession numbers below:

1.  **Primary scRNA-seq Discovery Cohort (GEO)**:
    *   ccRCC (GSE242299), BLCA (GSE277524), HCC (GSE290925), BRCA (GSE292824)
2.  **scRNA-seq Validation Cohort (GEO)**:
    *   CRC (GSE289314), NSCLC (GSE299111), GBM (GSE311151), PDAC (GSE311788)
3.  **Bulk Transcriptomic & Survival Cohorts**:
    *   **GEO**: 34 curated cohorts (e.g., GSE154261, GSE31684, GSE14520, GSE54236).
    *   **TCGA/GDC**: Pan-cancer bulk RNA-seq and clinical data.
    *   **ICGC**: LIRI and EU cohorts.
    *   **IMvigor210**: Atezolizumab metastatic urothelial carcinoma cohort (`IMvigor210CoreBiologies`).

## 🚀 How to Apply CC-VAE to New Data

To integrate your own scRNA-seq datasets using CC-VAE, follow these steps:

```python
import scanpy as sc
from CC_VAE_Model.model import CCVAE
from CC_VAE_Model.data_loader import prepare_data

# 1. Load your raw count AnnData object
adata = sc.read_h5ad("your_dataset.h5ad")

# 2. Prepare data (setup batch key and categorical covariates)
adata_prepared = prepare_data(adata, batch_key="Cancer_Type")

# 3. Initialize and train the model
model = CCVAE(input_dim=adata.n_vars, batch_dim=len(adata.obs["Cancer_Type"].unique()))
model.train(adata_prepared, epochs=100)

# 4. Extract the batch-corrected latent representation
adata.obsm["X_ccvae"] = model.get_latent_representation(adata_prepared)

# 5. Downstream Scanpy analysis
sc.pp.neighbors(adata, use_rep="X_ccvae")
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

## 📊 Reproducing the Results and Generating Figures

To reproduce the findings from the manuscript, execute the scripts in the `Real_World_Data_Analysis` and `Plotting_and_Utils` directories sequentially.

### Step 1: Pan-Cancer Integration & Broad Annotation
*   **Script**: `Real_World_Data_Analysis/CC-VAE_Pipeline/ccvae_new_step1_load.py` & `ccvae_new_step2_annotate.py`
*   **Output**: Generates the integrated UMAP and broad cell-type annotations.
*   **Figure Generation**: Use `Plotting_and_Utils/generate_annotation_dotplots.py` to plot the expression of canonical marker genes across broad lineages.

### Step 2: Subpopulation Trajectory & Pathway Analysis
*   **Scripts**: `ccvae_new_step3_tcells.py`, `ccvae_new_step4_macrophages.py`, `ccvae_new_step5_fibroblasts.py`.
*   **Action**: Subclusters major lineages. Uses scTour and Monocle2 (`run_monocle2.R`) to infer developmental pseudotime.
*   **Figure Generation**:
    *   `Plotting_and_Utils/plot_pseudotime_violin.py`: Visualizes gene dynamics along trajectories.
    *   `Plotting_and_Utils/plot_pathway_dotplot_pvalue_or_ccvae.py`: Generates the pathway enrichment dotplots (Hypoxia, Angiogenesis, Immunosuppression, T cell exhaustion).

### Step 3: Deep TME Profiling (Nature Medicine Style Figures)
*   **Script**: `Real_World_Data_Analysis/Advanced_TME_Analysis/nature_medicine_deep_tme.py`
*   **Action**: Computes the Clinical Functional Triad (CAF Remodeling vs. TAM Suppression vs. T Cell Exhaustion).
*   **Figure Generation**: Outputs multi-point scatter plots mapping patient-level functional scores across cancer types.

*   **Script**: `Real_World_Data_Analysis/Advanced_TME_Analysis/fibroblast_tf_analysis.py`
*   **Action**: Performs transcription factor regulatory network analysis using pseudotime-binned expression variance.
*   **Figure Generation**: Outputs the Master Regulator Matrix Plots and Correlation Heatmaps for CAF plasticity.

## 🛠 Dependencies
*   Python >= 3.8
*   PyTorch >= 1.9.0
*   Scanpy >= 1.9.1
*   scTour (for trajectory inference)
*   R >= 4.2 (with `Seurat` and `Monocle2` for branched trajectory analysis)

