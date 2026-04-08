#!/bin/bash

# Cross-Cancer Integration Pipeline Runner

# Activate conda environment 'st'
source ~/.bashrc
conda activate st

# Run the pipeline
# To run with synthetic data for testing:
# python main_pipeline.py --use_synthetic --max_epochs 50 --n_latent 30 --output_dir ./results_demo

# To run with real data (if present in subdirectories):
echo "Running pipeline with detected real data (or synthetic fallback)..."
python main_pipeline.py --max_epochs 100 --n_latent 30 --output_dir ./results_real

echo "Analysis complete. Check ./results_real for outputs."
