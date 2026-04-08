import scvi
import anndata
import numpy as np
import torch
import os

class CrossCancerVAE:
    """
    Cross-Cancer Variational Autoencoder (CC-VAE) for integrating scRNA-seq data 
    from multiple cancer types.
    
    Wraps scvi-tools VAE implementation with custom training and inference logic
    tailored for multi-domain adaptation.
    """
    
    def __init__(self, adata, batch_key="batch", n_latent=20, n_layers=2, dropout_rate=0.1):
        """
        Initialize the CC-VAE model.
        
        Args:
            adata (anndata.AnnData): Annotated data matrix.
            batch_key (str): Key in adata.obs for batch/cancer type information.
            n_latent (int): Dimensionality of the latent space.
            n_layers (int): Number of hidden layers in encoder/decoder.
            dropout_rate (float): Dropout rate.
        """
        self.adata = adata
        self.batch_key = batch_key
        
        # Setup AnnData for scVI
        # Use batch_key for conditional generation (batch correction)
        scvi.model.SCVI.setup_anndata(
            self.adata, 
            layer="counts", 
            batch_key=self.batch_key
        )
        
        # Initialize the VAE model
        self.model = scvi.model.SCVI(
            self.adata, 
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion="gene", # Use gene dispersion to avoid over-smoothing batch specific genes
            gene_likelihood="zinb"
        )
        
    def train(self, max_epochs=100, batch_size=128, early_stopping=True):
        """
        Train the model.
        """
        print(f"Training CC-VAE on {self.adata.n_obs} cells from {len(self.adata.obs[self.batch_key].unique())} cancer types...")
        # Use a lower learning rate and more patience for better convergence on weak signals
        plan_kwargs = {"lr": 1e-3, "weight_decay": 1e-5} # Reduced weight decay slightly
        self.model.train(
            max_epochs=max_epochs, 
            batch_size=batch_size, 
            early_stopping=early_stopping,
            early_stopping_patience=20,
            train_size=0.9,
            plan_kwargs=plan_kwargs
        )
        print("Training complete.")
        
    def get_latent_representation(self):
        """
        Get the batch-corrected latent representation.
        """
        return self.model.get_latent_representation()
        
    def get_denoised_expression(self):
        """
        Get the denoised (reconstructed) gene expression.
        """
        return self.model.get_normalized_expression(library_size=1e4)
        
    def save(self, dir_path, overwrite=True):
        """
        Save the model.
        """
        self.model.save(dir_path, overwrite=overwrite)
        
    @classmethod
    def load(cls, dir_path, adata):
        """
        Load a saved model.
        """
        # This is a bit tricky with wrappers, but for now we can just load the inner model
        # Re-initializing wrapper would be better
        # For simplicity in this demo, we rely on the user to re-instantiate or use scvi.model.SCVI.load
        pass
