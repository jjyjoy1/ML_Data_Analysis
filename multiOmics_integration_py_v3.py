import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MultiOmicsVAE(nn.Module):
    def __init__(self, rnaseq_dim, metag_dim, hidden_dim=128, latent_dim=32):
        super(MultiOmicsVAE, self).__init__()
        
        self.rnaseq_dim = rnaseq_dim
        self.metag_dim = metag_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # RNA-seq encoder
        self.rnaseq_encoder = nn.Sequential(
            nn.Linear(rnaseq_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Metagenomics encoder
        self.metag_encoder = nn.Sequential(
            nn.Linear(metag_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Latent space parameters - using both RNA-seq and metagenomics info
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # RNA-seq decoder
        self.rnaseq_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rnaseq_dim),
        )
        
        # Metagenomics decoder
        self.metag_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, metag_dim),
        )
    
    def encode(self, rnaseq_data, metag_data):
        # Encode each data type
        rnaseq_h = self.rnaseq_encoder(rnaseq_data)
        metag_h = self.metag_encoder(metag_data)
        
        # Combine features for joint latent space
        combined_h = torch.cat([rnaseq_h, metag_h], dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(combined_h)
        logvar = self.fc_logvar(combined_h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        # Decode to get reconstructions of both data types
        rnaseq_recon = self.rnaseq_decoder(z)
        metag_recon = self.metag_decoder(z)
        
        return rnaseq_recon, metag_recon
    
    def forward(self, rnaseq_data, metag_data):
        # Encode
        mu, logvar = self.encode(rnaseq_data, metag_data)
        
        # Sample latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode
        rnaseq_recon, metag_recon = self.decode(z)
        
        return rnaseq_recon, metag_recon, mu, logvar

def load_and_preprocess_data(rnaseq_adata, metag_adata):
    """
    Assumes both AnnData objects have the same observations (samples) in the same order
    Returns preprocessed data ready for the VAE
    """
    # Extract the matrices
    rnaseq_data = rnaseq_adata.X.copy()
    metag_data = metag_adata.X.copy()
    
    # Convert to dense if sparse
    if isinstance(rnaseq_data, np.ndarray) == False:
        rnaseq_data = rnaseq_data.toarray()
    if isinstance(metag_data, np.ndarray) == False:
        metag_data = metag_data.toarray()
    
    # Split data into train/test sets
    rnaseq_train, rnaseq_test, metag_train, metag_test = train_test_split(
        rnaseq_data, metag_data, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    rnaseq_train_tensor = torch.FloatTensor(rnaseq_train)
    rnaseq_test_tensor = torch.FloatTensor(rnaseq_test)
    metag_train_tensor = torch.FloatTensor(metag_train)
    metag_test_tensor = torch.FloatTensor(metag_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(rnaseq_train_tensor, metag_train_tensor)
    test_dataset = TensorDataset(rnaseq_test_tensor, metag_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, rnaseq_data.shape[1], metag_data.shape[1]

def vae_loss(rnaseq_recon, rnaseq_data, metag_recon, metag_data, mu, logvar, beta=1.0):
    """
    Compute VAE loss with both reconstruction terms and KL divergence
    """
    # Reconstruction loss for RNA-seq data (MSE)
    rnaseq_recon_loss = F.mse_loss(rnaseq_recon, rnaseq_data, reduction='sum')
    
    # Reconstruction loss for metagenomic data (MSE)
    metag_recon_loss = F.mse_loss(metag_recon, metag_data, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta parameter for KL term
    total_loss = rnaseq_recon_loss + metag_recon_loss + beta * kl_loss
    
    return total_loss, rnaseq_recon_loss, metag_recon_loss, kl_loss

def train_vae(model, train_loader, test_loader, num_epochs=100, beta=1.0, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = 0
        
        for batch_idx, (rnaseq_batch, metag_batch) in enumerate(train_loader):
            rnaseq_batch = rnaseq_batch.to(device)
            metag_batch = metag_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            rnaseq_recon, metag_recon, mu, logvar = model(rnaseq_batch, metag_batch)
            
            # Compute loss
            loss, rnaseq_loss, metag_loss, kl = vae_loss(
                rnaseq_recon, rnaseq_batch, metag_recon, metag_batch, mu, logvar, beta
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        test_epoch_loss = 0
        
        with torch.no_grad():
            for rnaseq_batch, metag_batch in test_loader:
                rnaseq_batch = rnaseq_batch.to(device)
                metag_batch = metag_batch.to(device)
                
                rnaseq_recon, metag_recon, mu, logvar = model(rnaseq_batch, metag_batch)
                
                loss, _, _, _ = vae_loss(
                    rnaseq_recon, rnaseq_batch, metag_recon, metag_batch, mu, logvar, beta
                )
                
                test_epoch_loss += loss.item()
        
        train_avg_loss = train_epoch_loss / len(train_loader.dataset)
        test_avg_loss = test_epoch_loss / len(test_loader.dataset)
        
        train_losses.append(train_avg_loss)
        test_losses.append(test_avg_loss)
        
        # Update learning rate
        scheduler.step(test_avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}")
    
    return train_losses, test_losses

def extract_latent_features(model, data_loader, device):
    """
    Extract latent space features for all samples
    """
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for rnaseq_batch, metag_batch in data_loader:
            rnaseq_batch = rnaseq_batch.to(device)
            metag_batch = metag_batch.to(device)
            
            mu, _ = model.encode(rnaseq_batch, metag_batch)
            latent_features.append(mu.cpu().numpy())
    
    return np.vstack(latent_features)

def feature_importance(model, rnaseq_dim, metag_dim, rnaseq_data, metag_data, threshold=0.5):
    """
    Estimate feature importance by analyzing decoder weights and feature impact across samples
    
    Parameters:
    - model: The trained VAE model
    - rnaseq_dim: Dimension of RNA-seq features
    - metag_dim: Dimension of metagenomic features
    - rnaseq_data: RNA-seq data tensor
    - metag_data: Metagenomic data tensor
    - threshold: Proportion of samples a feature must be important in to be selected (0.5 = 50%)
    
    Returns:
    - rnaseq_importance: Feature importance scores for RNA-seq features
    - metag_importance: Feature importance scores for metagenomic features
    - rnaseq_selected: Boolean mask of selected RNA-seq features
    - metag_selected: Boolean mask of selected metagenomic features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get the last layer weights for RNA-seq decoder
    rnaseq_decoder_weights = model.rnaseq_decoder[-1].weight.data.cpu().numpy()
    
    # Get the last layer weights for metagenomics decoder
    metag_decoder_weights = model.metag_decoder[-1].weight.data.cpu().numpy()
    
    # Calculate global importance score (sum of absolute weights for each feature)
    rnaseq_global_importance = np.sum(np.abs(rnaseq_decoder_weights), axis=0)
    metag_global_importance = np.sum(np.abs(metag_decoder_weights), axis=0)
    
    # Sample-specific feature importance analysis
    print("Analyzing feature importance across samples...")
    
    # Convert data to tensors if not already
    if not isinstance(rnaseq_data, torch.Tensor):
        rnaseq_data = torch.FloatTensor(rnaseq_data)
    if not isinstance(metag_data, torch.Tensor):
        metag_data = torch.FloatTensor(metag_data)
    
    # Move to device
    rnaseq_data = rnaseq_data.to(device)
    metag_data = metag_data.to(device)
    
    # Get latent representations
    with torch.no_grad():
        mu, _ = model.encode(rnaseq_data, metag_data)
        
        # Compute gradients for each feature in each sample
        rnaseq_sample_importance = np.zeros((len(rnaseq_data), rnaseq_dim))
        metag_sample_importance = np.zeros((len(metag_data), metag_dim))
        
        # Process in batches to avoid memory issues
        batch_size = 32
        num_samples = len(rnaseq_data)
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # Get batch data
            rnaseq_batch = rnaseq_data[start_idx:end_idx]
            metag_batch = metag_data[start_idx:end_idx]
            
            # Get latent batch
            mu_batch, _ = model.encode(rnaseq_batch, metag_batch)
            
            # For each sample in batch
            for j in range(end_idx - start_idx):
                # Enable gradients for this specific forward pass
                z = mu_batch[j].unsqueeze(0)
                
                # Reconstruction from this single latent vector
                with torch.set_grad_enabled(True):
                    z = z.clone().detach().requires_grad_(True)
                    rnaseq_recon, metag_recon = model.decode(z)
                    
                    # Compute sensitivity for RNA-seq features
                    for k in range(rnaseq_dim):
                        if k % 100 == 0 and j == 0:  # Print progress sparingly
                            print(f"Processing RNA-seq feature {k}/{rnaseq_dim}")
                            
                        # Zero gradients
                        if z.grad is not None:
                            z.grad.zero_()
                            
                        # Compute gradient of this feature with respect to z
                        rnaseq_recon[0, k].backward(retain_graph=True)
                        
                        # Store the gradient magnitude
                        rnaseq_sample_importance[start_idx + j, k] = torch.norm(z.grad).item()
                    
                    # Reset for metagenomic features
                    z = mu_batch[j].unsqueeze(0).clone().detach().requires_grad_(True)
                    rnaseq_recon, metag_recon = model.decode(z)
                    
                    # Compute sensitivity for metagenomic features
                    for k in range(metag_dim):
                        if k % 100 == 0 and j == 0:  # Print progress sparingly
                            print(f"Processing metagenomic feature {k}/{metag_dim}")
                            
                        # Zero gradients
                        if z.grad is not None:
                            z.grad.zero_()
                            
                        # Compute gradient of this feature with respect to z
                        metag_recon[0, k].backward(retain_graph=True)
                        
                        # Store the gradient magnitude
                        metag_sample_importance[start_idx + j, k] = torch.norm(z.grad).item()
    
    # Normalize sample importance within each sample
    rnaseq_sample_importance = rnaseq_sample_importance / (np.sum(rnaseq_sample_importance, axis=1, keepdims=True) + 1e-10)
    metag_sample_importance = metag_sample_importance / (np.sum(metag_sample_importance, axis=1, keepdims=True) + 1e-10)
    
    # Count in how many samples each feature is among the top 25% most important
    rnaseq_top_quartile = np.zeros(rnaseq_dim, dtype=int)
    metag_top_quartile = np.zeros(metag_dim, dtype=int)
    
    for i in range(len(rnaseq_data)):
        # Get threshold values for top 25% in this sample
        rnaseq_threshold = np.percentile(rnaseq_sample_importance[i], 75)
        metag_threshold = np.percentile(metag_sample_importance[i], 75)
        
        # Count features above threshold
        rnaseq_top_quartile += (rnaseq_sample_importance[i] >= rnaseq_threshold).astype(int)
        metag_top_quartile += (metag_sample_importance[i] >= metag_threshold).astype(int)
    
    # Calculate proportion of samples where each feature is important
    rnaseq_proportion = rnaseq_top_quartile / len(rnaseq_data)
    metag_proportion = metag_top_quartile / len(metag_data)
    
    # Select features important in at least threshold proportion of samples
    rnaseq_selected = rnaseq_proportion >= threshold
    metag_selected = metag_proportion >= threshold
    
    # Final importance scores combine global weights and sample-level importance
    rnaseq_importance = rnaseq_global_importance * rnaseq_proportion
    metag_importance = metag_global_importance * metag_proportion
    
    print(f"Selected {np.sum(rnaseq_selected)} RNA-seq features and {np.sum(metag_selected)} metagenomic features")
    
    return rnaseq_importance, metag_importance, rnaseq_selected, metag_selected

def plot_loss(train_losses, test_losses):
    """
    Plot training and test loss curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_latent_space(latent_features, metadata=None, color_by=None):
    """
    Plot the first two dimensions of the latent space
    """
    plt.figure(figsize=(10, 8))
    
    if metadata is not None and color_by is not None and color_by in metadata.columns:
        sns.scatterplot(x=latent_features[:, 0], y=latent_features[:, 1], 
                        hue=metadata[color_by], palette='viridis')
    else:
        plt.scatter(latent_features[:, 0], latent_features[:, 1])
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space')
    plt.colorbar()
    plt.grid(True)
    plt.show()

def visualize_feature_importance(rnaseq_importance, metag_importance, rnaseq_feature_names, metag_feature_names, 
                                rnaseq_selected=None, metag_selected=None, top_n=20):
    """
    Visualize the top most important features from each data type
    
    Parameters:
    - rnaseq_importance: Feature importance scores for RNA-seq features
    - metag_importance: Feature importance scores for metagenomic features
    - rnaseq_feature_names: Names of RNA-seq features
    - metag_feature_names: Names of metagenomic features
    - rnaseq_selected: Boolean mask of selected RNA-seq features (based on threshold)
    - metag_selected: Boolean mask of selected metagenomic features (based on threshold)
    - top_n: Number of top features to display
    """
    # If selection masks are provided, display selection statistics
    if rnaseq_selected is not None and metag_selected is not None:
        print(f"Selected {np.sum(rnaseq_selected)} out of {len(rnaseq_selected)} RNA-seq features")
        print(f"Selected {np.sum(metag_selected)} out of {len(metag_selected)} metagenomic features")
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(18, 16))
        
        # Top selected RNA-seq features
        selected_rnaseq_idx = np.where(rnaseq_selected)[0]
        selected_rnaseq_importance = rnaseq_importance[selected_rnaseq_idx]
        # Sort selected features by importance
        sorted_idx = np.argsort(selected_rnaseq_importance)[::-1][:min(top_n, len(selected_rnaseq_importance))]
        top_selected_rnaseq_idx = selected_rnaseq_idx[sorted_idx]
        top_selected_rnaseq_importance = selected_rnaseq_importance[sorted_idx]
        top_selected_rnaseq_names = [rnaseq_feature_names[i] for i in top_selected_rnaseq_idx]
        
        # Top selected metagenomic features
        selected_metag_idx = np.where(metag_selected)[0]
        selected_metag_importance = metag_importance[selected_metag_idx]
        # Sort selected features by importance
        sorted_idx = np.argsort(selected_metag_importance)[::-1][:min(top_n, len(selected_metag_importance))]
        top_selected_metag_idx = selected_metag_idx[sorted_idx]
        top_selected_metag_importance = selected_metag_importance[sorted_idx]
        top_selected_metag_names = [metag_feature_names[i] for i in top_selected_metag_idx]
        
        # Plot top selected RNA-seq features
        ax1 = fig.add_subplot(2, 2, 1)
        selected_rnaseq_df = pd.DataFrame({
            'Feature': top_selected_rnaseq_names,
            'Importance': top_selected_rnaseq_importance
        })
        sns.barplot(x='Importance', y='Feature', data=selected_rnaseq_df, ax=ax1)
        ax1.set_title(f'Top {min(top_n, len(selected_rnaseq_importance))} Selected RNA-seq Features')
        ax1.set_xlabel('Importance Score')
        
        # Plot top selected metagenomic features
        ax2 = fig.add_subplot(2, 2, 2)
        selected_metag_df = pd.DataFrame({
            'Feature': top_selected_metag_names,
            'Importance': top_selected_metag_importance
        })
        sns.barplot(x='Importance', y='Feature', data=selected_metag_df, ax=ax2)
        ax2.set_title(f'Top {min(top_n, len(selected_metag_importance))} Selected Metagenomic Features')
        ax2.set_xlabel('Importance Score')
        
        # Also show overall top features regardless of selection
        # RNA-seq features
        top_rnaseq_idx = np.argsort(rnaseq_importance)[::-1][:top_n]
        top_rnaseq_importance = rnaseq_importance[top_rnaseq_idx]
        top_rnaseq_names = [rnaseq_feature_names[i] for i in top_rnaseq_idx]
        
        # Metagenomics features
        top_metag_idx = np.argsort(metag_importance)[::-1][:top_n]
        top_metag_importance = metag_importance[top_metag_idx]
        top_metag_names = [metag_feature_names[i] for i in top_metag_idx]
        
        # Plot overall top RNA-seq features
        ax3 = fig.add_subplot(2, 2, 3)
        overall_rnaseq_df = pd.DataFrame({
            'Feature': top_rnaseq_names,
            'Importance': top_rnaseq_importance,
            'Selected': [i in selected_rnaseq_idx for i in top_rnaseq_idx]
        })
        sns.barplot(x='Importance', y='Feature', hue='Selected', data=overall_rnaseq_df, ax=ax3)
        ax3.set_title(f'Top {top_n} Overall RNA-seq Features')
        ax3.set_xlabel('Importance Score')
        
        # Plot overall top metagenomic features
        ax4 = fig.add_subplot(2, 2, 4)
        overall_metag_df = pd.DataFrame({
            'Feature': top_metag_names,
            'Importance': top_metag_importance,
            'Selected': [i in selected_metag_idx for i in top_metag_idx]
        })
        sns.barplot(x='Importance', y='Feature', hue='Selected', data=overall_metag_df, ax=ax4)
        ax4.set_title(f'Top {top_n} Overall Metagenomic Features')
        ax4.set_xlabel('Importance Score')
        
    else:
        # Original visualization for when no selection is provided
        # RNA-seq features
        top_rnaseq_idx = np.argsort(rnaseq_importance)[::-1][:top_n]
        top_rnaseq_importance = rnaseq_importance[top_rnaseq_idx]
        top_rnaseq_names = [rnaseq_feature_names[i] for i in top_rnaseq_idx]
        
        # Metagenomics features
        top_metag_idx = np.argsort(metag_importance)[::-1][:top_n]
        top_metag_importance = metag_importance[top_metag_idx]
        top_metag_names = [metag_feature_names[i] for i in top_metag_idx]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # RNA-seq
        sns.barplot(x=top_rnaseq_importance, y=top_rnaseq_names, ax=ax1)
        ax1.set_title(f'Top {top_n} RNA-seq Features')
        ax1.set_xlabel('Importance Score')
        
        # Metagenomics
        sns.barplot(x=top_metag_importance, y=top_metag_names, ax=ax2)
        ax2.set_title(f'Top {top_n} Metagenomic Features')
        ax2.set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.show()

# Main execution
def run_multi_omics_integration(rnaseq_adata, metag_adata, hidden_dim=128, latent_dim=32, 
                                num_epochs=100, beta=1.0, lr=1e-3, feature_threshold=0.5):
    """
    Main function to run the multi-omics integration pipeline
    
    Parameters:
    - rnaseq_adata: AnnData object with RNA-seq data
    - metag_adata: AnnData object with metagenomic data
    - hidden_dim: Dimension of hidden layers
    - latent_dim: Dimension of latent space
    - num_epochs: Number of training epochs
    - beta: Weight for KL divergence term
    - lr: Learning rate
    - feature_threshold: Proportion of samples a feature must be important in to be selected (0.5 = 50%)
    
    Returns:
    - model: Trained VAE model
    - latent_features: Extracted latent features for all samples
    - rnaseq_importance: Feature importance scores for RNA-seq features
    - metag_importance: Feature importance scores for metagenomic features
    - rnaseq_selected: Boolean mask of selected RNA-seq features
    - metag_selected: Boolean mask of selected metagenomic features
    - selected_feature_names: Dictionary with lists of selected feature names
    """
    # Check if samples match
    assert rnaseq_adata.obs_names.equals(metag_adata.obs_names), "Sample IDs don't match between datasets"
    
    # Load and preprocess data
    train_loader, test_loader, rnaseq_dim, metag_dim = load_and_preprocess_data(rnaseq_adata, metag_adata)
    
    # Initialize model
    model = MultiOmicsVAE(rnaseq_dim, metag_dim, hidden_dim, latent_dim)
    
    # Train model
    print("Training VAE model...")
    train_losses, test_losses = train_vae(model, train_loader, test_loader, num_epochs, beta, lr)
    
    # Plot loss curves
    plot_loss(train_losses, test_losses)
    
    # Extract latent features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare full data for feature importance analysis
    full_rnaseq_data = rnaseq_adata.X.toarray() if hasattr(rnaseq_adata.X, 'toarray') else rnaseq_adata.X
    full_metag_data = metag_adata.X.toarray() if hasattr(metag_adata.X, 'toarray') else metag_adata.X
    
    full_dataset = TensorDataset(torch.FloatTensor(full_rnaseq_data), torch.FloatTensor(full_metag_data))
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    print("Extracting latent features...")
    latent_features = extract_latent_features(model, full_loader, device)
    
    # Get feature importance with sample-level threshold
    print(f"Calculating feature importance with {feature_threshold*100}% sample threshold...")
    rnaseq_importance, metag_importance, rnaseq_selected, metag_selected = feature_importance(
        model, rnaseq_dim, metag_dim, full_rnaseq_data, full_metag_data, threshold=feature_threshold
    )
    
    # Plot latent space
    if hasattr(rnaseq_adata, 'obs') and isinstance(rnaseq_adata.obs, pd.DataFrame):
        plot_latent_space(latent_features, rnaseq_adata.obs)
    else:
        plot_latent_space(latent_features)
    
    # Plot feature importance
    if hasattr(rnaseq_adata, 'var_names') and hasattr(metag_adata, 'var_names'):
        visualize_feature_importance(
            rnaseq_importance, metag_importance, 
            rnaseq_adata.var_names, metag_adata.var_names,
            rnaseq_selected, metag_selected
        )
    
    # Add latent features to AnnData objects
    latent_df = pd.DataFrame(latent_features, index=rnaseq_adata.obs_names,
                             columns=[f'latent_{i}' for i in range(latent_dim)])
    
    rnaseq_adata.obsm['X_vae'] = latent_features
    metag_adata.obsm['X_vae'] = latent_features
    
    # Create lists of selected feature names
    selected_feature_names = {
        'rnaseq': [rnaseq_adata.var_names[i] for i in np.where(rnaseq_selected)[0]],
        'metag': [metag_adata.var_names[i] for i in np.where(metag_selected)[0]]
    }
    
    # Add feature selection results to AnnData objects
    rnaseq_adata.var['vae_importance'] = rnaseq_importance
    rnaseq_adata.var['vae_selected'] = rnaseq_selected
    
    metag_adata.var['vae_importance'] = metag_importance
    metag_adata.var['vae_selected'] = metag_selected
    
    return model, latent_features, rnaseq_importance, metag_importance, rnaseq_selected, metag_selected, selected_feature_names

# Example usage:
if __name__ == "__main__":
    # Simulated data creation for demonstration
    
    # Create simulated RNA-seq data
    n_samples = 100
    n_genes = 1000
    n_taxa = 500
    
    # Simulated RNA-seq data
    rna_data = np.random.negative_binomial(10, 0.5, size=(n_samples, n_genes))
    rna_adata = sc.AnnData(rna_data)
    rna_adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    rna_adata.obs_names = [f'sample_{i}' for i in range(n_samples)]
    
    # Simulated metagenomic data
    meta_data = np.random.negative_binomial(5, 0.7, size=(n_samples, n_taxa))
    meta_adata = sc.AnnData(meta_data)
    meta_adata.var_names = [f'taxa_{i}' for i in range(n_taxa)]
    meta_adata.obs_names = [f'sample_{i}' for i in range(n_samples)]
    
    # Add some group labels for visualization
    groups = np.random.choice(['A', 'B', 'C'], size=n_samples)
    rna_adata.obs['group'] = groups
    meta_adata.obs['group'] = groups
    
    # Normalize data
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    
    sc.pp.normalize_total(meta_adata, target_sum=1e4)
    sc.pp.log1p(meta_adata)
    
    # Run multi-omics integration
    model, latent_features, rna_importance, meta_importance, latent_df = run_multi_omics_integration(
        rna_adata, meta_adata, hidden_dim=128, latent_dim=20, num_epochs=50, beta=0.5
    )
    
    print("Integration complete!")
    print(f"Extracted {latent_features.shape[1]} latent dimensions from {n_samples} samples")
