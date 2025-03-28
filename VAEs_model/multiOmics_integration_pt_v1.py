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

def feature_importance(model, rnaseq_dim, metag_dim):
    """
    Estimate feature importance by analyzing decoder weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get the last layer weights for RNA-seq decoder
    rnaseq_decoder_weights = model.rnaseq_decoder[-1].weight.data.cpu().numpy()
    
    # Get the last layer weights for metagenomics decoder
    metag_decoder_weights = model.metag_decoder[-1].weight.data.cpu().numpy()
    
    # Calculate importance score (sum of absolute weights for each feature)
    rnaseq_importance = np.sum(np.abs(rnaseq_decoder_weights), axis=0)
    metag_importance = np.sum(np.abs(metag_decoder_weights), axis=0)
    
    return rnaseq_importance, metag_importance

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

def visualize_feature_importance(rnaseq_importance, metag_importance, rnaseq_feature_names, metag_feature_names, top_n=20):
    """
    Visualize the top most important features from each data type
    """
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
                                num_epochs=100, beta=1.0, lr=1e-3):
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
    
    Returns:
    - model: Trained VAE model
    - latent_features: Extracted latent features for all samples
    - rnaseq_importance: Feature importance scores for RNA-seq features
    - metag_importance: Feature importance scores for metagenomic features
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
    full_dataset = TensorDataset(torch.FloatTensor(rnaseq_adata.X.toarray() if hasattr(rnaseq_adata.X, 'toarray') else rnaseq_adata.X),
                                torch.FloatTensor(metag_adata.X.toarray() if hasattr(metag_adata.X, 'toarray') else metag_adata.X))
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    print("Extracting latent features...")
    latent_features = extract_latent_features(model, full_loader, device)
    
    # Get feature importance
    print("Calculating feature importance...")
    rnaseq_importance, metag_importance = feature_importance(model, rnaseq_dim, metag_dim)
    
    # Plot latent space
    if hasattr(rnaseq_adata, 'obs') and isinstance(rnaseq_adata.obs, pd.DataFrame):
        plot_latent_space(latent_features, rnaseq_adata.obs)
    else:
        plot_latent_space(latent_features)
    
    # Plot feature importance
    if hasattr(rnaseq_adata, 'var_names') and hasattr(metag_adata, 'var_names'):
        visualize_feature_importance(rnaseq_importance, metag_importance, 
                                     rnaseq_adata.var_names, metag_adata.var_names)
    
    # Add latent features to AnnData objects
    latent_df = pd.DataFrame(latent_features, index=rnaseq_adata.obs_names,
                             columns=[f'latent_{i}' for i in range(latent_dim)])
    
    rnaseq_adata.obsm['X_vae'] = latent_features
    metag_adata.obsm['X_vae'] = latent_features
    
    return model, latent_features, rnaseq_importance, metag_importance, latent_df

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


