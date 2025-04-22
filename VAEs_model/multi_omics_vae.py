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
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MultiOmicsVAE(nn.Module):
    def __init__(self, rnaseq_dim, metag_dim, clinical_dim=0, hidden_dim=128, latent_dim=32):
        super(MultiOmicsVAE, self).__init__()
        
        self.rnaseq_dim = rnaseq_dim
        self.metag_dim = metag_dim
        self.clinical_dim = clinical_dim
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
        
        # Clinical data encoder (if provided)
        if clinical_dim > 0:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_dim, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.BatchNorm1d(hidden_dim // 8),
                nn.ReLU(),
            )
            # Combined dimension from all encoders
            combined_dim = hidden_dim + hidden_dim // 8
        else:
            self.clinical_encoder = None
            combined_dim = hidden_dim  # Just RNA-seq and metagenomics
        
        # Latent space parameters - using both RNA-seq and metagenomics info
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)
        
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
        
        # Clinical data decoder (if provided) - for reconstruction
        if clinical_dim > 0:
            self.clinical_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 4, clinical_dim),
            )
        else:
            self.clinical_decoder = None
            
        # Cancer stage classifier from latent space
        self.stage_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 5)  # 5 cancer stages (I, II, III, IV, V)
        )
        
        # Drug response predictor from latent space
        self.drug_response_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Binary prediction
        )
    
    def encode(self, rnaseq_data, metag_data, clinical_data=None):
        # Encode each data type
        rnaseq_h = self.rnaseq_encoder(rnaseq_data)
        metag_h = self.metag_encoder(metag_data)
        
        # Combine features for joint latent space
        if clinical_data is not None and self.clinical_encoder is not None:
            clinical_h = self.clinical_encoder(clinical_data)
            combined_h = torch.cat([rnaseq_h, metag_h, clinical_h], dim=1)
        else:
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
        # Decode to get reconstructions of all data types
        rnaseq_recon = self.rnaseq_decoder(z)
        metag_recon = self.metag_decoder(z)
        
        if self.clinical_decoder is not None:
            clinical_recon = self.clinical_decoder(z)
            return rnaseq_recon, metag_recon, clinical_recon
        else:
            return rnaseq_recon, metag_recon
    
    def predict_stage(self, z):
        # Predict cancer stage from latent representation
        return self.stage_classifier(z)
    
    def predict_drug_response(self, z):
        # Predict drug response from latent representation
        return self.drug_response_predictor(z)
    
    def forward(self, rnaseq_data, metag_data, clinical_data=None):
        # Encode
        mu, logvar = self.encode(rnaseq_data, metag_data, clinical_data)
        
        # Sample latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode
        if clinical_data is not None and self.clinical_decoder is not None:
            rnaseq_recon, metag_recon, clinical_recon = self.decode(z)
        else:
            rnaseq_recon, metag_recon = self.decode(z)
            clinical_recon = None
        
        # Predictions from latent space
        stage_logits = self.predict_stage(z)
        drug_response = self.predict_drug_response(z)
        
        if clinical_recon is not None:
            return rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_response
        else:
            return rnaseq_recon, metag_recon, mu, logvar, stage_logits, drug_response


def load_and_preprocess_data(rnaseq_adata, metag_adata, clinical_vars=None):
    """
    Assumes both AnnData objects have the same observations (samples) in the same order
    Returns preprocessed data ready for the VAE
    
    Parameters:
    - rnaseq_adata: AnnData object with RNA-seq data
    - metag_adata: AnnData object with metagenomic data
    - clinical_vars: List of clinical variables to include (must be columns in rnaseq_adata.obs)
    
    Returns:
    - train_loader, test_loader: PyTorch data loaders for training and testing
    - rnaseq_dim, metag_dim, clinical_dim: Dimensions of each data type
    - clinical_info: Dictionary with metadata about clinical variables
    """
    # Extract the matrices
    rnaseq_data = rnaseq_adata.X.copy()
    metag_data = metag_adata.X.copy()
    
    # Convert to dense if sparse
    if isinstance(rnaseq_data, np.ndarray) == False:
        rnaseq_data = rnaseq_data.toarray()
    if isinstance(metag_data, np.ndarray) == False:
        metag_data = metag_data.toarray()
    
    # Process clinical data if provided
    clinical_data = None
    clinical_info = {'included_vars': [], 'encoders': {}, 'continuous': [], 'categorical': []}
    
    if clinical_vars is not None and len(clinical_vars) > 0:
        # Check if all variables exist in the AnnData object
        for var in clinical_vars:
            if var not in rnaseq_adata.obs.columns:
                raise ValueError(f"Clinical variable '{var}' not found in RNA-seq AnnData object")
        
        # Extract clinical data
        clinical_df = rnaseq_adata.obs[clinical_vars].copy()
        
        # Process each clinical variable
        processed_columns = []
        
        for col in clinical_df.columns:
            # Check data type
            if pd.api.types.is_numeric_dtype(clinical_df[col]):
                # For continuous variables, just standardize
                scaler = StandardScaler()
                clinical_df[col] = scaler.fit_transform(clinical_df[[col]])
                clinical_info['continuous'].append(col)
                clinical_info['included_vars'].append(col)
                processed_columns.append(col)
                clinical_info['encoders'][col] = scaler
            else:
                # For categorical variables, use one-hot encoding
                # Special handling for cancer stage (I, II, III, IV, V)
                if col.lower() in ['stage', 'cancer_stage', 'tumor_stage']:
                    # Map roman numerals to integers
                    stage_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4}
                    # Convert to numeric
                    try:
                        clinical_df[col] = clinical_df[col].map(stage_mapping)
                        clinical_info['categorical'].append(col)
                        clinical_info['included_vars'].append(col)
                        processed_columns.append(col)
                    except:
                        # If mapping fails, try one-hot encoding
                        print(f"Could not map stage values. Using one-hot encoding for '{col}'")
                        dummies = pd.get_dummies(clinical_df[col], prefix=col)
                        clinical_df = pd.concat([clinical_df, dummies], axis=1)
                        clinical_info['categorical'].extend(dummies.columns.tolist())
                        clinical_info['included_vars'].extend(dummies.columns.tolist())
                        processed_columns.extend(dummies.columns.tolist())
                        # Drop original column
                        clinical_df = clinical_df.drop(col, axis=1)
                # Special handling for drug response (positive/negative)
                elif col.lower() in ['drug_response', 'response', 'treatment_response']:
                    # Map to binary
                    response_mapping = {'positive': 1, 'negative': 0, 'yes': 1, 'no': 0, 'responsive': 1, 'non-responsive': 0}
                    # Try to convert to binary
                    try:
                        clinical_df[col] = clinical_df[col].str.lower().map(response_mapping)
                        clinical_info['categorical'].append(col)
                        clinical_info['included_vars'].append(col)
                        processed_columns.append(col)
                    except:
                        print(f"Could not map response values. Using one-hot encoding for '{col}'")
                        dummies = pd.get_dummies(clinical_df[col], prefix=col)
                        clinical_df = pd.concat([clinical_df, dummies], axis=1)
                        clinical_info['categorical'].extend(dummies.columns.tolist())
                        clinical_info['included_vars'].extend(dummies.columns.tolist())
                        processed_columns.extend(dummies.columns.tolist())
                        # Drop original column
                        clinical_df = clinical_df.drop(col, axis=1)
                else:
                    # General case: one-hot encode
                    dummies = pd.get_dummies(clinical_df[col], prefix=col)
                    clinical_df = pd.concat([clinical_df, dummies], axis=1)
                    clinical_info['categorical'].extend(dummies.columns.tolist())
                    clinical_info['included_vars'].extend(dummies.columns.tolist())
                    processed_columns.extend(dummies.columns.tolist())
                    # Drop original column
                    clinical_df = clinical_df.drop(col, axis=1)
        
        # Keep only processed columns
        clinical_df = clinical_df[processed_columns]
        
        # Convert to numpy array
        clinical_data = clinical_df.values
    
    # Split data into train/test sets
    if clinical_data is not None:
        rnaseq_train, rnaseq_test, metag_train, metag_test, clinical_train, clinical_test = train_test_split(
            rnaseq_data, metag_data, clinical_data, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        rnaseq_train_tensor = torch.FloatTensor(rnaseq_train)
        rnaseq_test_tensor = torch.FloatTensor(rnaseq_test)
        metag_train_tensor = torch.FloatTensor(metag_train)
        metag_test_tensor = torch.FloatTensor(metag_test)
        clinical_train_tensor = torch.FloatTensor(clinical_train)
        clinical_test_tensor = torch.FloatTensor(clinical_test)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(rnaseq_train_tensor, metag_train_tensor, clinical_train_tensor)
        test_dataset = TensorDataset(rnaseq_test_tensor, metag_test_tensor, clinical_test_tensor)
        
        clinical_dim = clinical_data.shape[1]
    else:
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
        
        clinical_dim = 0
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, rnaseq_data.shape[1], metag_data.shape[1], clinical_dim, clinical_info



def vae_loss(rnaseq_recon, rnaseq_data, metag_recon, metag_data, mu, logvar, 
             clinical_recon=None, clinical_data=None, 
             stage_logits=None, true_stage=None,
             drug_response_pred=None, true_drug_response=None,
             beta=1.0, lambda_stage=0.5, lambda_drug=0.5):
    """
    Compute VAE loss with both reconstruction terms, KL divergence, and optional supervised terms
    
    Parameters:
    - rnaseq_recon, rnaseq_data: RNA-seq reconstruction and data
    - metag_recon, metag_data: Metagenomic reconstruction and data
    - mu, logvar: Latent space parameters
    - clinical_recon, clinical_data: Optional clinical data reconstruction and data
    - stage_logits, true_stage: Optional cancer stage prediction and true labels
    - drug_response_pred, true_drug_response: Optional drug response prediction and true labels
    - beta: Weight for KL divergence term
    - lambda_stage: Weight for cancer stage prediction loss
    - lambda_drug: Weight for drug response prediction loss
    
    Returns:
    - total_loss: Overall loss
    - Component losses for monitoring
    """
    # Reconstruction loss for RNA-seq data (MSE)
    rnaseq_recon_loss = F.mse_loss(rnaseq_recon, rnaseq_data, reduction='sum')
    
    # Reconstruction loss for metagenomic data (MSE)
    metag_recon_loss = F.mse_loss(metag_recon, metag_data, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Initialize total loss with basic components
    total_loss = rnaseq_recon_loss + metag_recon_loss + beta * kl_loss
    
    # Track all loss components
    loss_components = {
        'rnaseq_loss': rnaseq_recon_loss.item(),
        'metag_loss': metag_recon_loss.item(),
        'kl_loss': kl_loss.item()
    }
    
    # Add clinical reconstruction loss if provided
    if clinical_recon is not None and clinical_data is not None:
        clinical_recon_loss = F.mse_loss(clinical_recon, clinical_data, reduction='sum')
        total_loss += clinical_recon_loss
        loss_components['clinical_loss'] = clinical_recon_loss.item()
    
    # Add cancer stage prediction loss if provided
    if stage_logits is not None and true_stage is not None:
        stage_loss = F.cross_entropy(stage_logits, true_stage, reduction='sum')
        total_loss += lambda_stage * stage_loss
        loss_components['stage_loss'] = stage_loss.item()
    
    # Add drug response prediction loss if provided
    if drug_response_pred is not None and true_drug_response is not None:
        drug_loss = F.binary_cross_entropy(drug_response_pred, true_drug_response, reduction='sum')
        total_loss += lambda_drug * drug_loss
        loss_components['drug_loss'] = drug_loss.item()
    
    return total_loss, loss_components


def train_vae(model, train_loader, test_loader, has_clinical=False, has_stage=False, has_drug=False, 
             num_epochs=100, beta=1.0, lambda_stage=0.5, lambda_drug=0.5, lr=1e-3):
    """
    Train the multi-omics VAE model
    
    Parameters:
    - model: The VAE model
    - train_loader, test_loader: Data loaders for training and testing
    - has_clinical, has_stage, has_drug: Flags indicating if clinical data, stage, and drug response data are available
    - num_epochs: Number of training epochs
    - beta, lambda_stage, lambda_drug: Loss weights
    - lr: Learning rate
    
    Returns:
    - train_losses, test_losses: Loss history
    - train_metrics, test_metrics: Additional metrics (accuracy, etc.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    test_losses = []
    
    # Track metrics
    train_metrics = {
        'stage_acc': [], 
        'drug_auc': []
    }
    test_metrics = {
        'stage_acc': [], 
        'drug_auc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = 0
        train_loss_components = defaultdict(float)
        
        # For tracking metrics
        all_stage_preds = []
        all_stage_true = []
        all_drug_preds = []
        all_drug_true = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle different data configurations
            if has_clinical:
                rnaseq_batch, metag_batch, clinical_batch = batch_data
                clinical_batch = clinical_batch.to(device)
                
                # Extract stage and drug response if available
                if has_stage and has_drug:
                    # Assuming stage is first column and drug response is second column
                    true_stage = clinical_batch[:, 0].long()
                    true_drug = clinical_batch[:, 1].float().unsqueeze(1)
                    # Remove these from clinical data for reconstruction
                    clinical_features = clinical_batch[:, 2:]
                elif has_stage:
                    true_stage = clinical_batch[:, 0].long()
                    true_drug = None
                    clinical_features = clinical_batch[:, 1:]
                elif has_drug:
                    true_drug = clinical_batch[:, 0].float().unsqueeze(1)
                    true_stage = None
                    clinical_features = clinical_batch[:, 1:]
                else:
                    true_stage = None
                    true_drug = None
                    clinical_features = clinical_batch
            else:
                rnaseq_batch, metag_batch = batch_data
                clinical_features = None
                true_stage = None
                true_drug = None
            
            rnaseq_batch = rnaseq_batch.to(device)
            metag_batch = metag_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - handle different data configurations
            if has_clinical:
                outputs = model(rnaseq_batch, metag_batch, clinical_features)
                
                if has_stage and has_drug:
                    rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_pred = outputs
                    
                    # Store predictions for metrics
                    all_stage_preds.append(stage_logits.detach().cpu())
                    all_stage_true.append(true_stage.detach().cpu())
                    all_drug_preds.append(drug_pred.detach().cpu())
                    all_drug_true.append(true_drug.detach().cpu())
                    
                    # Compute loss
                    loss, loss_comps = vae_loss(
                        rnaseq_recon, rnaseq_batch, 
                        metag_recon, metag_batch, 
                        mu, logvar,
                        clinical_recon, clinical_features,
                        stage_logits, true_stage,
                        drug_pred, true_drug,
                        beta, lambda_stage, lambda_drug
                    )
                elif has_stage:
                    rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_pred = outputs
                    
                    # Store predictions for metrics
                    all_stage_preds.append(stage_logits.detach().cpu())
                    all_stage_true.append(true_stage.detach().cpu())
                    
                    # Compute loss
                    loss, loss_comps = vae_loss(
                        rnaseq_recon, rnaseq_batch, 
                        metag_recon, metag_batch, 
                        mu, logvar,
                        clinical_recon, clinical_features,
                        stage_logits, true_stage,
                        None, None,
                        beta, lambda_stage, lambda_drug
                    )
                elif has_drug:
                    rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_pred = outputs
                    
                    # Store predictions for metrics
                    all_drug_preds.append(drug_pred.detach().cpu())
                    all_drug_true.append(true_drug.detach().cpu())
                    
                    # Compute loss
                    loss, loss_comps = vae_loss(
                        rnaseq_recon, rnaseq_batch, 
                        metag_recon, metag_batch, 
                        mu, logvar,
                        clinical_recon, clinical_features,
                        None, None,
                        drug_pred, true_drug,
                        beta, lambda_stage, lambda_drug
                    )
                else:
                    rnaseq_recon, metag_recon, clinical_recon, mu, logvar, _, _ = outputs
                    
                    # Compute loss
                    loss, loss_comps = vae_loss(
                        rnaseq_recon, rnaseq_batch, 
                        metag_recon, metag_batch, 
                        mu, logvar,
                        clinical_recon, clinical_features,
                        None, None,
                        None, None,
                        beta
                    )
            else:
                # Basic case without clinical data
                rnaseq_recon, metag_recon, mu, logvar, _, _ = model(rnaseq_batch, metag_batch)
                
                # Compute loss
                loss, loss_comps = vae_loss(
                    rnaseq_recon, rnaseq_batch, 
                    metag_recon, metag_batch, 
                    mu, logvar,
                    beta=beta
                )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
            # Track loss components
            for k, v in loss_comps.items():
                train_loss_components[k] += v
        
        # Calculate metrics for the epoch
        if has_stage and len(all_stage_true) > 0:
            all_stage_true = torch.cat(all_stage_true)
            all_stage_preds = torch.cat(all_stage_preds)
            stage_acc = (all_stage_preds.argmax(dim=1) == all_stage_true).float().mean().item()
            train_metrics['stage_acc'].append(stage_acc)
        
        if has_drug and len(all_drug_true) > 0:
            all_drug_true = torch.cat(all_drug_true)
            all_drug_preds = torch.cat(all_drug_preds)
            # Compute AUC
            try:
                drug_auc = roc_auc_score(all_drug_true.numpy(), all_drug_preds.numpy())
                train_metrics['drug_auc'].append(drug_auc)
            except:
                train_metrics['drug_auc'].append(0.5)  # Default if calculation fails
        
        # Evaluate on test set
        model.eval()
        test_epoch_loss = 0
        test_loss_components = defaultdict(float)
        
        # For tracking metrics
        all_stage_preds = []
        all_stage_true = []
        all_drug_preds = []
        all_drug_true = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Handle different data configurations
                if has_clinical:
                    rnaseq_batch, metag_batch, clinical_batch = batch_data
                    clinical_batch = clinical_batch.to(device)
                    
                    # Extract stage and drug response if available
                    if has_stage and has_drug:
                        true_stage = clinical_batch[:, 0].long()
                        true_drug = clinical_batch[:, 1].float().unsqueeze(1)
                        clinical_features = clinical_batch[:, 2:]
                    elif has_stage:
                        true_stage = clinical_batch[:, 0].long()
                        true_drug = None
                        clinical_features = clinical_batch[:, 1:]
                    elif has_drug:
                        true_drug = clinical_batch[:, 0].float().unsqueeze(1)
                        true_stage = None
                        clinical_features = clinical_batch[:, 1:]
                    else:
                        true_stage = None
                        true_drug = None
                        clinical_features = clinical_batch
                else:
                    rnaseq_batch, metag_batch = batch_data
                    clinical_features = None
                    true_stage = None
                    true_drug = None
                
                rnaseq_batch = rnaseq_batch.to(device)
                metag_batch = metag_batch.to(device)
                
                # Forward pass - handle different data configurations
                if has_clinical:
                    outputs = model(rnaseq_batch, metag_batch, clinical_features)
                    
                    if has_stage and has_drug:
                        rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_pred = outputs
                        
                        # Store predictions for metrics
                        all_stage_preds.append(stage_logits.detach().cpu())
                        all_stage_true.append(true_stage.detach().cpu())
                        all_drug_preds.append(drug_pred.detach().cpu())
                        all_drug_true.append(true_drug.detach().cpu())
                        
                        # Compute loss
                        loss, loss_comps = vae_loss(
                            rnaseq_recon, rnaseq_batch, 
                            metag_recon, metag_batch, 
                            mu, logvar,
                            clinical_recon, clinical_features,
                            stage_logits, true_stage,
                            drug_pred, true_drug,
                            beta, lambda_stage, lambda_drug
                        )
                    elif has_stage:
                        rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_pred = outputs
                        
                        # Store predictions for metrics
                        all_stage_preds.append(stage_logits.detach().cpu())
                        all_stage_true.append(true_stage.detach().cpu())
                        
                        # Compute loss
                        loss, loss_comps = vae_loss(
                            rnaseq_recon, rnaseq_batch, 
                            metag_recon, metag_batch, 
                            mu, logvar,
                            clinical_recon, clinical_features,
                            stage_logits, true_stage,
                            None, None,
                            beta, lambda_stage, lambda_drug
                        )
                    elif has_drug:
                        rnaseq_recon, metag_recon, clinical_recon, mu, logvar, stage_logits, drug_pred = outputs
                        
                        # Store predictions for metrics
                        all_drug_preds.append(drug_pred.detach().cpu())
                        all_drug_true.append(true_drug.detach().cpu())
                        
                        # Compute loss
                        loss, loss_comps = vae_loss(
                            rnaseq_recon, rnaseq_batch, 
                            metag_recon, metag_batch, 
                            mu, logvar,
                            clinical_recon, clinical_features,
                            None, None,
                            drug_pred, true_drug,
                            beta, lambda_stage, lambda_drug
                        )
                    else:
                        rnaseq_recon, metag_recon, clinical_recon, mu, logvar, _, _ = outputs
                        
                        # Compute loss
                        loss, loss_comps = vae_loss(
                            rnaseq_recon, rnaseq_batch, 
                            metag_recon, metag_batch, 
                            mu, logvar,
                            clinical_recon, clinical_features,
                            None, None,
                            None, None,
                            beta
                        )
                else:
                    # Basic case without clinical data
                    rnaseq_recon, metag_recon, mu, logvar, _, _ = model(rnaseq_batch, metag_batch)
                    
                    # Compute loss
                    loss, loss_comps = vae_loss(
                        rnaseq_recon, rnaseq_batch, 
                        metag_recon, metag_batch, 
                        mu, logvar,
                        beta=beta
                    )
                
                test_epoch_loss += loss.item()
                
                # Track loss components
                for k, v in loss_comps.items():
                    test_loss_components[k] += v
        
        # Calculate metrics for the test set
        if has_stage and len(all_stage_true) > 0:
            all_stage_true = torch.cat(all_stage_true)
            all_stage_preds = torch.cat(all_stage_preds)
            stage_acc = (all_stage_preds.argmax(dim=1) == all_stage_true).float().mean().item()
            test_metrics['stage_acc'].append(stage_acc)
        
        if has_drug and len(all_drug_true) > 0:
            all_drug_true = torch.cat(all_drug_true)
            all_drug_preds = torch.cat(all_drug_preds)
            # Compute AUC
            try:
                drug_auc = roc_auc_score(all_drug_true.numpy(), all_drug_preds.numpy())
                test_metrics['drug_auc'].append(drug_auc)
            except:
                test_metrics['drug_auc'].append(0.5)  # Default if calculation fails
        
        train_avg_loss = train_epoch_loss / len(train_loader.dataset)
        test_avg_loss = test_epoch_loss / len(test_loader.dataset)
        
        train_losses.append(train_avg_loss)
        test_losses.append(test_avg_loss)
        
        # Update learning rate
        scheduler.step(test_avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}")
            
            # Print additional metrics if available
            metrics_str = ""
            if has_stage and 'stage_acc' in test_metrics and len(test_metrics['stage_acc']) > 0:
                metrics_str += f", Stage Acc: {test_metrics['stage_acc'][-1]:.4f}"
            if has_drug and 'drug_auc' in test_metrics and len(test_metrics['drug_auc']) > 0:
                metrics_str += f", Drug AUC: {test_metrics['drug_auc'][-1]:.4f}"
            
            if metrics_str:
                print(f"Test Metrics:{metrics_str}")
    
    return train_losses, test_losses, train_metrics, test_metrics



def extract_latent_features(model, data_loader, device, has_clinical=False):
    """
    Extract latent space features for all samples
    """
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if has_clinical:
                rnaseq_batch, metag_batch, clinical_batch = batch_data
                rnaseq_batch = rnaseq_batch.to(device)
                metag_batch = metag_batch.to(device)
                clinical_batch = clinical_batch.to(device)
                
                mu, _ = model.encode(rnaseq_batch, metag_batch, clinical_batch)
            else:
                rnaseq_batch, metag_batch = batch_data
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


def plot_latent_space(latent_features, metadata=None, color_by=None, shape_by=None, annotate=False):
    """
    Plot the first two dimensions of the latent space
    
    Parameters:
    - latent_features: Extracted latent features
    - metadata: DataFrame with sample metadata
    - color_by: Column in metadata to use for coloring points
    - shape_by: Column in metadata to use for point shapes
    - annotate: Whether to annotate points with sample IDs
    """
    plt.figure(figsize=(12, 10))
    
    # If we have both coloring and shapes
    if metadata is not None and color_by is not None and color_by in metadata.columns:
        if shape_by is not None and shape_by in metadata.columns:
            # Create a scatter plot for each combination of color and shape
            color_categories = metadata[color_by].unique()
            shape_categories = metadata[shape_by].unique()
            
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
            if len(shape_categories) > len(markers):
                print(f"Warning: Too many shape categories ({len(shape_categories)}). Using only first {len(markers)}.")
                shape_categories = shape_categories[:len(markers)]
            
            for i, shape_cat in enumerate(shape_categories):
                shape_mask = metadata[shape_by] == shape_cat
                
                # Create a subset of data for this shape
                shape_latent = latent_features[shape_mask]
                shape_metadata = metadata[shape_mask]
                
                # Plot this shape with colors
                scatter = plt.scatter(
                    shape_latent[:, 0], shape_latent[:, 1],
                    c=shape_metadata[color_by].astype('category').cat.codes,
                    marker=markers[i],
                    cmap='viridis',
                    alpha=0.8,
                    s=80,
                    label=f"{shape_by}={shape_cat}"
                )
            
            plt.legend(title=shape_by, loc='best')
            
            # Add a color bar for the color variable
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_by)
            
            # If color is categorical, adjust colorbar ticks
            if not pd.api.types.is_numeric_dtype(metadata[color_by]):
                unique_colors = metadata[color_by].unique()
                cbar.set_ticks(np.arange(len(unique_colors)))
                cbar.set_ticklabels(unique_colors)
        else:
            # Color only, no shapes
            if pd.api.types.is_numeric_dtype(metadata[color_by]):
                # Continuous color scale
                scatter = plt.scatter(
                    latent_features[:, 0], latent_features[:, 1],
                    c=metadata[color_by],
                    cmap='viridis',
                    alpha=0.8,
                    s=80
                )
                cbar = plt.colorbar(scatter)
                cbar.set_label(color_by)
            else:
                # Categorical color
                sns.scatterplot(
                    x=latent_features[:, 0], y=latent_features[:, 1],
                    hue=metadata[color_by],
                    palette='viridis',
                    s=80,
                    alpha=0.8
                )
                plt.legend(title=color_by)
    else:
        # Basic plot without metadata coloring
        plt.scatter(latent_features[:, 0], latent_features[:, 1], alpha=0.8, s=80)
    
    # Annotate points if requested
    if annotate and metadata is not None and metadata.index is not None:
        for i, sample_id in enumerate(metadata.index):
            plt.annotate(
                sample_id,
                (latent_features[i, 0], latent_features[i, 1]),
                fontsize=8,
                alpha=0.7
            )
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # If we have more than 2 dimensions, also plot dimensions 3 vs 4
    if latent_features.shape[1] > 3:
        plt.figure(figsize=(12, 10))
        
        if metadata is not None and color_by is not None and color_by in metadata.columns:
            if pd.api.types.is_numeric_dtype(metadata[color_by]):
                scatter = plt.scatter(
                    latent_features[:, 2], latent_features[:, 3],
                    c=metadata[color_by],
                    cmap='viridis',
                    alpha=0.8,
                    s=80
                )
                cbar = plt.colorbar(scatter)
                cbar.set_label(color_by)
            else:
                sns.scatterplot(
                    x=latent_features[:, 2], y=latent_features[:, 3],
                    hue=metadata[color_by],
                    palette='viridis',
                    s=80,
                    alpha=0.8
                )
                plt.legend(title=color_by)
        else:
            plt.scatter(latent_features[:, 2], latent_features[:, 3], alpha=0.8, s=80)
        
        plt.xlabel('Latent Dimension 3')
        plt.ylabel('Latent Dimension 4')
        plt.title('VAE Latent Space (Dimensions 3 vs 4)')
        plt.grid(True, alpha=0.3)
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



def run_multi_omics_integration(rnaseq_adata, metag_adata, clinical_vars=None,
                                hidden_dim=128, latent_dim=32, 
                                num_epochs=100, beta=1.0, lr=1e-3, 
                                lambda_stage=0.5, lambda_drug=0.5,
                                feature_threshold=0.5):
    """
    Main function to run the multi-omics integration pipeline
    
    Parameters:
    - rnaseq_adata: AnnData object with RNA-seq data
    - metag_adata: AnnData object with metagenomic data
    - clinical_vars: List of clinical variables to include (must be columns in rnaseq_adata.obs)
    - hidden_dim: Dimension of hidden layers
    - latent_dim: Dimension of latent space
    - num_epochs: Number of training epochs
    - beta: Weight for KL divergence term
    - lr: Learning rate
    - lambda_stage: Weight for cancer stage prediction loss
    - lambda_drug: Weight for drug response prediction loss
    - feature_threshold: Proportion of samples a feature must be important in to be selected (0.5 = 50%)
    
    Returns:
    - model: Trained VAE model
    - latent_features: Extracted latent features for all samples
    - rnaseq_importance: Feature importance scores for RNA-seq features
    - metag_importance: Feature importance scores for metagenomic features
    - rnaseq_selected: Boolean mask of selected RNA-seq features
    - metag_selected: Boolean mask of selected metagenomic features
    - selected_feature_names: Dictionary with lists of selected feature names
    - clinical_info: Information about clinical variables
    - metrics: Dictionary with model performance metrics
    """
    print("Starting multi-omics integration pipeline...")
    
    # Check if samples match
    assert rnaseq_adata.obs_names.equals(metag_adata.obs_names), "Sample IDs don't match between datasets"
    
    # Check for clinical variables
    has_clinical = clinical_vars is not None and len(clinical_vars) > 0
    
    # Determine if we have cancer stage and drug response data
    has_stage = False
    has_drug = False
    
    if has_clinical:
        for var in clinical_vars:
            if var.lower() in ['stage', 'cancer_stage', 'tumor_stage']:
                has_stage = True
            elif var.lower() in ['drug_response', 'response', 'treatment_response']:
                has_drug = True
    
    # Load and preprocess data
    if has_clinical:
        train_loader, test_loader, rnaseq_dim, metag_dim, clinical_dim, clinical_info = load_and_preprocess_data(
            rnaseq_adata, metag_adata, clinical_vars
        )
    else:
        train_loader, test_loader, rnaseq_dim, metag_dim, clinical_dim, clinical_info = load_and_preprocess_data(
            rnaseq_adata, metag_adata
        )
    
    # Initialize model
    print(f"Initializing model with dimensions: RNA-seq={rnaseq_dim}, Metagenomic={metag_dim}, Clinical={clinical_dim}")
    model = MultiOmicsVAE(rnaseq_dim, metag_dim, clinical_dim, hidden_dim, latent_dim)
    
    # Train model
    print("Training VAE model...")
    train_losses, test_losses, train_metrics, test_metrics = train_vae(
        model, train_loader, test_loader, 
        has_clinical=has_clinical, has_stage=has_stage, has_drug=has_drug,
        num_epochs=num_epochs, beta=beta, lr=lr,
        lambda_stage=lambda_stage, lambda_drug=lambda_drug
    )
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot additional metrics if available
    if has_stage and 'stage_acc' in test_metrics and len(test_metrics['stage_acc']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_metrics['stage_acc'], label='Train')
        plt.plot(test_metrics['stage_acc'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Cancer Stage Prediction Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    if has_drug and 'drug_auc' in test_metrics and len(test_metrics['drug_auc']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_metrics['drug_auc'], label='Train')
        plt.plot(test_metrics['drug_auc'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Drug Response Prediction AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Extract latent features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare full data for feature importance analysis
    full_rnaseq_data = rnaseq_adata.X.toarray() if hasattr(rnaseq_adata.X, 'toarray') else rnaseq_adata.X
    full_metag_data = metag_adata.X.toarray() if hasattr(metag_adata.X, 'toarray') else metag_adata.X
    
    if has_clinical:
        # Extract clinical data for full dataset
        clinical_df = rnaseq_adata.obs[clinical_info['included_vars']].copy()
        clinical_data = clinical_df.values
        
        # Create full dataset with clinical data
        full_dataset = TensorDataset(
            torch.FloatTensor(full_rnaseq_data), 
            torch.FloatTensor(full_metag_data),
            torch.FloatTensor(clinical_data)
        )
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
        
        print("Extracting latent features...")
        latent_features = extract_latent_features(model, full_loader, device, has_clinical=True)
    else:
        # Create full dataset without clinical data
        full_dataset = TensorDataset(
            torch.FloatTensor(full_rnaseq_data), 
            torch.FloatTensor(full_metag_data)
        )
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
        
        print("Extracting latent features...")
        latent_features = extract_latent_features(model, full_loader, device, has_clinical=False)
    
    # Get feature importance with sample-level threshold
    print(f"Calculating feature importance with {feature_threshold*100}% sample threshold...")
    rnaseq_importance, metag_importance, rnaseq_selected, metag_selected = feature_importance(
        model, rnaseq_dim, metag_dim, full_rnaseq_data, full_metag_data, threshold=feature_threshold
    )
    
    # Plot latent space with clinical variables
    if has_clinical:
        # Plot latent space colored by stage if available
        if has_stage:
            stage_col = [var for var in clinical_vars if var.lower() in ['stage', 'cancer_stage', 'tumor_stage']][0]
            plot_latent_space(latent_features, rnaseq_adata.obs, color_by=stage_col)
        
        # Plot latent space colored by drug response if available
        if has_drug:
            response_col = [var for var in clinical_vars if var.lower() in ['drug_response', 'response', 'treatment_response']][0]
            plot_latent_space(latent_features, rnaseq_adata.obs, color_by=response_col)
        
        # Plot latent space colored by age if available
        if 'age' in clinical_vars:
            plot_latent_space(latent_features, rnaseq_adata.obs, color_by='age')
        
        # Plot with multiple clinical variables if we have both stage and drug response
        if has_stage and has_drug:
            stage_col = [var for var in clinical_vars if var.lower() in ['stage', 'cancer_stage', 'tumor_stage']][0]
            response_col = [var for var in clinical_vars if var.lower() in ['drug_response', 'response', 'treatment_response']][0]
            plot_latent_space(latent_features, rnaseq_adata.obs, color_by=stage_col, shape_by=response_col)
    else:
        # Basic latent space plot
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
    
    # Feature counts by stage if available
    if has_stage:
        stage_col = [var for var in clinical_vars if var.lower() in ['stage', 'cancer_stage', 'tumor_stage']][0]
        stage_values = rnaseq_adata.obs[stage_col].unique()
        
        print("\nAnalyzing feature importance by cancer stage:")
        # Create stage-specific importance scores
        for stage in stage_values:
            stage_mask = rnaseq_adata.obs[stage_col] == stage
            if sum(stage_mask) < 5:  # Skip if too few samples
                print(f"Skipping stage {stage} analysis (too few samples: {sum(stage_mask)})")
                continue
                
            print(f"Analyzing stage {stage} with {sum(stage_mask)} samples")
            
            # Get stage-specific data
            stage_rnaseq = full_rnaseq_data[stage_mask]
            stage_metag = full_metag_data[stage_mask]
            
            # Analyze this subset
            stage_rnaseq_imp, stage_metag_imp, stage_rnaseq_sel, stage_metag_sel = feature_importance(
                model, rnaseq_dim, metag_dim, stage_rnaseq, stage_metag, threshold=0.7  # Higher threshold for stage-specific
            )
            
            # Store results
            rnaseq_adata.var[f'vae_importance_stage_{stage}'] = stage_rnaseq_imp
            rnaseq_adata.var[f'vae_selected_stage_{stage}'] = stage_rnaseq_sel
            
            metag_adata.var[f'vae_importance_stage_{stage}'] = stage_metag_imp
            metag_adata.var[f'vae_selected_stage_{stage}'] = stage_metag_sel
            
            # Add to selected features
            selected_feature_names[f'rnaseq_stage_{stage}'] = [rnaseq_adata.var_names[i] for i in np.where(stage_rnaseq_sel)[0]]
            selected_feature_names[f'metag_stage_{stage}'] = [metag_adata.var_names[i] for i in np.where(stage_metag_sel)[0]]
    
    # Feature counts by drug response if available
    if has_drug:
        response_col = [var for var in clinical_vars if var.lower() in ['drug_response', 'response', 'treatment_response']][0]
        response_values = rnaseq_adata.obs[response_col].unique()
        
        print("\nAnalyzing feature importance by drug response:")
        # Create response-specific importance scores
        for response in response_values:
            response_mask = rnaseq_adata.obs[response_col] == response
            if sum(response_mask) < 5:  # Skip if too few samples
                print(f"Skipping response {response} analysis (too few samples: {sum(response_mask)})")
                continue
                
            print(f"Analyzing response {response} with {sum(response_mask)} samples")
            
            # Get response-specific data
            response_rnaseq = full_rnaseq_data[response_mask]
            response_metag = full_metag_data[response_mask]
            
            # Analyze this subset
            response_rnaseq_imp, response_metag_imp, response_rnaseq_sel, response_metag_sel = feature_importance(
                model, rnaseq_dim, metag_dim, response_rnaseq, response_metag, threshold=0.7  # Higher threshold for response-specific
            )
            
            # Store results
            rnaseq_adata.var[f'vae_importance_response_{response}'] = response_rnaseq_imp
            rnaseq_adata.var[f'vae_selected_response_{response}'] = response_rnaseq_sel
            
            metag_adata.var[f'vae_importance_response_{response}'] = response_metag_imp
            metag_adata.var[f'vae_selected_response_{response}'] = response_metag_sel
            
            # Add to selected features
            selected_feature_names[f'rnaseq_response_{response}'] = [rnaseq_adata.var_names[i] for i in np.where(response_rnaseq_sel)[0]]
            selected_feature_names[f'metag_response_{response}'] = [metag_adata.var_names[i] for i in np.where(response_metag_sel)[0]]
    
    # Combine all metrics
    metrics = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'selected_rnaseq_count': sum(rnaseq_selected),
        'selected_metag_count': sum(metag_selected),
    }
    
    print("\nIntegration complete!")
    print(f"Extracted {latent_features.shape[1]} latent dimensions from {latent_features.shape[0]} samples")
    print(f"Selected {sum(rnaseq_selected)} RNA-seq features and {sum(metag_selected)} metagenomic features")
    
    if has_stage and 'stage_acc' in test_metrics and len(test_metrics['stage_acc']) > 0:
        print(f"Final stage prediction accuracy: {test_metrics['stage_acc'][-1]:.4f}")
    
    if has_drug and 'drug_auc' in test_metrics and len(test_metrics['drug_auc']) > 0:
        print(f"Final drug response prediction AUC: {test_metrics['drug_auc'][-1]:.4f}")
    
    return model, latent_features, rnaseq_importance, metag_importance, rnaseq_selected, metag_selected, selected_feature_names, clinical_info, metrics



# Example usage with clinical data:
if __name__ == "__main__":
    # Create simulated data
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
    
    # Add clinical metadata
    # Cancer stage (I to V)
    stages = np.random.choice(['I', 'II', 'III', 'IV', 'V'], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    rna_adata.obs['cancer_stage'] = stages
    meta_adata.obs['cancer_stage'] = stages
    
    # Age
    ages = np.random.normal(65, 10, size=n_samples).astype(int)
    ages = np.clip(ages, 30, 90)  # Clip to realistic range
    rna_adata.obs['age'] = ages
    meta_adata.obs['age'] = ages
    
    # Diagnosis age
    diagnosis_ages = ages - np.random.randint(0, 5, size=n_samples)  # Diagnosis within last 5 years
    rna_adata.obs['diagnosis_age'] = diagnosis_ages
    meta_adata.obs['diagnosis_age'] = diagnosis_ages
    
    # Drug response (positive/negative)
    # Make it correlated with stage for realism
    stage_to_prob = {'I': 0.8, 'II': 0.7, 'III': 0.5, 'IV': 0.3, 'V': 0.1}
    drug_response = []
    
    for stage in stages:
        prob_positive = stage_to_prob[stage]
        response = np.random.choice(['positive', 'negative'], p=[prob_positive, 1-prob_positive])
        drug_response.append(response)
    
    rna_adata.obs['drug_response'] = drug_response
    meta_adata.obs['drug_response'] = drug_response
    
    # Normalize data
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    
    sc.pp.normalize_total(meta_adata, target_sum=1e4)
    sc.pp.log1p(meta_adata)
    
    # Run multi-omics integration with clinical data
    clinical_vars = ['cancer_stage', 'age', 'diagnosis_age', 'drug_response']
    
    model, latent_features, rna_importance, meta_importance, rna_selected, meta_selected, selected_features, clinical_info, metrics = run_multi_omics_integration(
        rna_adata, meta_adata, 
        clinical_vars=clinical_vars,
        hidden_dim=128, 
        latent_dim=20, 
        num_epochs=50, 
        beta=0.5,
        lambda_stage=1.0,
        lambda_drug=1.0,
        feature_threshold=0.5
    )
    
    print("Integration complete!")
    
    # Generate UMAP visualization of latent space
    reducer = umap.UMAP(random_state=42)
    umap_embedding = reducer.fit_transform(latent_features)
    
    # Add UMAP to AnnData objects
    rna_adata.obsm['X_umap'] = umap_embedding
    meta_adata.obsm['X_umap'] = umap_embedding
    
    # Plot UMAP with stage and drug response
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                      c=[{'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4}[s] for s in rna_adata.obs['cancer_stage']], 
                      cmap='viridis', s=100, alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['I', 'II', 'III', 'IV', 'V'])
    plt.title('UMAP of Latent Space by Cancer Stage')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    plt.subplot(1, 2, 2)
    colors = {'positive': 'green', 'negative': 'red'}
    for response, color in colors.items():
        mask = rna_adata.obs['drug_response'] == response
        plt.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1], 
                   color=color, label=response, s=100, alpha=0.8)
    plt.title('UMAP of Latent Space by Drug Response')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print feature counts
    print("\nFeature Selection Summary:")
    for key, features in selected_features.items():
        print(f"{key}: {len(features)} features selected")
        
    # Print overlapping features between stages
    if 'rnaseq_stage_I' in selected_features:
        print("\nRNA-seq Feature Overlap Between Cancer Stages:")
        for i, stage1 in enumerate(['I', 'II', 'III', 'IV', 'V']):
            if f'rnaseq_stage_{stage1}' not in selected_features:
                continue
            for stage2 in ['I', 'II', 'III', 'IV', 'V'][i+1:]:
                if f'rnaseq_stage_{stage2}' not in selected_features:
                    continue
                set1 = set(selected_features[f'rnaseq_stage_{stage1}'])
                set2 = set(selected_features[f'rnaseq_stage_{stage2}'])
                overlap = set1.intersection(set2)
                print(f"Stage {stage1} and {stage2}: {len(overlap)} shared features out of {len(set1)} and {len(set2)}")
    
    # Print model performance
    if 'stage_acc' in metrics['test_metrics'] and len(metrics['test_metrics']['stage_acc']) > 0:
        print(f"\nFinal cancer stage prediction accuracy: {metrics['test_metrics']['stage_acc'][-1]:.4f}")
    
    if 'drug_auc' in metrics['test_metrics'] and len(metrics['test_metrics']['drug_auc']) > 0:
        print(f"Final drug response prediction AUC: {metrics['test_metrics']['drug_auc'][-1]:.4f}")
