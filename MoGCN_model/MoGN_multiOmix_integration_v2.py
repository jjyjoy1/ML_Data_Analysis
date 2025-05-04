import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# 1. Data Preparation Functions
def load_omics_data(gene_exp_file, methylation_file, mutation_file, clinical_file):
    """
    Load multi-omics data from files
    
    Args:
        gene_exp_file: Path to gene expression data (samples x genes)
        methylation_file: Path to methylation data (samples x CpG sites)
        mutation_file: Path to mutation data (samples x genes)
        clinical_file: Path to clinical data with outcomes
        
    Returns:
        Dictionary containing dataframes for each data type
    """
    gene_exp = pd.read_csv(gene_exp_file, index_col=0)
    methylation = pd.read_csv(methylation_file, index_col=0)
    mutation = pd.read_csv(mutation_file, index_col=0)
    clinical = pd.read_csv(clinical_file, index_col=0)
    
    # Ensure all datasets have the same samples and in the same order
    common_samples = list(set(gene_exp.index) & set(methylation.index) & 
                         set(mutation.index) & set(clinical.index))
    
    gene_exp = gene_exp.loc[common_samples]
    methylation = methylation.loc[common_samples]
    mutation = mutation.loc[common_samples]
    clinical = clinical.loc[common_samples]
    
    print(f"Number of samples after alignment: {len(common_samples)}")
    
    return {
        'gene_expression': gene_exp,
        'methylation': methylation,
        'mutation': mutation,
        'clinical': clinical
    }


def preprocess_omics_data(omics_data):
    """
    Preprocess multi-omics data (normalization, etc.)
    
    Args:
        omics_data: Dictionary of omics dataframes
        
    Returns:
        Dictionary of preprocessed data
    """
    preprocessed = {}
    
    # Scale gene expression data
    scaler = StandardScaler()
    gene_exp_scaled = scaler.fit_transform(omics_data['gene_expression'])
    preprocessed['gene_expression'] = gene_exp_scaled
    
    # Scale methylation data
    meth_scaled = scaler.fit_transform(omics_data['methylation'])
    preprocessed['methylation'] = meth_scaled
    
    # Leave mutation data as binary
    preprocessed['mutation'] = omics_data['mutation'].values
    
    # Extract outcome (e.g., survival status)
    preprocessed['outcome'] = omics_data['clinical']['survival_status'].values
    
    return preprocessed


def create_similarity_graphs(omics_data, k=10):
    """
    Create sample similarity graphs for each omics data type
    
    Args:
        omics_data: Dictionary of preprocessed omics data arrays
        k: Number of nearest neighbors for graph construction
        
    Returns:
        Dictionary of adjacency matrices and edge indices for PyTorch Geometric
    """
    graphs = {}
    
    for data_type, data in omics_data.items():
        if data_type == 'outcome':
            continue
            
        # Compute pairwise cosine similarity between samples
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(data)
        
        # Keep only top k neighbors for each sample
        n_samples = similarity.shape[0]
        adj_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # Get indices of top k similar samples (excluding self)
            indices = np.argsort(similarity[i])[::-1][1:k+1]
            adj_matrix[i, indices] = similarity[i, indices]
            adj_matrix[indices, i] = similarity[i, indices]  # Make symmetric
        
        # Convert to edge index format for PyTorch Geometric
        edge_index = []
        edge_weight = []
        
        for i in range(n_samples):
            for j in range(n_samples):
                if adj_matrix[i, j] > 0:
                    edge_index.append([i, j])
                    edge_weight.append(adj_matrix[i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        graphs[data_type] = {
            'adj_matrix': adj_matrix,
            'edge_index': edge_index,
            'edge_weight': edge_weight
        }
    
    return graphs


# 2. Model Definition
class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer"""
    
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_features, out_features)
        
    def forward(self, x, edge_index, edge_weight=None):
        return F.relu(self.conv(x, edge_index, edge_weight))


class MoGCN(nn.Module):
    """Multi-omics Graph Convolutional Network"""
    
    def __init__(self, feature_dims, hidden_dims, output_dim):
        """
        Args:
            feature_dims: Dictionary of input feature dimensions for each omics type
            hidden_dims: List of hidden dimensions for GCN layers
            output_dim: Dimension of output (e.g., 1 for binary classification)
        """
        super(MoGCN, self).__init__()
        
        # Create GCN modules for each omics type
        self.omics_gcns = nn.ModuleDict()
        self.feature_dims = feature_dims
        
        for omics_type, dim in feature_dims.items():
            # Create a 2-layer GCN for each omics type
            self.omics_gcns[omics_type] = nn.ModuleList([
                GCNLayer(dim, hidden_dims[0]),
                GCNLayer(hidden_dims[0], hidden_dims[1])
            ])
        
        # Fusion layer to combine features from different omics types
        self.fusion_layer = nn.Linear(hidden_dims[1] * len(feature_dims), hidden_dims[1])
        
        # Final prediction layer
        self.prediction_layer = nn.Linear(hidden_dims[1], output_dim)
        
    def forward(self, data_dict):
        """
        Forward pass through the MoGCN
        
        Args:
            data_dict: Dictionary containing data tensors and graph structures for each omics type
            
        Returns:
            Output prediction
        """
        # Process each omics type through its respective GCN
        omics_embeddings = []
        
        for omics_type, modules in self.omics_gcns.items():
            x = data_dict[omics_type]['features']
            edge_index = data_dict[omics_type]['edge_index']
            edge_weight = data_dict[omics_type]['edge_weight']
            
            # Apply GCN layers
            for gcn_layer in modules:
                x = gcn_layer(x, edge_index, edge_weight)
            
            # Global mean pooling to get graph-level representations
            omics_embeddings.append(x)
        
        # Concatenate embeddings from different omics types
        combined_embedding = torch.cat(omics_embeddings, dim=1)
        
        # Fusion layer to integrate multi-omics information
        fused_embedding = F.relu(self.fusion_layer(combined_embedding))
        
        # Final prediction
        output = self.prediction_layer(fused_embedding)
        
        return output


# 3. Training and Evaluation Functions
def prepare_pytorch_data(preprocessed_data, graphs):
    """
    Prepare data for PyTorch models
    
    Args:
        preprocessed_data: Dictionary of preprocessed omics data
        graphs: Dictionary of graph structures
        
    Returns:
        Dictionary of PyTorch tensors and graph structures
    """
    data_dict = {}
    
    for data_type in preprocessed_data:
        if data_type == 'outcome':
            continue
            
        features = torch.tensor(preprocessed_data[data_type], dtype=torch.float)
        edge_index = graphs[data_type]['edge_index']
        edge_weight = graphs[data_type]['edge_weight']
        
        data_dict[data_type] = {
            'features': features,
            'edge_index': edge_index,
            'edge_weight': edge_weight
        }
    
    # Convert outcome to tensor
    labels = torch.tensor(preprocessed_data['outcome'], dtype=torch.float).view(-1, 1)
    
    return data_dict, labels


def train_mogcn_model(data_dict, labels, model, epochs=100, lr=0.001):
    """
    Train the MoGCN model
    
    Args:
        data_dict: Dictionary of PyTorch tensors and graph structures
        labels: Tensor of outcome labels
        model: MoGCN model instance
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    # Split data into train and validation sets
    n_samples = labels.shape[0]
    indices = list(range(n_samples))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(data_dict)
        
        train_loss = criterion(outputs[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(data_dict)
            val_loss = criterion(val_outputs[val_idx], labels[val_idx])
            
            # Calculate AUC
            val_probs = torch.sigmoid(val_outputs[val_idx]).numpy()
            val_auc = roc_auc_score(labels[val_idx].numpy(), val_probs)
        
        # Record metrics
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_auc'].append(val_auc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"Val AUC: {val_auc:.4f}")
    
    return model, history


def evaluate_model(model, data_dict, labels, test_idx):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained MoGCN model
        data_dict: Dictionary of PyTorch tensors and graph structures
        labels: Tensor of outcome labels
        test_idx: Indices of test samples
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        outputs = model(data_dict)
        probs = torch.sigmoid(outputs[test_idx]).numpy()
        preds = (probs > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels[test_idx].numpy(), preds)
        auc = roc_auc_score(labels[test_idx].numpy(), probs)
        
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': preds,
        'probabilities': probs
    }


# 4. Main Function to Run the Pipeline
def run_mogcn_pipeline(gene_exp_file, methylation_file, mutation_file, clinical_file):
    """
    Run the complete MoGCN pipeline
    
    Args:
        gene_exp_file: Path to gene expression data
        methylation_file: Path to methylation data
        mutation_file: Path to mutation data
        clinical_file: Path to clinical data
        
    Returns:
        Trained model and evaluation results
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    omics_data = load_omics_data(gene_exp_file, methylation_file, mutation_file, clinical_file)
    preprocessed_data = preprocess_omics_data(omics_data)
    
    # Create similarity graphs
    print("Creating sample similarity graphs...")
    graphs = create_similarity_graphs(preprocessed_data)
    
    # Prepare PyTorch data
    data_dict, labels = prepare_pytorch_data(preprocessed_data, graphs)
    
    # Define feature dimensions (number of features for each omics type)
    feature_dims = {
        'gene_expression': preprocessed_data['gene_expression'].shape[1],
        'methylation': preprocessed_data['methylation'].shape[1],
        'mutation': preprocessed_data['mutation'].shape[1]
    }
    
    # Initialize model
    print("Initializing MoGCN model...")
    model = MoGCN(
        feature_dims=feature_dims,
        hidden_dims=[128, 64],
        output_dim=1
    )
    
    # Train model
    print("Training model...")
    trained_model, history = train_mogcn_model(data_dict, labels, model, epochs=100)
    
    # Evaluate on test set
    print("Evaluating model...")
    n_samples = labels.shape[0]
    _, test_idx = train_test_split(range(n_samples), test_size=0.2, random_state=42)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    
    results = evaluate_model(trained_model, data_dict, labels, test_idx)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")
    
    return trained_model, results, history


# Example usage (with synthetic data for demonstration)
if __name__ == "__main__":
    import os
    
    # Generate synthetic multi-omics data for demonstration
    def generate_synthetic_data(n_samples=100, n_genes=200, n_cpg=200, outcome_corr=0.7):
        """Generate synthetic multi-omics data for testing"""
        # Generate random gene expression data
        gene_exp = np.random.normal(0, 1, (n_samples, n_genes))
        
        # Generate random methylation data
        methylation = np.random.normal(0, 1, (n_samples, n_cpg))
        
        # Generate random mutation data (binary)
        mutation = np.random.binomial(1, 0.05, (n_samples, n_genes))
        
        # Generate outcome with some correlation to the data
        outcome_features = 0.6 * gene_exp[:, :10].mean(axis=1) - 0.4 * methylation[:, :10].mean(axis=1)
        outcome_features = outcome_features + np.random.normal(0, 1 - outcome_corr, n_samples)
        outcome = (outcome_features > 0).astype(int)
        
        # Create dataframes
        sample_ids = [f"PATIENT_{i}" for i in range(n_samples)]
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        cpg_names = [f"CPG_{i}" for i in range(n_cpg)]
        
        gene_exp_df = pd.DataFrame(gene_exp, index=sample_ids, columns=gene_names)
        methylation_df = pd.DataFrame(methylation, index=sample_ids, columns=cpg_names)
        mutation_df = pd.DataFrame(mutation, index=sample_ids, columns=gene_names)
        clinical_df = pd.DataFrame({
            'survival_status': outcome,
            'age': np.random.normal(60, 10, n_samples).astype(int),
            'gender': np.random.binomial(1, 0.5, n_samples)
        }, index=sample_ids)
        
        # Save to CSV files
        os.makedirs("synthetic_data", exist_ok=True)
        gene_exp_df.to_csv("synthetic_data/gene_expression.csv")
        methylation_df.to_csv("synthetic_data/methylation.csv")
        mutation_df.to_csv("synthetic_data/mutation.csv")
        clinical_df.to_csv("synthetic_data/clinical.csv")
        
        print("Synthetic data generated and saved to 'synthetic_data' directory")
    
    # Generate synthetic data
    generate_synthetic_data(n_samples=100, n_genes=200, n_cpg=200)
    
    # Run MoGCN pipeline
    trained_model, results, history = run_mogcn_pipeline(
        gene_exp_file="synthetic_data/gene_expression.csv",
        methylation_file="synthetic_data/methylation.csv",
        mutation_file="synthetic_data/mutation.csv",
        clinical_file="synthetic_data/clinical.csv"
    )
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation AUC')
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()




