import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.stats import kstest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Load and preprocess RNA-seq and survival data
# This is a placeholder for actual data loading code
def load_data(rna_seq_path, survival_path, pathway_data_path):
    """
    Load RNA-seq expression data, survival data and pathway information
    
    Parameters:
    -----------
    rna_seq_path : str
        Path to RNA-seq gene expression matrix (samples x genes)
    survival_path : str
        Path to survival data with patient IDs, survival time, and event status
    pathway_data_path : str
        Path to pathway data mapping genes to functional modules/pathways
        
    Returns:
    --------
    rna_seq_data : pd.DataFrame
        Preprocessed RNA-seq data
    survival_data : pd.DataFrame
        Preprocessed survival data
    pathway_data : dict
        Dictionary mapping pathways to genes
    gene_to_index : dict
        Dictionary mapping gene names to indices in RNA-seq data
    """
    # Load RNA-seq data (samples x genes)
    rna_seq_data = pd.read_csv(rna_seq_path, index_col=0)
    
    # Load survival data
    survival_data = pd.read_csv(survival_path, index_col=0)
    
    # Make sure survival data and RNA-seq data have matching patient IDs
    common_patients = list(set(rna_seq_data.index) & set(survival_data.index))
    rna_seq_data = rna_seq_data.loc[common_patients]
    survival_data = survival_data.loc[common_patients]
    
    # Load pathway data
    pathway_data = {}
    with open(pathway_data_path, 'r') as f:
        for line in f:
            pathway, genes = line.strip().split('\t')
            pathway_data[pathway] = genes.split(',')
    
    # Create gene to index mapping
    gene_to_index = {gene: i for i, gene in enumerate(rna_seq_data.columns)}
    
    # Log transform and normalize RNA-seq data if needed
    # rna_seq_data = np.log2(rna_seq_data + 1)
    scaler = StandardScaler()
    rna_seq_data = pd.DataFrame(
        scaler.fit_transform(rna_seq_data),
        index=rna_seq_data.index,
        columns=rna_seq_data.columns
    )
    
    return rna_seq_data, survival_data, pathway_data, gene_to_index

# 2. Create a dataset class for RNA-seq and survival data
class RNASeqSurvivalDataset(Dataset):
    """
    Dataset class for RNA-seq expression and survival data
    """
    def __init__(self, rna_seq_data, survival_data):
        """
        Initialize dataset
        
        Parameters:
        -----------
        rna_seq_data : pd.DataFrame or np.ndarray
            RNA-seq expression data (samples x genes)
        survival_data : pd.DataFrame
            Survival data with time and event columns
        """
        self.rna_seq = torch.tensor(rna_seq_data.values, dtype=torch.float32)
        self.survival_time = torch.tensor(
            survival_data['time'].values, dtype=torch.float32
        )
        self.survival_event = torch.tensor(
            survival_data['event'].values, dtype=torch.float32
        )
        
    def __len__(self):
        return len(self.rna_seq)
    
    def __getitem__(self, idx):
        return {
            'rna_seq': self.rna_seq[idx],
            'time': self.survival_time[idx],
            'event': self.survival_event[idx]
        }

# 3. Build the DeepOmix model architecture
class DeepOmix(nn.Module):
    """
    Implementation of DeepOmix-inspired model for RNA-seq and survival analysis
    """
    def __init__(self, input_dim, pathway_connections, hidden_dims=[128, 64]):
        """
        Initialize DeepOmix model
        
        Parameters:
        -----------
        input_dim : int
            Number of genes in the input data
        pathway_connections : dict
            Dictionary mapping each pathway node to a list of gene indices
        hidden_dims : list
            Dimensions of hidden layers
        """
        super(DeepOmix, self).__init__()
        
        # Store the pathway connections for later analysis
        self.pathway_connections = pathway_connections
        self.num_pathways = len(pathway_connections)
        
        # Gene layer to pathway layer (first biological knowledge layer)
        self.gene_to_pathway = nn.Linear(input_dim, self.num_pathways, bias=True)
        
        # Initialize the gene-to-pathway connections based on biological knowledge
        # All weights are initialized to near zero
        with torch.no_grad():
            self.gene_to_pathway.weight.fill_(0.01)
            
            # Set weights according to pathway connections
            for pathway_idx, gene_indices in pathway_connections.items():
                for gene_idx in gene_indices:
                    # Set a higher initial weight for known gene-pathway connections
                    self.gene_to_pathway.weight[pathway_idx, gene_idx] = 0.1
        
        # Pathway activation function (ReLU to model pathway activation)
        self.pathway_activation = nn.ReLU()
        
        # Functional module layers (deeper representation)
        self.layers = nn.ModuleList()
        
        # First hidden layer receives input from pathway layer
        self.layers.append(nn.Linear(self.num_pathways, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(0.3))
        
        # Additional hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.Dropout(0.3))
        
        # Output layer for risk score
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Batch of RNA-seq expression data
            
        Returns:
        --------
        risk_scores : torch.Tensor
            Predicted risk scores
        pathway_activations : torch.Tensor
            Activations of the pathway layer
        """
        # Gene layer to pathway layer
        pathway_layer = self.gene_to_pathway(x)
        pathway_activations = self.pathway_activation(pathway_layer)
        
        # Forward through the rest of the network
        out = pathway_activations
        for layer in self.layers:
            out = layer(out)
        
        # Output risk score
        risk_scores = self.output_layer(out)
        
        return risk_scores, pathway_activations
    
    def get_pathway_weights(self):
        """
        Get the weights connecting genes to pathways
        
        Returns:
        --------
        weights : torch.Tensor
            Weights of gene-to-pathway connections
        """
        return self.gene_to_pathway.weight.detach()
    
    def get_pathway_importance(self, x):
        """
        Calculate the importance of each pathway for a set of samples
        
        Parameters:
        -----------
        x : torch.Tensor
            Batch of RNA-seq expression data
            
        Returns:
        --------
        pathway_importance : torch.Tensor
            Importance scores for each pathway
        """
        with torch.no_grad():
            _, pathway_activations = self.forward(x)
            
            # Calculate mean activation for each pathway
            pathway_importance = pathway_activations.mean(dim=0)
            
        return pathway_importance

# 4. Define the loss function for survival analysis
class NegativeLogLikelihood(nn.Module):
    """
    Negative Log Likelihood loss for Cox proportional hazards model
    """
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        
    def forward(self, risk_scores, survival_time, survival_event):
        """
        Calculate the negative log likelihood loss
        
        Parameters:
        -----------
        risk_scores : torch.Tensor
            Predicted risk scores
        survival_time : torch.Tensor
            Survival time for each patient
        survival_event : torch.Tensor
            Event status for each patient (1 for event, 0 for censored)
            
        Returns:
        --------
        loss : torch.Tensor
            Negative log likelihood loss
        """
        # Sort by survival time in descending order
        sorted_indices = torch.argsort(survival_time, descending=True)
        risk_scores = risk_scores[sorted_indices]
        survival_event = survival_event[sorted_indices]
        
        # Calculate log likelihood
        log_risk = risk_scores
        cumsum_risk = torch.cumsum(torch.exp(log_risk), dim=0)
        log_cumsum_risk = torch.log(cumsum_risk)
        
        # Uncensored likelihood: log(risk) - log(sum(exp(risk)))
        uncensored_likelihood = log_risk - log_cumsum_risk
        
        # Apply event mask
        neg_likelihood = -torch.sum(uncensored_likelihood * survival_event)
        return neg_likelihood

# 5. Create a function to map genes to pathways
def create_pathway_connections(gene_list, pathway_data, gene_to_index):
    """
    Create a dictionary mapping pathway indices to lists of gene indices
    
    Parameters:
    -----------
    gene_list : list
        List of gene names in the RNA-seq data
    pathway_data : dict
        Dictionary mapping pathways to lists of genes
    gene_to_index : dict
        Dictionary mapping gene names to indices in RNA-seq data
        
    Returns:
    --------
    pathway_connections : dict
        Dictionary mapping pathway indices to lists of gene indices
    pathway_to_name : dict
        Dictionary mapping pathway indices to pathway names
    """
    pathway_connections = {}
    pathway_to_name = {}
    
    for idx, (pathway, genes) in enumerate(pathway_data.items()):
        # Find gene indices that are in both the pathway and our RNA-seq data
        valid_genes = [gene for gene in genes if gene in gene_to_index]
        gene_indices = [gene_to_index[gene] for gene in valid_genes]
        
        if gene_indices:  # Only add pathways with at least one matching gene
            pathway_connections[idx] = gene_indices
            pathway_to_name[idx] = pathway
    
    return pathway_connections, pathway_to_name

# 6. Training function
def train_deepomix(model, train_loader, val_loader, learning_rate=0.001, num_epochs=100, patience=10):
    """
    Train the DeepOmix model
    
    Parameters:
    -----------
    model : DeepOmix
        DeepOmix model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    learning_rate : float
        Learning rate for optimizer
    num_epochs : int
        Number of training epochs
    patience : int
        Number of epochs to wait for improvement before early stopping
        
    Returns:
    --------
    model : DeepOmix
        Trained DeepOmix model
    training_history : dict
        Dictionary containing training and validation losses
    """
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = NegativeLogLikelihood()
    
    # For tracking training progress
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_c_index': []
    }
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Get data from batch
            x = batch['rna_seq']
            time = batch['time']
            event = batch['event']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            risk_scores, _ = model(x)
            risk_scores = risk_scores.squeeze()
            
            # Calculate loss
            loss = criterion(risk_scores, time, event)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        training_history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_risk_scores = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data from batch
                x = batch['rna_seq']
                time = batch['time']
                event = batch['event']
                
                # Forward pass
                risk_scores, _ = model(x)
                risk_scores = risk_scores.squeeze()
                
                # Calculate loss
                loss = criterion(risk_scores, time, event)
                val_loss += loss.item()
                
                # Store predictions and targets for calculating c-index
                all_risk_scores.append(risk_scores)
                all_times.append(time)
                all_events.append(event)
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        training_history['val_loss'].append(val_loss)
        
        # Calculate concordance index (c-index)
        all_risk_scores = torch.cat(all_risk_scores).cpu().numpy()
        all_times = torch.cat(all_times).cpu().numpy()
        all_events = torch.cat(all_events).cpu().numpy()
        c_index = concordance_index(all_times, -all_risk_scores, all_events)
        training_history['val_c_index'].append(c_index)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val C-index: {c_index:.4f}")
        
        # Check for improvement for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_history

# 7. Evaluation function
def evaluate_deepomix(model, test_loader, pathway_to_name=None):
    """
    Evaluate the DeepOmix model on test data
    
    Parameters:
    -----------
    model : DeepOmix
        Trained DeepOmix model
    test_loader : DataLoader
        DataLoader for test data
    pathway_to_name : dict, optional
        Dictionary mapping pathway indices to pathway names
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_risk_scores = []
    all_times = []
    all_events = []
    all_pathway_activations = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get data from batch
            x = batch['rna_seq']
            time = batch['time']
            event = batch['event']
            
            # Forward pass
            risk_scores, pathway_activations = model(x)
            risk_scores = risk_scores.squeeze()
            
            # Store predictions and targets
            all_risk_scores.append(risk_scores)
            all_times.append(time)
            all_events.append(event)
            all_pathway_activations.append(pathway_activations)
    
    # Concatenate batches
    all_risk_scores = torch.cat(all_risk_scores).cpu().numpy()
    all_times = torch.cat(all_times).cpu().numpy()
    all_events = torch.cat(all_events).cpu().numpy()
    all_pathway_activations = torch.cat(all_pathway_activations).cpu().numpy()
    
    # Calculate concordance index (c-index)
    c_index = concordance_index(all_times, -all_risk_scores, all_events)
    
    # Classify patients into high and low risk groups based on median risk score
    median_risk = np.median(all_risk_scores)
    risk_groups = (all_risk_scores >= median_risk).astype(int)
    
    # Calculate pathway importance
    pathway_importance = np.mean(all_pathway_activations, axis=0)
    
    # Identify important pathways
    important_pathways = []
    if pathway_to_name is not None:
        sorted_indices = np.argsort(-pathway_importance)
        for idx in sorted_indices:
            if idx in pathway_to_name:
                important_pathways.append({
                    'pathway': pathway_to_name[idx],
                    'importance': pathway_importance[idx]
                })
    
    # Return evaluation results
    results = {
        'c_index': c_index,
        'risk_scores': all_risk_scores,
        'risk_groups': risk_groups,
        'times': all_times,
        'events': all_events,
        'pathway_activations': all_pathway_activations,
        'pathway_importance': pathway_importance,
        'important_pathways': important_pathways
    }
    
    return results

# 8. Survival analysis and visualization
def perform_survival_analysis(times, events, risk_groups):
    """
    Perform survival analysis comparing high and low risk groups
    
    Parameters:
    -----------
    times : np.ndarray
        Survival times
    events : np.ndarray
        Event indicators (1 for event, 0 for censored)
    risk_groups : np.ndarray
        Risk group assignments (1 for high risk, 0 for low risk)
        
    Returns:
    --------
    p_value : float
        Log-rank test p-value
    """
    # Create a DataFrame for Cox model
    df = pd.DataFrame({
        'time': times,
        'event': events,
        'risk_group': risk_groups
    })
    
    # Fit Cox proportional hazards model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    
    # Get summary
    summary = cph.summary
    p_value = summary.loc['risk_group', 'p']
    hr = np.exp(summary.loc['risk_group', 'coef'])
    
    # Plot Kaplan-Meier curves
    kmf = KaplanMeierFitter()
    
    # High risk group
    high_risk = (risk_groups == 1)
    kmf.fit(times[high_risk], events[high_risk], label='High Risk')
    ax = kmf.plot_survival_function()
    
    # Low risk group
    low_risk = (risk_groups == 0)
    kmf.fit(times[low_risk], events[low_risk], label='Low Risk')
    kmf.plot_survival_function(ax=ax)
    
    plt.title(f'Kaplan-Meier Curves by Risk Group\nHR={hr:.2f}, p={p_value:.4f}')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.grid(alpha=0.3)
    
    return p_value

# 9. Pathway analysis
def analyze_pathways(model, test_data, pathway_to_name, risk_groups, top_n=10):
    """
    Analyze pathway importance between high and low risk groups
    
    Parameters:
    -----------
    model : DeepOmix
        Trained DeepOmix model
    test_data : torch.Tensor
        Test data RNA-seq expression values
    pathway_to_name : dict
        Dictionary mapping pathway indices to pathway names
    risk_groups : np.ndarray
        Risk group assignments (1 for high risk, 0 for low risk)
    top_n : int
        Number of top pathways to show
        
    Returns:
    --------
    top_pathways : list
        List of top pathways and their importance scores
    """
    model.eval()
    
    # Get high and low risk data
    high_risk_data = test_data[risk_groups == 1]
    low_risk_data = test_data[risk_groups == 0]
    
    # Get pathway activations for each group
    with torch.no_grad():
        _, high_risk_pathway_act = model(high_risk_data)
        _, low_risk_pathway_act = model(low_risk_data)
    
    # Calculate mean activations
    high_risk_mean = high_risk_pathway_act.mean(dim=0).cpu().numpy()
    low_risk_mean = low_risk_pathway_act.mean(dim=0).cpu().numpy()
    
    # Calculate difference between groups
    pathway_diff = high_risk_mean - low_risk_mean
    
    # Create a DataFrame for pathway analysis
    pathway_df = pd.DataFrame({
        'Pathway': [pathway_to_name.get(i, f"Pathway {i}") for i in range(len(pathway_diff))],
        'High_Risk_Mean': high_risk_mean,
        'Low_Risk_Mean': low_risk_mean,
        'Difference': pathway_diff
    })
    
    # Sort by absolute difference
    pathway_df['Abs_Difference'] = np.abs(pathway_df['Difference'])
    pathway_df = pathway_df.sort_values('Abs_Difference', ascending=False)
    
    # Plot top pathways
    top_paths = pathway_df.head(top_n).copy()
    top_paths = top_paths.sort_values('Difference')
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x > 0 else 'blue' for x in top_paths['Difference']]
    plt.barh(top_paths['Pathway'], top_paths['Difference'], color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Difference in Pathway Activation (High Risk - Low Risk)')
    plt.title(f'Top {top_n} Differentially Activated Pathways Between Risk Groups')
    plt.tight_layout()
    
    return pathway_df

# 10. Main function to run the analysis
def run_deepomix_analysis(rna_seq_path, survival_path, pathway_data_path):
    """
    Run the complete DeepOmix analysis pipeline
    
    Parameters:
    -----------
    rna_seq_path : str
        Path to RNA-seq expression data
    survival_path : str
        Path to survival data
    pathway_data_path : str
        Path to pathway gene sets data
        
    Returns:
    --------
    model : DeepOmix
        Trained DeepOmix model
    results : dict
        Dictionary containing evaluation results
    """
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    rna_seq_data, survival_data, pathway_data, gene_to_index = load_data(
        rna_seq_path, survival_path, pathway_data_path
    )
    
    # 2. Create pathway connections
    print("Creating pathway connections...")
    pathway_connections, pathway_to_name = create_pathway_connections(
        rna_seq_data.columns, pathway_data, gene_to_index
    )
    
    # 3. Split data into train, validation, and test sets
    patient_ids = rna_seq_data.index
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)
    
    # Create datasets
    train_dataset = RNASeqSurvivalDataset(
        rna_seq_data.loc[train_ids],
        survival_data.loc[train_ids]
    )
    val_dataset = RNASeqSurvivalDataset(
        rna_seq_data.loc[val_ids],
        survival_data.loc[val_ids]
    )
    test_dataset = RNASeqSurvivalDataset(
        rna_seq_data.loc[test_ids],
        survival_data.loc[test_ids]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Initialize the model
    print("Initializing DeepOmix model...")
    model = DeepOmix(
        input_dim=rna_seq_data.shape[1],
        pathway_connections=pathway_connections,
        hidden_dims=[128, 64]
    )
    
    # 5. Train the model
    print("Training DeepOmix model...")
    model, training_history = train_deepomix(
        model, train_loader, val_loader,
        learning_rate=0.001,
        num_epochs=100,
        patience=10
    )
    
    # 6. Evaluate the model
    print("Evaluating DeepOmix model...")
    results = evaluate_deepomix(model, test_loader, pathway_to_name)
    
    # 7. Perform survival analysis
    print("Performing survival analysis...")
    p_value = perform_survival_analysis(
        results['times'], results['events'], results['risk_groups']
    )
    results['log_rank_p_value'] = p_value
    
    # 8. Analyze pathways
    print("Analyzing pathways...")
    test_data = torch.tensor(rna_seq_data.loc[test_ids].values, dtype=torch.float32)
    pathway_df = analyze_pathways(
        model, test_data, pathway_to_name, results['risk_groups'], top_n=10
    )
    results['pathway_analysis'] = pathway_df
    
    # 9. Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_history['val_c_index'], label='Validation C-index')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title('Validation C-index')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    print(f"Analysis completed! Test C-index: {results['c_index']:.4f}")
    
    return model, results, training_history

# Example usage:
# model, results, history = run_deepomix_analysis('rna_seq_data.csv', 'survival_data.csv', 'pathway_data.txt')

# If you want to demonstrate usage with synthetic data:

def generate_synthetic_data(n_samples=200, n_genes=1000, n_pathways=50, seed=42):
    """
    Generate synthetic RNA-seq and survival data for demonstration
    
    Parameters:
    -----------
    n_samples : int
        Number of samples/patients
    n_genes : int
        Number of genes
    n_pathways : int
        Number of pathways
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    rna_seq_data : pd.DataFrame
        Synthetic RNA-seq data
    survival_data : pd.DataFrame
        Synthetic survival data
    pathway_data : dict
        Synthetic pathway data
    """
    np.random.seed(seed)
    
    # Generate RNA-seq data
    # Random gene expression values with some correlation structure
    gene_expr = np.random.normal(0, 1, (n_samples, n_genes))
    
    # Patient IDs
    patient_ids = [f'TCGA-{i:04d}' for i in range(n_samples)]
    
    # Gene names
    gene_names = [f'GENE_{i}' for i in range(n_genes)]
    
    # Create RNA-seq DataFrame
    rna_seq_data = pd.DataFrame(gene_expr, index=patient_ids, columns=gene_names)
    
    # Generate pathway data
    pathway_data = {}
    genes_per_pathway = np.random.randint(10, 50, n_pathways)
    
    for i in range(n_pathways):
        pathway_name = f'PATHWAY_{i}'
        # Randomly select genes for this pathway
        pathway_genes = np.random.choice(gene_names, genes_per_pathway[i], replace=False)
        pathway_data[pathway_name] = list(pathway_genes)
    
    # Generate survival data
    # Create some driver genes that affect survival
    driver_genes = np.random.choice(n_genes, 20, replace=False)
    driver_weights = np.random.uniform(-1, 1, 20)
    
    # Calculate risk scores based on driver genes
    risk_scores = np.zeros(n_samples)
    for i, gene_idx in enumerate(driver_genes):
        risk_scores += gene_expr[:, gene_idx] * driver_weights[i]
    
    # Generate survival times based on risk scores
    # Higher risk score -> shorter survival time
    baseline_hazard = 0.1
    scale = 1.0 / (baseline_hazard * np.exp(risk_scores))
    
    # Generate survival times from exponential distribution
    survival_times = np.random.exponential(scale=scale)
    
    # Generate censoring times
    censoring_times = np.random.exponential(scale=scale.mean() * 1.5)
    
    # Observed time is the minimum of survival time and censoring time
    observed_times = np.minimum(survival_times, censoring_times)
    
    # Event indicator (1 if observed, 0 if censored)
    events = (survival_times <= censoring_times).astype(int)
    
    # Create survival DataFrame
    survival_data = pd.DataFrame({
        'time': observed_times,
        'event': events
    }, index=patient_ids)
    
    return rna_seq_data, survival_data, pathway_data

# 11. Additional functions for pathway importance analysis

def compute_pathway_importance(model, data_loader):
    """
    Compute pathway importance scores for all samples
    
    Parameters:
    -----------
    model : DeepOmix
        Trained DeepOmix model
    data_loader : DataLoader
        DataLoader for the dataset
        
    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with pathway importance scores for each sample
    """
    model.eval()
    all_pathway_activations = []
    all_patient_ids = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Get batch data
            x = batch['rna_seq']
            
            # Get patient IDs for this batch
            batch_size = x.shape[0]
            batch_start = i * data_loader.batch_size
            batch_end = batch_start + batch_size
            batch_ids = list(data_loader.dataset.patient_ids[batch_start:batch_end])
            
            # Forward pass to get pathway activations
            _, pathway_activations = model(x)
            
            # Store results
            all_pathway_activations.append(pathway_activations)
            all_patient_ids.extend(batch_ids)
    
    # Concatenate all pathway activations
    all_pathway_activations = torch.cat(all_pathway_activations, dim=0).cpu().numpy()
    
    # Create DataFrame with pathway importance scores
    importance_df = pd.DataFrame(
        all_pathway_activations,
        index=all_patient_ids
    )
    
    return importance_df

def identify_differential_pathways(importance_df, risk_groups, pathway_to_name, p_threshold=0.05):
    """
    Identify pathways with significantly different activation between risk groups
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with pathway importance scores
    risk_groups : pd.Series
        Series with risk group assignments (1 for high risk, 0 for low risk)
    pathway_to_name : dict
        Dictionary mapping pathway indices to pathway names
    p_threshold : float
        P-value threshold for significance
        
    Returns:
    --------
    diff_pathways : pd.DataFrame
        DataFrame with differentially activated pathways
    """
    # Initialize results
    results = []
    
    # Test each pathway for differential activation
    for col in importance_df.columns:
        # Get pathway values for each group
        high_risk_values = importance_df.loc[risk_groups[importance_df.index] == 1, col]
        low_risk_values = importance_df.loc[risk_groups[importance_df.index] == 0, col]
        
        # Perform t-test
        t_stat, p_val = ttest_ind(high_risk_values, low_risk_values, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        high_mean = high_risk_values.mean()
        low_mean = low_risk_values.mean()
        pooled_std = np.sqrt((high_risk_values.std()**2 + low_risk_values.std()**2) / 2)
        
        if pooled_std == 0:
            effect_size = 0
        else:
            effect_size = (high_mean - low_mean) / pooled_std
        
        # Store results
        results.append({
            'pathway_idx': col,
            'pathway_name': pathway_to_name.get(col, f"Pathway {col}"),
            'high_risk_mean': high_mean,
            'low_risk_mean': low_mean,
            'diff': high_mean - low_mean,
            't_stat': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'significant': p_val < p_threshold
        })
    
    # Create DataFrame
    diff_pathways = pd.DataFrame(results)
    
    # Sort by significance and effect size
    diff_pathways = diff_pathways.sort_values(['significant', 'p_value', 'effect_size'], 
                                           ascending=[False, True, False])
    
    return diff_pathways

def visualize_pathway_heatmap(importance_df, risk_groups, top_pathways, pathway_to_name):
    """
    Create heatmap visualization of top pathway activations
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with pathway importance scores
    risk_groups : pd.Series
        Series with risk group assignments (1 for high risk, 0 for low risk)
    top_pathways : list
        List of top pathway indices
    pathway_to_name : dict
        Dictionary mapping pathway indices to pathway names
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Heatmap figure
    """
    # Select top pathways
    heatmap_data = importance_df[top_pathways].copy()
    
    # Rename columns with pathway names
    heatmap_data.columns = [pathway_to_name.get(idx, f"Pathway {idx}") for idx in top_pathways]
    
    # Add risk group information
    heatmap_data['Risk_Group'] = risk_groups[heatmap_data.index].values
    
    # Sort by risk group and pathway values
    heatmap_data = heatmap_data.sort_values(['Risk_Group'] + list(heatmap_data.columns[:-1]))
    
    # Extract risk groups
    risk_groups_sorted = heatmap_data['Risk_Group']
    heatmap_data = heatmap_data.drop('Risk_Group', axis=1)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, cmap='viridis', yticklabels=False)
    
    # Add risk group color bar on the left
    risk_colors = ['blue' if x == 0 else 'red' for x in risk_groups_sorted]
    
    # Create a new axes for the risk group color bar
    risk_ax = plt.axes([0.05, 0.1, 0.02, 0.8])
    risk_ax.imshow(np.array(risk_groups_sorted).reshape(-1, 1), 
                  aspect='auto', cmap=plt.cm.get_cmap('coolwarm', 2))
    risk_ax.set_yticks([])
    risk_ax.set_xticks([0])
    risk_ax.set_xticklabels(['Risk'])
    
    # Set title and labels
    plt.suptitle('Pathway Activation Heatmap', fontsize=16)
    ax.set_title('Red: High Risk, Blue: Low Risk', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return plt.gcf()

# 12. Gene importance analysis

def analyze_gene_importance(model, pathway_connections, gene_names):
    """
    Analyze gene importance based on connection weights to pathways
    
    Parameters:
    -----------
    model : DeepOmix
        Trained DeepOmix model
    pathway_connections : dict
        Dictionary mapping pathway indices to gene indices
    gene_names : list
        List of gene names
        
    Returns:
    --------
    gene_importance_df : pd.DataFrame
        DataFrame with gene importance scores
    """
    # Get gene-to-pathway weights
    weights = model.get_pathway_weights().cpu().numpy()
    
    # Calculate importance for each gene
    gene_importance = np.zeros(len(gene_names))
    
    # Sum of absolute weights for each gene across all pathways
    for i in range(weights.shape[1]):  # iterate over genes
        gene_importance[i] = np.sum(np.abs(weights[:, i]))
    
    # Create DataFrame
    gene_importance_df = pd.DataFrame({
        'gene': gene_names,
        'importance': gene_importance
    })
    
    # Sort by importance
    gene_importance_df = gene_importance_df.sort_values('importance', ascending=False)
    
    return gene_importance_df

def visualize_gene_pathway_network(model, pathway_connections, gene_names, pathway_to_name,
                                 top_genes=10, top_pathways=5):
    """
    Visualize gene-pathway network for top genes and pathways
    
    Parameters:
    -----------
    model : DeepOmix
        Trained DeepOmix model
    pathway_connections : dict
        Dictionary mapping pathway indices to gene indices
    gene_names : list
        List of gene names
    pathway_to_name : dict
        Dictionary mapping pathway indices to pathway names
    top_genes : int
        Number of top genes to include
    top_pathways : int
        Number of top pathways to include
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Network visualization figure
    """
    try:
        import networkx as nx
        
        # Get gene importance
        gene_importance_df = analyze_gene_importance(model, pathway_connections, gene_names)
        
        # Get top genes
        top_gene_names = gene_importance_df['gene'].iloc[:top_genes].values
        top_gene_indices = [list(gene_names).index(gene) for gene in top_gene_names]
        
        # Get pathway activation importance
        pathway_weights = model.gene_to_pathway.weight.detach().cpu().numpy()
        pathway_importance = np.sum(np.abs(pathway_weights), axis=1)
        
        # Get top pathways
        top_pathway_indices = np.argsort(-pathway_importance)[:top_pathways]
        top_pathway_names = [pathway_to_name.get(idx, f"Pathway {idx}") for idx in top_pathway_indices]
        
        # Create graph
        G = nx.Graph()
        
        # Add gene nodes
        for gene in top_gene_names:
            G.add_node(gene, type='gene')
        
        # Add pathway nodes
        for pathway in top_pathway_names:
            G.add_node(pathway, type='pathway')
        
        # Add edges
        for p_idx, p_name in zip(top_pathway_indices, top_pathway_names):
            for g_idx, g_name in zip(top_gene_indices, top_gene_names):
                weight = pathway_weights[p_idx, g_idx]
                if abs(weight) > 0.1:  # Only add edges with significant weights
                    G.add_edge(g_name, p_name, weight=abs(weight))
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Set positions
        pos = nx.spring_layout(G, seed=42)
        
        # Get node types
        gene_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'gene']
        pathway_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'pathway']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color='lightblue', 
                              node_size=500, alpha=0.8, label='Genes')
        nx.draw_networkx_nodes(G, pos, nodelist=pathway_nodes, node_color='lightgreen', 
                              node_size=700, alpha=0.8, label='Pathways')
        
        # Get edge weights for width
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title('Gene-Pathway Network')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    except ImportError:
        print("NetworkX library is required for network visualization.")
        return None

# Example of complete pipeline with synthetic data

def run_demo_with_synthetic_data():
    """
    Run the complete DeepOmix pipeline with synthetic data
    """
    from scipy.stats import ttest_ind
    import matplotlib.pyplot as plt
    
    print("Generating synthetic data...")
    rna_seq_data, survival_data, pathway_data = generate_synthetic_data(
        n_samples=200, n_genes=1000, n_pathways=50
    )
    
    # Map genes to indices
    gene_to_index = {gene: i for i, gene in enumerate(rna_seq_data.columns)}
    
    # Create pathway connections
    pathway_connections, pathway_to_name = {}, {}
    for idx, (pathway, genes) in enumerate(pathway_data.items()):
        gene_indices = [gene_to_index[gene] for gene in genes if gene in gene_to_index]
        if gene_indices:
            pathway_connections[idx] = gene_indices
            pathway_to_name[idx] = pathway
    
    # Split data
    patient_ids = rna_seq_data.index.tolist()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)
    
    # Create custom dataset class with patient IDs
    class RNASeqSurvivalDatasetWithIDs(Dataset):
        def __init__(self, rna_seq_data, survival_data):
            self.patient_ids = rna_seq_data.index.tolist()
            self.rna_seq = torch.tensor(rna_seq_data.values, dtype=torch.float32)
            self.survival_time = torch.tensor(
                survival_data['time'].values, dtype=torch.float32
            )
            self.survival_event = torch.tensor(
                survival_data['event'].values, dtype=torch.float32
            )
            
        def __len__(self):
            return len(self.rna_seq)
        
        def __getitem__(self, idx):
            return {
                'rna_seq': self.rna_seq[idx],
                'time': self.survival_time[idx],
                'event': self.survival_event[idx]
            }
    
    # Create datasets
    train_dataset = RNASeqSurvivalDatasetWithIDs(
        rna_seq_data.loc[train_ids], survival_data.loc[train_ids]
    )
    val_dataset = RNASeqSurvivalDatasetWithIDs(
        rna_seq_data.loc[val_ids], survival_data.loc[val_ids]
    )
    test_dataset = RNASeqSurvivalDatasetWithIDs(
        rna_seq_data.loc[test_ids], survival_data.loc[test_ids]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("Initializing DeepOmix model...")
    model = DeepOmix(
        input_dim=rna_seq_data.shape[1],
        pathway_connections=pathway_connections,
        hidden_dims=[64, 32]
    )
    
    # Train model
    print("Training model...")
    model, history = train_deepomix(
        model, train_loader, val_loader,
        learning_rate=0.001,
        num_epochs=50,
        patience=5
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_deepomix(model, test_loader, pathway_to_name)
    
    # Create risk groups DataFrame
    risk_groups = pd.Series(
        results['risk_groups'],
        index=test_ids
    )
    
    # Survival analysis
    print("Performing survival analysis...")
    p_value = perform_survival_analysis(
        results['times'], results['events'], results['risk_groups']
    )
    print(f"Log-rank test p-value: {p_value:.4e}")
    
    # Pathway importance analysis
    test_data = torch.tensor(rna_seq_data.loc[test_ids].values, dtype=torch.float32)
    importance_df = compute_pathway_importance(model, test_loader)
    importance_df.columns = range(len(importance_df.columns))
    
    # Find differentially activated pathways
    diff_pathways = identify_differential_pathways(
        importance_df, risk_groups, pathway_to_name
    )
    
    print("\nTop differentially activated pathways:")
    print(diff_pathways.head(10)[['pathway_name', 'diff', 'p_value', 'significant']])
    
    # Visualize top pathways
    top_pathway_indices = diff_pathways['pathway_idx'].iloc[:10].values
    visualize_pathway_heatmap(importance_df, risk_groups, top_pathway_indices, pathway_to_name)
    
    # Gene importance analysis
    gene_importance = analyze_gene_importance(model, pathway_connections, rna_seq_data.columns)
    
    print("\nTop important genes:")
    print(gene_importance.head(10))
    
    # Network visualization if networkx is available
    try:
        visualize_gene_pathway_network(
            model, pathway_connections, rna_seq_data.columns.tolist(), 
            pathway_to_name, top_genes=10, top_pathways=5
        )
    except Exception as e:
        print(f"Network visualization error: {e}")
    
    # Display all plots
    plt.show()
    
    return model, results, history

# Run the demo if this script is executed directly
if __name__ == "__main__":
    model, results, history = run_demo_with_synthetic_data()


