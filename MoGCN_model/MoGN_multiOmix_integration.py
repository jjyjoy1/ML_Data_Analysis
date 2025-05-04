import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# Simulated multi-omics data
# In reality, these would be loaded from files
n_patients = 100
n_genes = 1000
n_proteins = 500
n_mutations = 200

# Create sample multi-omics data
gene_expression = np.random.randn(n_patients, n_genes)  # RNA-seq data
protein_abundance = np.random.randn(n_patients, n_proteins)  # Proteomics data
mutation_data = np.random.binomial(1, 0.05, (n_patients, n_mutations))  # Genomic data

# Patient metadata and outcome
patient_metadata = pd.DataFrame({
    'patient_id': [f'P{i}' for i in range(n_patients)],
    'age': np.random.randint(30, 80, n_patients),
    'gender': np.random.choice(['M', 'F'], n_patients),
    'outcome': np.random.choice([0, 1], n_patients)  # 0: healthy, 1: disease
})

# Normalize the data
scaler = StandardScaler()
gene_expression_norm = scaler.fit_transform(gene_expression)
protein_abundance_norm = scaler.fit_transform(protein_abundance)
# Mutation data is already binary, no need to normalize

class MultiOmicsAutoencoder(nn.Module):
    def __init__(self, gene_dim, protein_dim, mutation_dim, latent_dim):
        super(MultiOmicsAutoencoder, self).__init__()
        
        # Encoders for each omics type
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.mutation_encoder = nn.Sequential(
            nn.Linear(mutation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = 256 + 128 + 64  # Combined dimensions from individual encoders
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoders for each omics type
        self.gene_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, gene_dim)
        )
        
        self.protein_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, protein_dim)
        )
        
        self.mutation_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, mutation_dim),
            nn.Sigmoid()  # Mutation data is binary
        )
    
    def encode(self, gene_data, protein_data, mutation_data):
        # Encode each omics type
        gene_features = self.gene_encoder(gene_data)
        protein_features = self.protein_encoder(protein_data)
        mutation_features = self.mutation_encoder(mutation_data)
        
        # Concatenate features from all omics types
        combined_features = torch.cat([gene_features, protein_features, mutation_features], dim=1)
        
        # Fuse features
        latent_representation = self.fusion_layer(combined_features)
        
        return latent_representation
    
    def decode(self, latent_representation):
        # Decode to reconstruct original omics data
        gene_recon = self.gene_decoder(latent_representation)
        protein_recon = self.protein_decoder(latent_representation)
        mutation_recon = self.mutation_decoder(latent_representation)
        
        return gene_recon, protein_recon, mutation_recon
    
    def forward(self, gene_data, protein_data, mutation_data):
        # Encode
        latent = self.encode(gene_data, protein_data, mutation_data)
        
        # Decode
        gene_recon, protein_recon, mutation_recon = self.decode(latent)
        
        return latent, gene_recon, protein_recon, mutation_recon

def create_patient_similarity_graph(patient_features, threshold=0.7):
    """
    Construct a patient similarity graph based on feature correlation.
    Returns edge_index for PyTorch Geometric.
    """
    # Calculate pairwise cosine similarity
    patient_features_normalized = F.normalize(patient_features, p=2, dim=1)
    similarity_matrix = torch.mm(patient_features_normalized, patient_features_normalized.t())
    
    # Apply threshold to create adjacency matrix
    adjacency_matrix = (similarity_matrix > threshold).float()
    
    # Remove self-loops
    adjacency_matrix.fill_diagonal_(0)
    
    # Convert adjacency matrix to edge_index format for PyTorch Geometric
    edge_index = adjacency_matrix.nonzero().t().contiguous()
    
    return edge_index

class MoGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MoGCN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

# Convert the data to PyTorch tensors
gene_tensor = torch.FloatTensor(gene_expression_norm)
protein_tensor = torch.FloatTensor(protein_abundance_norm)
mutation_tensor = torch.FloatTensor(mutation_data)
labels = torch.LongTensor(patient_metadata['outcome'].values)

# Initialize the autoencoder
latent_dim = 64
autoencoder = MultiOmicsAutoencoder(
    gene_dim=n_genes,
    protein_dim=n_proteins,
    mutation_dim=n_mutations,
    latent_dim=latent_dim
)

# Train the autoencoder
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
criterion_continuous = nn.MSELoss()
criterion_binary = nn.BCELoss()

num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    latent, gene_recon, protein_recon, mutation_recon = autoencoder(gene_tensor, protein_tensor, mutation_tensor)
    
    # Calculate reconstruction loss for each omics type
    gene_loss = criterion_continuous(gene_recon, gene_tensor)
    protein_loss = criterion_continuous(protein_recon, protein_tensor)
    mutation_loss = criterion_binary(mutation_recon, mutation_tensor)
    
    # Total loss
    total_loss = gene_loss + protein_loss + mutation_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Get the integrated features for all patients
with torch.no_grad():
    integrated_features = autoencoder.encode(gene_tensor, protein_tensor, mutation_tensor)

# Create patient similarity graph
edge_index = create_patient_similarity_graph(integrated_features)

# Create PyTorch Geometric Data object
data = Data(x=integrated_features, edge_index=edge_index, y=labels)

# Initialize the GCN model
gcn_model = MoGCN(input_dim=latent_dim, hidden_dim=32, output_dim=2)  # 2 classes: healthy/disease

# Train the GCN model
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

num_gcn_epochs = 100
for epoch in range(num_gcn_epochs):
    # Training mode
    gcn_model.train()
    
    # Forward pass
    gcn_optimizer.zero_grad()
    out = gcn_model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    
    # Backward pass and optimization
    loss.backward()
    gcn_optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        # Evaluation mode
        gcn_model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = (pred == data.y).sum().item()
            acc = correct / data.y.size(0)
            print(f'Epoch [{epoch+1}/{num_gcn_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')


import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def visualize_patient_graph(integrated_features, edge_index, labels):
    """Visualize the patient similarity graph using t-SNE."""
    # Convert to numpy for t-SNE
    features_np = integrated_features.detach().numpy()
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    node_positions = tsne.fit_transform(features_np)
    
    # Create a NetworkX graph
    G = nx.Graph()
    for i in range(features_np.shape[0]):
        G.add_node(i, pos=node_positions[i], label=labels[i].item())
    
    # Add edges
    edges = edge_index.t().numpy()
    for i in range(edges.shape[0]):
        G.add_edge(edges[i, 0], edges[i, 1])
    
    # Get positions and labels
    pos = nx.get_node_attributes(G, 'pos')
    node_labels = nx.get_node_attributes(G, 'label')
    
    # Plot
    plt.figure(figsize=(10, 8))
    node_colors = [node_labels[i] for i in G.nodes()]
    nx.draw(G, pos, with_labels=False, node_color=node_colors, 
            cmap=plt.cm.coolwarm, node_size=100, alpha=0.8)
    plt.title("Patient Similarity Graph")
    plt.savefig("patient_similarity_graph.png")
    plt.show()

def analyze_feature_importance(gcn_model, integrated_features):
    """Analyze which features are most important for the GCN model."""
    # Get weights from the first GCN layer
    weights = gcn_model.conv1.lin.weight.detach().numpy()
    
    # Calculate feature importance based on weight magnitude
    feature_importance = np.abs(weights).mean(axis=0)
    
    # Sort features by importance
    sorted_idx = np.argsort(-feature_importance)
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), feature_importance[sorted_idx[:20]])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('Top 20 Important Features in the Integrated Representation')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

# Call visualization and analysis functions
visualize_patient_graph(integrated_features, edge_index, labels)
analyze_feature_importance(gcn_model, integrated_features)




