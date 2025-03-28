import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# 1. DATA PREPARATION
# ----------------------

# This is a simulated example - in real applications, you would load your multi-omics datasets
# Let's simulate gene expression and methylation data for 100 patients with 3 cancer subtypes

# Simulate gene expression data (1000 genes)
np.random.seed(42)
gene_expr = np.random.normal(0, 1, size=(100, 1000))
gene_names = [f"gene_{i}" for i in range(1000)]

# Simulate methylation data (800 methylation sites)
methyl_data = np.random.normal(0, 1, size=(100, 800))
methyl_sites = [f"methyl_{i}" for i in range(800)]

# Simulate cancer subtypes (0, 1, 2)
cancer_subtypes = np.random.randint(0, 3, size=100)

# ----------------------
# 2. PATHWAY KNOWLEDGE
# ----------------------

# In a real application, you would import pathway information from databases like KEGG, Reactome, etc.
# Here we'll create a simplified simulation of pathway knowledge

# Create 50 pathways with gene membership
num_pathways = 50
pathway_gene_membership = {}

for i in range(num_pathways):
    # Each pathway contains a random subset of genes (between 10-50 genes)
    num_genes_in_pathway = np.random.randint(10, 50)
    gene_indices = np.random.choice(1000, size=num_genes_in_pathway, replace=False)
    pathway_gene_membership[f"pathway_{i}"] = gene_indices

# Create a gene-to-pathway matrix (genes × pathways)
gene_pathway_matrix = np.zeros((1000, num_pathways))
for pathway_idx, (pathway_name, gene_indices) in enumerate(pathway_gene_membership.items()):
    for gene_idx in gene_indices:
        gene_pathway_matrix[gene_idx, pathway_idx] = 1

# Also create 30 higher-level biological processes that group pathways
num_processes = 30
pathway_process_membership = {}

for i in range(num_processes):
    # Each process contains 2-5 pathways
    num_pathways_in_process = np.random.randint(2, 6)
    pathway_indices = np.random.choice(num_pathways, size=num_pathways_in_process, replace=False)
    pathway_process_membership[f"process_{i}"] = pathway_indices

# Create a pathway-to-process matrix (pathways × processes)
pathway_process_matrix = np.zeros((num_pathways, num_processes))
for process_idx, (process_name, pathway_indices) in enumerate(pathway_process_membership.items()):
    for pathway_idx in pathway_indices:
        pathway_process_matrix[pathway_idx, process_idx] = 1

# Also create a similar biological knowledge mapping for methylation sites
methyl_pathway_matrix = np.zeros((800, num_pathways))
for pathway_idx in range(num_pathways):
    # Assign some methylation sites to each pathway
    num_sites = np.random.randint(5, 30)
    site_indices = np.random.choice(800, size=num_sites, replace=False)
    for site_idx in site_indices:
        methyl_pathway_matrix[site_idx, pathway_idx] = 1

# ----------------------
# 3. CUSTOM PYTORCH DATASET
# ----------------------

class MultiOmicsDataset(Dataset):
    def __init__(self, gene_data, methyl_data, labels):
        self.gene_data = torch.FloatTensor(gene_data)
        self.methyl_data = torch.FloatTensor(methyl_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'gene_expr': self.gene_data[idx],
            'methyl_data': self.methyl_data[idx],
            'label': self.labels[idx]
        }

# Split data into train and test sets
X_gene_train, X_gene_test, X_methyl_train, X_methyl_test, y_train, y_test = train_test_split(
    gene_expr, methyl_data, cancer_subtypes, test_size=0.2, random_state=42
)

# Standardize the data
gene_scaler = StandardScaler()
methyl_scaler = StandardScaler()

X_gene_train = gene_scaler.fit_transform(X_gene_train)
X_gene_test = gene_scaler.transform(X_gene_test)
X_methyl_train = methyl_scaler.fit_transform(X_methyl_train)
X_methyl_test = methyl_scaler.transform(X_methyl_test)

# Create datasets
train_dataset = MultiOmicsDataset(X_gene_train, X_methyl_train, y_train)
test_dataset = MultiOmicsDataset(X_gene_test, X_methyl_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ----------------------
# 4. DEEPOMIX MODEL
# ----------------------

class AttentionLayer(nn.Module):
    """Attention mechanism to allow the model to focus on different pathways/processes."""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Calculate attention weights
        weights = self.attention(x)
        # Apply attention weights
        weighted_x = x * weights
        return weighted_x, weights

class DeepOmix(nn.Module):
    def __init__(self, gene_dim, methyl_dim, num_pathways, num_processes, num_classes,
                 gene_pathway_matrix, methyl_pathway_matrix, pathway_process_matrix):
        super(DeepOmix, self).__init__()
        
        # Convert biological knowledge matrices to PyTorch tensors
        self.gene_pathway_connections = nn.Parameter(
            torch.FloatTensor(gene_pathway_matrix), 
            requires_grad=False  # Fixed connections based on prior knowledge
        )
        
        self.methyl_pathway_connections = nn.Parameter(
            torch.FloatTensor(methyl_pathway_matrix),
            requires_grad=False
        )
        
        self.pathway_process_connections = nn.Parameter(
            torch.FloatTensor(pathway_process_matrix),
            requires_grad=False
        )
        
        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Methylation encoder
        self.methyl_encoder = nn.Sequential(
            nn.Linear(methyl_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Pathway layer with attention
        self.pathway_transform_gene = nn.Linear(256, num_pathways, bias=False)
        self.pathway_transform_methyl = nn.Linear(256, num_pathways, bias=False)
        self.pathway_attention = AttentionLayer(num_pathways)
        
        # Biological process layer
        self.process_layer = nn.Linear(num_pathways, num_processes, bias=False)
        self.process_attention = AttentionLayer(num_processes)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_processes, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # Store intermediate activations for interpretability
        self.gene_activations = None
        self.methyl_activations = None
        self.pathway_activations = None
        self.pathway_attention_weights = None
        self.process_activations = None
        self.process_attention_weights = None
        
    def forward(self, gene_expr, methyl_data):
        # 1. Encode individual omics data
        gene_encoded = self.gene_encoder(gene_expr)
        methyl_encoded = self.methyl_encoder(methyl_data)
        
        # Store intermediate gene/methyl activations
        self.gene_activations = gene_encoded
        self.methyl_activations = methyl_encoded
        
        # 2. Project to pathway space (applying biological constraints)
        pathway_from_genes = self.pathway_transform_gene(gene_encoded)
        pathway_from_methyl = self.pathway_transform_methyl(methyl_encoded)
        
        # 3. Combine pathway signals from different omics
        pathway_combined = pathway_from_genes + pathway_from_methyl
        
        # Apply pathway attention
        pathway_attended, self.pathway_attention_weights = self.pathway_attention(pathway_combined)
        
        # Store pathway activations for interpretation
        self.pathway_activations = pathway_attended
        
        # 4. Project to biological process space
        biological_processes = self.process_layer(pathway_attended)
        
        # Apply process attention
        process_attended, self.process_attention_weights = self.process_attention(biological_processes)
        
        # Store process activations
        self.process_activations = process_attended
        
        # 5. Final classification
        output = self.classifier(process_attended)
        
        return output
    
    def get_attention_weights(self):
        """Return attention weights for interpretation."""
        return {
            'pathway_attention': self.pathway_attention_weights,
            'process_attention': self.process_attention_weights
        }
    
    def get_activations(self):
        """Return intermediate activations for interpretation."""
        return {
            'gene_activations': self.gene_activations,
            'methyl_activations': self.methyl_activations,
            'pathway_activations': self.pathway_activations,
            'process_activations': self.process_activations
        }

# Initialize the model
model = DeepOmix(
    gene_dim=1000,
    methyl_dim=800,
    num_pathways=num_pathways,
    num_processes=num_processes,
    num_classes=3,
    gene_pathway_matrix=gene_pathway_matrix,
    methyl_pathway_matrix=methyl_pathway_matrix,
    pathway_process_matrix=pathway_process_matrix
)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ----------------------
# 5. TRAINING FUNCTION
# ----------------------

def train(model, train_loader, criterion, optimizer, epochs=30):
    model.train()
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            gene_expr = batch['gene_expr']
            methyl_data = batch['methyl_data']
            labels = batch['label']
            
            # Forward pass
            outputs = model(gene_expr, methyl_data)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return history

# ----------------------
# 6. EVALUATION FUNCTION
# ----------------------

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            gene_expr = batch['gene_expr']
            methyl_data = batch['methyl_data']
            labels = batch['label']
            
            outputs = model(gene_expr, methyl_data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    # Generate classification report
    report = classification_report(all_labels, all_preds)
    
    return accuracy, report, all_preds, all_labels

# ----------------------
# 7. INTERPRETATION FUNCTION
# ----------------------

def interpret_model(model, test_loader):
    """Extract and visualize biological insights from the trained model."""
    model.eval()
    
    # Get a batch for interpretation
    batch = next(iter(test_loader))
    gene_expr = batch['gene_expr']
    methyl_data = batch['methyl_data']
    labels = batch['label']
    
    # Forward pass to collect activations and attention weights
    with torch.no_grad():
        outputs = model(gene_expr, methyl_data)
    
    # Get attention weights
    attention_weights = model.get_attention_weights()
    pathway_attention = attention_weights['pathway_attention'].squeeze().cpu().numpy()
    process_attention = attention_weights['process_attention'].squeeze().cpu().numpy()
    
    # Calculate average pathway and process importance
    pathway_importance = pathway_attention.mean(axis=0)  # Average across batch
    process_importance = process_attention.mean(axis=0)  # Average across batch
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot pathway importance
    pathway_names = [f"Pathway {i}" for i in range(num_pathways)]
    top_pathways_idx = np.argsort(pathway_importance)[-10:]  # Top 10 pathways
    ax1.barh([pathway_names[i] for i in top_pathways_idx], 
             [pathway_importance[i] for i in top_pathways_idx])
    ax1.set_title('Top 10 Important Pathways')
    ax1.set_xlabel('Attention Weight')
    
    # Plot process importance
    process_names = [f"Process {i}" for i in range(num_processes)]
    top_processes_idx = np.argsort(process_importance)[-10:]  # Top 10 processes
    ax2.barh([process_names[i] for i in top_processes_idx], 
             [process_importance[i] for i in top_processes_idx])
    ax2.set_title('Top 10 Important Biological Processes')
    ax2.set_xlabel('Attention Weight')
    
    plt.tight_layout()
    return fig

# ----------------------
# 8. TRAINING AND EVALUATION
# ----------------------

# Train the model
history = train(model, train_loader, criterion, optimizer, epochs=30)

# Evaluate the model
accuracy, report, all_preds, all_labels = evaluate(model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Interpret the model
interpretation_plot = interpret_model(model, test_loader)
plt.show()

# ----------------------
# 9. CASE STUDY: INVESTIGATING IMPORTANT PATHWAYS FOR SPECIFIC SUBTYPE
# ----------------------

def analyze_subtype_specific_pathways(model, test_loader, target_class=0):
    """Analyze which pathways are most important for a specific cancer subtype."""
    model.eval()
    
    # Collect samples for target class
    target_samples = {'gene_expr': [], 'methyl_data': [], 'labels': []}
    
    for batch in test_loader:
        for i, label in enumerate(batch['label']):
            if label.item() == target_class:
                target_samples['gene_expr'].append(batch['gene_expr'][i])
                target_samples['methyl_data'].append(batch['methyl_data'][i])
                target_samples['labels'].append(label)
    
    # Convert to tensors
    target_gene_expr = torch.stack(target_samples['gene_expr'])
    target_methyl_data = torch.stack(target_samples['methyl_data'])
    
    # Forward pass to collect activations
    with torch.no_grad():
        outputs = model(target_gene_expr, target_methyl_data)
    
    # Get attention weights for this subtype
    attention_weights = model.get_attention_weights()
    pathway_attention = attention_weights['pathway_attention'].squeeze().cpu().numpy()
    
    # Calculate average pathway importance for this subtype
    pathway_importance = pathway_attention.mean(axis=0)
    
    # Get most important pathways for this subtype
    top_pathways_idx = np.argsort(pathway_importance)[-5:]  # Top 5 pathways
    top_pathways = {f"Pathway {i}": pathway_importance[i] for i in top_pathways_idx}
    
    # Create a report
    report = f"Top pathways for cancer subtype {target_class}:\n"
    for pathway, importance in sorted(top_pathways.items(), key=lambda x: x[1], reverse=True):
        report += f"- {pathway}: importance score {importance:.4f}\n"
        
        # Find genes in this pathway (in a real application, these would be actual gene names)
        pathway_idx = int(pathway.split(' ')[1])
        genes_in_pathway = np.where(gene_pathway_matrix[:, pathway_idx] > 0)[0]
        gene_list = [gene_names[g] for g in genes_in_pathway[:5]]  # Show first 5 genes
        report += f"  Contains genes: {', '.join(gene_list)}...\n"
        
        # Find biological processes this pathway belongs to
        processes = np.where(pathway_process_matrix[pathway_idx, :] > 0)[0]
        process_list = [f"Process {p}" for p in processes]
        report += f"  Part of processes: {', '.join(process_list)}\n"
    
    return report

# Analyze pathways specific to subtype 0
subtype_analysis = analyze_subtype_specific_pathways(model, test_loader, target_class=0)
print(subtype_analysis)

# ----------------------
# 10. REAL-WORLD APPLICATION NOTES
# ----------------------

'''
In a real-world application of DeepOmix, you would make the following enhancements:

1. DATA:
   - Use real multi-omics datasets (gene expression, methylation, protein expression, etc.)
   - Apply proper normalization techniques specific to each omics type
   - Handle missing values appropriately

2. BIOLOGICAL KNOWLEDGE:
   - Import pathway information from databases like KEGG, Reactome, or MSigDB
   - Use Gene Ontology (GO) terms for biological processes
   - Consider protein-protein interaction networks for additional structure

3. MODEL ARCHITECTURE:
   - Fine-tune the architecture based on the specific biological problem
   - Consider using graph neural networks for more complex biological relationships
   - Implement more sophisticated attention mechanisms (e.g., multi-head attention)

4. INTERPRETATION:
   - Map pathway and process IDs to actual biological pathway names
   - Conduct enrichment analysis on important features
   - Validate findings against literature or with wet lab experiments

5. VALIDATION:
   - Use cross-validation to ensure robustness
   - Compare with simpler models as baselines
   - Perform external validation on independent datasets

The key advantage of DeepOmix-style models is that they provide mechanistic
insights that are directly interpretable in terms of biological pathways and processes,
unlike standard deep learning approaches.
'''



