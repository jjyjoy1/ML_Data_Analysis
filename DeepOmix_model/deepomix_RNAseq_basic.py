'''
Case Study: Cancer Subtype Prediction

Objective: Predict tumor subtypes (e.g., "Aggressive" vs. "Non-Aggressive") using gene expression data, while interpreting pathway-level contributions.
'''

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Simulate data
n_samples = 500  # 500 patients
n_genes = 1000
n_pathways = 50

# Gene expression data (random normalized values)
X = np.random.randn(n_samples, n_genes)
y = np.random.randint(0, 2, size=n_samples)  # Binary labels

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prior knowledge: gene-pathway membership matrix (binary, shape: [genes, pathways])
# Example: Each pathway has 20 genes, randomly assigned
pathway_mask = np.zeros((n_genes, n_pathways), dtype=np.float32)
for pathway in range(n_pathways):
    genes_in_pathway = np.random.choice(n_genes, size=20, replace=False)
    pathway_mask[genes_in_pathway, pathway] = 1.0  # Mask connections

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
pathway_mask_tensor = torch.tensor(pathway_mask, dtype=torch.float32)


class DeepOmix(nn.Module):
    def __init__(self, n_genes, n_pathways, pathway_mask):
        super(DeepOmix, self).__init__()
        
        # Gene-to-Pathway Layer (enforced by prior knowledge)
        self.gene_to_pathway = nn.Linear(n_genes, n_pathways, bias=False)
        
        # Apply pathway mask: only allow connections defined by prior knowledge
        with torch.no_grad():
            self.gene_to_pathway.weight.data = self.gene_to_pathway.weight * pathway_mask.T
        
        # Pathway-to-Module Layer (learned hierarchy)
        self.module_layer = nn.Sequential(
            nn.Linear(n_pathways, 20),  # 20 higher-order modules
            nn.ReLU(),
            nn.Dropout(0.5)
        
        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(20, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.gene_to_pathway(x)  # Gene → Pathway (constrained)
        x = torch.relu(x)            # Non-linearity
        x = self.module_layer(x)     # Pathway → Module
        return self.output_layer(x)

# Initialize model
model = DeepOmix(n_genes=n_genes, n_pathways=n_pathways, pathway_mask=pathway_mask_tensor)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print loss
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Get pathway importance (absolute weights)
pathway_weights = model.gene_to_pathway.weight.data.abs().mean(dim=0).numpy()

# Rank pathways by importance
top_pathways = np.argsort(-pathway_weights)[:5]  # Top 5 pathways
print("Top 5 influential pathways:")
for idx in top_pathways:
    print(f"Pathway {idx}: Weight = {pathway_weights[idx]:.4f}")




