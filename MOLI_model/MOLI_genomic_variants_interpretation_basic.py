import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data (same as before)
n_samples = 1000

# Variant annotations (categorical)
variant_data = pd.DataFrame({
    'gene': np.random.choice(['BRCA1', 'TP53', 'PTEN'], n_samples),
    'variant_type': np.random.choice(['missense', 'frameshift', 'splice'], n_samples)
})

# Pre-computed scores (continuous)
score_data = pd.DataFrame({
    'cadd_score': np.random.uniform(0, 30, n_samples),
    'polyphen_score': np.random.uniform(0, 1, n_samples)
})

# Functional genomics (binary)
functional_data = pd.DataFrame({
    'chromatin_access': np.random.randint(0, 2, n_samples),
    'tf_binding': np.random.randint(0, 2, n_samples)
})

# Labels (pathogenic: 1, benign: 0)
labels = np.random.randint(0, 2, n_samples)

# Preprocess data
variant_encoded = pd.get_dummies(variant_data, columns=['gene', 'variant_type'])
scaler = StandardScaler()
score_normalized = scaler.fit_transform(score_data)

# Convert to tensors
variant_tensor = torch.tensor(variant_encoded.values, dtype=torch.float32)
score_tensor = torch.tensor(score_normalized, dtype=torch.float32)
functional_tensor = torch.tensor(functional_data.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Split into train/test sets (80/20)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
    variant_tensor, score_tensor, functional_tensor, labels_tensor,
    test_size=0.2, random_state=42
)

class MOLI(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, output_dim=1):
        super().__init__()
        # Branch 1: Variant annotations
        self.branch1 = nn.Sequential(
            nn.Linear(input_dims[0], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Branch 2: Pre-computed scores
        self.branch2 = nn.Sequential(
            nn.Linear(input_dims[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Branch 3: Functional genomics
        self.branch3 = nn.Sequential(
            nn.Linear(input_dims[2], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)
        combined = torch.cat([x1, x2, x3], dim=1)
        return self.classifier(combined)

# Initialize model
input_dims = [X1_train.shape[1], X2_train.shape[1], X3_train.shape[1]]
model = MOLI(input_dims)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Track training loss
train_losses = []
epochs = 50

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X1_train, X2_train, X3_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Predict on test set
model.eval()
with torch.no_grad():
    y_pred_prob = model(X1_test, X2_test, X3_test).numpy().flatten()

# Convert probabilities to binary predictions (threshold=0.5)
y_pred = (y_pred_prob >= 0.5).astype(int)
y_true = y_test.numpy().flatten()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_prob)

print(f"""
Evaluation Metrics:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}
- ROC-AUC: {roc_auc:.4f}
""")


plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Benign', 'Pathogenic'],
            yticklabels=['Benign', 'Pathogenic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()




