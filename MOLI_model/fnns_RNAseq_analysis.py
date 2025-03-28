import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load RNA-seq data
# In a real scenario, this would be a CSV or TSV file with gene expression counts
# Format: rows = samples, columns = genes (features)
def load_rnaseq_data(file_path):
    """
    Load RNA-seq expression data from a file
    Returns expression matrix and labels
    """
    data = pd.read_csv(file_path)
    
    # Extract labels (e.g., cancer vs. normal)
    y = data['class'].map({'normal': 0, 'tumor': 1})
    
    # Extract gene expression values (dropping non-expression columns)
    X = data.drop(['sample_id', 'class'], axis=1)
    
    return X, y

# For demonstration, let's create synthetic RNA-seq data
def create_synthetic_rnaseq_data(n_samples=200, n_genes=1000, random_state=42):
    """
    Create synthetic RNA-seq data for demonstration
    """
    np.random.seed(random_state)
    
    # Create gene expression matrix (log2 transformed counts)
    X = np.random.normal(0, 1, size=(n_samples, n_genes))
    
    # Create synthetic differential expression pattern
    # Select 50 genes that will be differentially expressed
    diff_genes = np.random.choice(n_genes, 50, replace=False)
    
    # Create labels
    y = np.zeros(n_samples)
    tumor_samples = np.random.choice(n_samples, n_samples // 2, replace=False)
    y[tumor_samples] = 1
    
    # Add differential expression pattern to tumor samples
    for i in tumor_samples:
        X[i, diff_genes] += np.random.normal(2, 0.5, size=50)  # Upregulated genes
    
    # Convert to DataFrame
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    sample_ids = [f'sample_{i}' for i in range(n_samples)]
    
    X_df = pd.DataFrame(X, columns=gene_names, index=sample_ids)
    X_df['sample_id'] = sample_ids
    X_df['class'] = ['tumor' if label == 1 else 'normal' for label in y]
    
    return X_df

# Preprocess RNA-seq data
def preprocess_data(X, y):
    """
    Preprocess RNA-seq data:
    1. Split into train/test sets
    2. Scale the data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the data (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Build a feed-forward neural network
def build_fnn_model(input_dim, dropout_rate=0.3):
    """
    Build a feed-forward neural network for gene expression classification
    """
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        
        # Hidden layers
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        
        # Output layer - binary classification (tumor vs normal)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train and evaluate the model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the FNN model and evaluate its performance
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, history

# Feature importance analysis
def analyze_feature_importance(model, gene_names):
    """
    Analyze which genes are most important for classification
    by examining the weights of the first layer
    """
    # Get weights from the first layer
    weights = model.layers[0].get_weights()[0]
    
    # Calculate absolute importance
    importance = np.mean(np.abs(weights), axis=1)
    
    # Create DataFrame with gene names and importance
    importance_df = pd.DataFrame({
        'Gene': gene_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top 20 genes
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Gene', data=importance_df.head(20))
    plt.title('Top 20 Most Important Genes')
    plt.tight_layout()
    plt.show()
    
    return importance_df

# Main workflow
def main():
    # Generate or load data
    print("Creating synthetic RNA-seq data...")
    rnaseq_df = create_synthetic_rnaseq_data(n_samples=200, n_genes=1000)
    
    # Extract features and labels
    X = rnaseq_df.drop(['sample_id', 'class'], axis=1)
    y = rnaseq_df['class'].map({'normal': 0, 'tumor': 1})
    gene_names = X.columns
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} genes")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Build and train FNN model
    print("\nBuilding and training feed-forward neural network...")
    model = build_fnn_model(input_dim=X.shape[1])
    model, history = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    
    # Analyze feature importance
    print("\nAnalyzing gene importance...")
    importance_df = analyze_feature_importance(model, gene_names)
    print("Top 10 important genes:")
    print(importance_df.head(10))

if __name__ == "__main__":
    main()


