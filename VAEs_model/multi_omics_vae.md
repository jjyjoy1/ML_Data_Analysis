# Multi-omics Integration with Variational Autoencoders

A PyTorch implementation of a Variational Autoencoder (VAE) framework for integrating RNA-seq and metagenomic data with clinical metadata. This repository provides tools for joint dimensionality reduction, feature selection, and clinical outcome prediction.

## Features

- **Multi-omics Data Integration**: Seamlessly integrate RNA-seq and metagenomic data in a unified latent space
- **Clinical Metadata Incorporation**: Include clinical variables like cancer stage, age, and drug response
- **Robust Feature Selection**: Identify important features across multiple samples
- **Clinical Outcome Prediction**: Predict cancer stage and drug response from integrated data
- **Sample-Specific Analysis**: Perform stratified analysis by clinical groups
- **Comprehensive Visualization**: Explore latent space and feature importance patterns


```
## Quick Start

```python
import scanpy as sc
from multi_omics_vae import run_multi_omics_integration

# Load your data as AnnData objects
rnaseq_adata = sc.read_h5ad("path/to/rnaseq_data.h5ad")
metag_adata = sc.read_h5ad("path/to/metagenomic_data.h5ad")

# Run integration with clinical variables
results = run_multi_omics_integration(
    rnaseq_adata, 
    metag_adata,
    clinical_vars=['cancer_stage', 'age', 'drug_response'],
    hidden_dim=128,
    latent_dim=32,
    num_epochs=100,
    feature_threshold=0.5  # Features must be important in 50% of samples
)

# Unpack results
model, latent_features, rnaseq_importance, metag_importance, \
rnaseq_selected, metag_selected, selected_feature_names, \
clinical_info, metrics = results

# Access selected features
print(f"Selected {len(selected_feature_names['rnaseq'])} RNA-seq features")
print(f"Selected {len(selected_feature_names['metag'])} metagenomic features")

# Access latent space coordinates
latent_df = pd.DataFrame(latent_features, index=rnaseq_adata.obs_names,
                        columns=[f'latent_{i}' for i in range(latent_features.shape[1])])
```

## Model Architecture

The VAE architecture consists of:

1. **Dual Encoders**: Separate encoding pathways for RNA-seq and metagenomic data
2. **Clinical Data Integration**: Optional pathways for clinical variables
3. **Joint Latent Space**: Combined representation of all data modalities
4. **Predictive Heads**: For cancer stage and drug response prediction
5. **Dual Decoders**: Reconstruction pathways for each data type

<details>
<summary>Click to see architecture diagram</summary>

```
RNA-seq Data → RNA-seq Encoder ────┐
                                   │
Metagenomic Data → Metag Encoder ──┼─→ Joint Latent Space → [mu, logvar] → Sampling → z
                                   │         ↓        ↓           ↓
Clinical Data → Clinical Encoder ──┘   Stage Pred  Drug Pred   Decoder
                                                              /      \
                                                RNA-seq Recon  Metag Recon
```
</details>

## Feature Selection

The model uses a hybrid approach for feature selection:

1. **Global Importance**: Based on decoder weights connecting latent space to features
2. **Sample-Specific Analysis**: Features must be important across a threshold percentage of samples
3. **Clinical Stratification**: Separate feature selection for each cancer stage and drug response group

## Evaluation Metrics

The framework evaluates performance using:

| Metric Type | Specific Metrics |
|-------------|------------------|
| Reconstruction | MSE, Feature Correlation |
| Classification | Accuracy, F1-Score, AUC-ROC, Confusion Matrix |
| Latent Space | Silhouette Score, Cluster Separation |
| Feature Selection | Stability, Enrichment Analysis |

## Examples

### Visualizing the Latent Space

```python
import umap
import matplotlib.pyplot as plt

# Generate UMAP embedding
reducer = umap.UMAP(random_state=42)
umap_embedding = reducer.fit_transform(latent_features)

# Plot by cancer stage
plt.figure(figsize=(10, 8))
stages = rnaseq_adata.obs['cancer_stage']
for stage in sorted(stages.unique()):
    mask = stages == stage
    plt.scatter(
        umap_embedding[mask, 0], 
        umap_embedding[mask, 1],
        label=f'Stage {stage}', 
        alpha=0.7, 
        s=80
    )
plt.legend()
plt.title('UMAP of Latent Space by Cancer Stage')
plt.show()
```

### Analyzing Feature Importance

```python
import seaborn as sns

# Plot top RNA-seq features
top_n = 20
top_idx = np.argsort(rnaseq_importance)[::-1][:top_n]
top_genes = [rnaseq_adata.var_names[i] for i in top_idx]
top_importance = rnaseq_importance[top_idx]

plt.figure(figsize=(12, 8))
sns.barplot(x=top_importance, y=top_genes)
plt.title(f'Top {top_n} RNA-seq Features')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

## Advanced Usage

### Cross-Validation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(rnaseq_adata.obs_names)):
    print(f"Running fold {fold+1}/5...")
    
    # Create train/test split
    train_rna = rnaseq_adata[train_idx].copy()
    train_meta = metag_adata[train_idx].copy()
    test_rna = rnaseq_adata[test_idx].copy()
    test_meta = metag_adata[test_idx].copy()
    
    # Run integration
    results = run_multi_omics_integration(
        train_rna, train_meta, 
        clinical_vars=['cancer_stage', 'drug_response'],
        num_epochs=50  # Reduced for CV
    )
    
    # Store results
    cv_results.append(results[-1])  # Store metrics

# Analyze CV results
stage_acc = [r['test_metrics']['stage_acc'][-1] for r in cv_results]
print(f"Mean stage prediction accuracy: {np.mean(stage_acc):.4f} ± {np.std(stage_acc):.4f}")
```

### Hyperparameter Tuning

The framework supports tuning of various hyperparameters:

- `hidden_dim`: Size of hidden layers
- `latent_dim`: Dimension of latent space
- `beta`: Weight of KL divergence term
- `lambda_stage`: Weight for cancer stage prediction loss
- `lambda_drug`: Weight for drug response prediction loss
- `feature_threshold`: Proportion of samples a feature must be important in


## Acknowledgments

- The implementation is based on the VAE architecture originally proposed by Kingma & Welling (2013)
- Clinical integration approach inspired by the work of Lopez et al. (2020) on scVI for single-cell data


