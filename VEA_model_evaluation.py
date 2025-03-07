# For latent space quality
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(latent_features, rnaseq_adata.obs['cancer_stage'])

# For stage prediction
from sklearn.metrics import classification_report, confusion_matrix
stage_preds = model.predict_stage(torch.tensor(latent_features).to(device)).argmax(dim=1).cpu().numpy()
print(classification_report(true_stages, stage_preds))
print(confusion_matrix(true_stages, stage_preds))

# For reconstruction quality
rnaseq_recon, metag_recon = model.decode(torch.tensor(latent_features).to(device))
rnaseq_mse = F.mse_loss(rnaseq_recon.cpu(), torch.tensor(full_rnaseq_data)).item()

#Classification Report
from sklearn.metrics import classification_report

# Assuming stage_preds contains predicted stages (0-4 for stages I-V)
# And true_stages contains the actual stages
report = classification_report(true_stages, stage_preds)
print(report)

###output like
'''
precision    recall  f1-score   support

     Stage I       0.85      0.92      0.88        25
    Stage II       0.79      0.84      0.81        32
   Stage III       0.76      0.70      0.73        27
    Stage IV       0.68      0.65      0.66        15
     Stage V       0.71      0.55      0.62        11

    accuracy                           0.78       110
   macro avg       0.76      0.73      0.74       110
weighted avg       0.77      0.78      0.77       110
'''

#Confusion Matrix, shows where your model makes mistakes
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(true_stages, stage_preds)

# Visualize it
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['I', 'II', 'III', 'IV', 'V'],
            yticklabels=['I', 'II', 'III', 'IV', 'V'])
plt.xlabel('Predicted Stage')
plt.ylabel('True Stage')
plt.title('Cancer Stage Prediction Confusion Matrix')
plt.show()


#ROC Curve and AUC for Drug Response
#For binary predictions like drug response, ROC curves and AUC are standard metrics:

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get drug response predictions (continuous values between 0-1)
drug_preds = model.predict_drug_response(torch.tensor(latent_features).to(device)).cpu().numpy()

# True drug responses (0 for negative, 1 for positive)
true_response = np.array([1 if r == 'positive' else 0 for r in rnaseq_adata.obs['drug_response']])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(true_response, drug_preds)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Drug Response Prediction')
plt.legend(loc="lower right")
plt.show()


#Reconstruction Quality
#To evaluate the unsupervised component of VAE model, measure reconstruction quality
# Get original and reconstructed data
rnaseq_recon, metag_recon = model.decode(torch.tensor(latent_features).to(device))
rnaseq_recon = rnaseq_recon.cpu().numpy()
metag_recon = metag_recon.cpu().numpy()

# Calculate MSE
rnaseq_mse = np.mean((full_rnaseq_data - rnaseq_recon) ** 2)
metag_mse = np.mean((full_metag_data - metag_recon) ** 2)

print(f"RNAseq reconstruction MSE: {rnaseq_mse:.4f}")
print(f"Metagenomic reconstruction MSE: {metag_mse:.4f}")

# Calculate correlation of features
from scipy.stats import pearsonr

# For each gene, calculate correlation between original and reconstructed
gene_cors = []
for i in range(rnaseq_recon.shape[1]):
    cor, _ = pearsonr(full_rnaseq_data[:, i], rnaseq_recon[:, i])
    gene_cors.append(cor)

# Average correlation across all genes
avg_gene_cor = np.mean(gene_cors)
print(f"Average gene correlation: {avg_gene_cor:.4f}")

#Latent Space Clustering Evaluation
#Evaluate how well your latent space separates clinical groups

from sklearn.metrics import silhouette_score, adjusted_rand_score

# Convert stages to numerical for silhouette score
stage_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4}
numeric_stages = np.array([stage_mapping[s] for s in rnaseq_adata.obs['cancer_stage']])

# Calculate silhouette score - higher is better (-1 to 1)
silhouette = silhouette_score(latent_features, numeric_stages)
print(f"Silhouette score for stage separation: {silhouette:.4f}")

# If you have performed clustering on the latent space:
from sklearn.cluster import KMeans

# Cluster the latent space
kmeans = KMeans(n_clusters=5, random_state=42).fit(latent_features)
cluster_labels = kmeans.labels_

# Compare clusters to true stages using Adjusted Rand Index
ari = adjusted_rand_score(numeric_stages, cluster_labels)
print(f"Adjusted Rand Index: {ari:.4f}")  # Higher is better (0 to 1)


#Feature Selection Stability
#To assess the stability of feature selection across multiple runs

def calculate_jaccard_index(set1, set2):
    """Calculate Jaccard index (intersection over union) for two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Run the model multiple times (e.g., with different seeds)
selected_features_runs = []
for seed in range(5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Run your model and get selected features
    # (simplified - in practice you'd need to re-run the full pipeline)
    _, _, _, _, rnaseq_selected, _, _ = run_multi_omics_integration(
        rna_adata, meta_adata, clinical_vars, seed=seed
    )

    selected_features_runs.append(set(np.where(rnaseq_selected)[0]))

# Calculate pairwise Jaccard indices
jaccard_indices = []
for i in range(len(selected_features_runs)):
    for j in range(i+1, len(selected_features_runs)):
        jaccard = calculate_jaccard_index(selected_features_runs[i], selected_features_runs[j])
        jaccard_indices.append(jaccard)

# Average stability
avg_jaccard = np.mean(jaccard_indices)
print(f"Feature selection stability (Jaccard index): {avg_jaccard:.4f}")


#Cross-Validation
#For robust evaluation, implement k-fold cross-validation

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    'stage_acc': [],
    'drug_auc': [],
    'rnaseq_mse': [],
    'metag_mse': []
}

for train_idx, test_idx in kf.split(range(len(rna_adata))):
    # Create train/test split
    train_rna = rna_adata[train_idx].copy()
    test_rna = rna_adata[test_idx].copy()
    train_meta = meta_adata[train_idx].copy()
    test_meta = meta_adata[test_idx].copy()

    # Train model
    model, _, _, _, _, _, _, _, metrics = run_multi_omics_integration(
        train_rna, train_meta, clinical_vars
    )

    # Evaluate on test set
    # (simplified - you'd need to create a proper evaluation function)
    cv_results['stage_acc'].append(evaluate_on_test_set(model, test_rna, test_meta)['stage_acc'])
    # Add other metrics...

# Print average results
for metric, values in cv_results.items():
    print(f"CV {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")


'''
These evaluation methods will give you a comprehensive picture of your model's performance across its various tasks. The most appropriate metrics will depend on your specific research questions and the downstream applications of your model.
'''
