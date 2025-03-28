# Cancer Status Prediction Pipeline

## Overview
This repository builds upon work from the Ruppin Lab published paper and extends it with additional machine learning models. Key improvements include:
- Conversion of all code from R to Python
- Replacement of R's `ExperimentSet` object with Python's `Anndata` object
- Integration of advanced deep learning models for robust analysis

## Pipeline Architecture

### 1. Problem Definition
**Outcome Definition:**
- Clearly specify "cancer status" (e.g., cancer vs. non-cancer, tumor grade, survival outcome)
- Determine classification vs. regression task framing

**Scope:**
- Option 1: Separate models per data type (clinical, transcriptomic, microbiome)
- Option 2: Integrated multi-modal model combining all sources

**Performance Metrics:**
- Standard metrics: Accuracy, AUC-ROC, precision/recall
- Special considerations for:
  - Class imbalances
  - Clinical relevance

### 2. Data Preprocessing
#### A. Clinical Metadata
| Task               | Approach                          |
|--------------------|-----------------------------------|
| Missing Data       | Imputation or removal            |
| Categorical Encoding | One-hot or label encoding       |
| Normalization      | Standardize numerical variables  |

#### B. RNA Sequence Data
- **Normalization:** TPM, FPKM, or DESeq2's VST
- **Filtering:** Remove lowly expressed genes
- **Transformation:** Log-transform skewed distributions

#### C. Metagenomics Taxonomy Data
- **Normalization:** Relative abundances or CLR transformation
- **Dimensionality Reduction:** Filter rare taxa or cluster related features

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Feature distributions
- **Correlation Analysis:** Clinical-omics relationships
- **Visualization:** PCA, t-SNE, UMAP for sample clustering

### 4. Feature Engineering
**Clinical Data:**
- Create composite variables (e.g., age groups)

**Omics Data:**
- Feature selection (differential expression analysis)
- Dimensionality reduction (PCA, autoencoders, LASSO)
- Biological context aggregation (pathways, taxonomies)

**Integration Strategies:**
| Type              | Description                      |
|-------------------|----------------------------------|
| Early Integration | Concatenate all feature vectors  |
| Late Integration  | Ensemble separate model outputs  |

### 5. Model Development
#### A. Algorithm Selection
**Classical ML:**
- Random Forests, SVM, XGBoost, LightGBM
- Regularized Logistic Regression

**Deep Learning:**
- Multi-layer perceptrons
- Multi-modal architectures
- **Extended Models:**
  - VAEs
  - MOLI
  - DeepOmix
  - MoGN

#### B. Training Process
- **Data Splitting:** Stratified train/validation/test
- **Cross-Validation:** k-fold for robust estimation
- **Hyperparameter Tuning:** Grid/Random/Bayesian search

### 6. Model Evaluation
**Performance Metrics:**
- Standard: AUC, accuracy
- Imbalanced data: F1-score, sensitivity/specificity

**Interpretability:**
- Feature importance scores
- SHAP/LIME explanations
- Biological enrichment analysis

### 7. Validation & Deployment
**Validation Strategy:**
1. Internal: Held-out test set
2. External: Independent dataset
3. Clinical: Expert consultation

**Deployment Considerations:**
- Data collection feasibility
- Processing time requirements
- Interpretation needs

### 8. Iterative Refinement
- Cycle through feature/model tuning
- Documentation with:
  - Jupyter notebooks
  - Version control (Git)
  - Workflow systems (Snakemake/Nextflow)

## Summary Workflow
1. Clearly define prediction task and metrics
2. Preprocess each data type appropriately
3. Conduct thorough EDA
4. Engineer and select meaningful features
5. Train and validate models systematically
6. Interpret results biologically and clinically
7. Prepare for potential clinical implementation

## Extended Documentation
For details on each added deep learning model (VAEs, MOLI, DeepOmix, MoGN), please see the respective model documentation files.

