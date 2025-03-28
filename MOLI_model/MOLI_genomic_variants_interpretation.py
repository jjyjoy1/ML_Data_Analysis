import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create synthetic genomic variant data
def create_synthetic_variant_data(n_samples=1000, random_state=42):
    """
    Create synthetic genomic variant data with multiple objectives:
    1. Pathogenicity score
    2. Expression impact score
    3. Structural impact score
    4. Conservation score
    """
    np.random.seed(random_state)
    
    # Create feature data for variants
    # Features might include:
    # - Position-based features (distance to exon boundaries, etc.)
    # - Sequence-based features (GC content, motif disruption scores)
    # - Protein-based features (amino acid changes, domain disruption)
    n_features = 50
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Some features are more important for specific objectives
    pathogenicity_features = np.random.choice(n_features, 15, replace=False)
    expression_features = np.random.choice(n_features, 15, replace=False)
    structure_features = np.random.choice(n_features, 15, replace=False)
    conservation_features = np.random.choice(n_features, 15, replace=False)
    
    # Create ground truth for each objective with some overlap in signals
    # but also with specific features for each objective
    
    # Base signals with some noise
    pathogenicity = np.random.normal(0, 0.2, size=n_samples)
    expression_impact = np.random.normal(0, 0.2, size=n_samples)
    structural_impact = np.random.normal(0, 0.2, size=n_samples)
    conservation = np.random.normal(0, 0.2, size=n_samples)
    
    # Add feature-specific effects
    for i in range(n_samples):
        # Pathogenicity influenced by its specific features
        pathogenicity[i] += np.sum(X[i, pathogenicity_features]) * 0.3
        
        # Expression impact influenced by its specific features
        expression_impact[i] += np.sum(X[i, expression_features]) * 0.3
        
        # Structural impact influenced by its specific features
        structural_impact[i] += np.sum(X[i, structure_features]) * 0.3
        
        # Conservation influenced by its specific features
        conservation[i] += np.sum(X[i, conservation_features]) * 0.3
    
    # Normalize scores to reasonable ranges
    pathogenicity = (pathogenicity - pathogenicity.min()) / (pathogenicity.max() - pathogenicity.min())
    expression_impact = (expression_impact - expression_impact.min()) / (expression_impact.max() - expression_impact.min())
    structural_impact = (structural_impact - structural_impact.min()) / (structural_impact.max() - structural_impact.min())
    conservation = (conservation - conservation.min()) / (conservation.max() - conservation.min())
    
    # Binary classification for known disease associations (derived partly from pathogenicity)
    disease_assoc = (pathogenicity > 0.7).astype(int)
    
    # Create DataFrame
    variant_ids = [f'var_{i}' for i in range(n_samples)]
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X_df = pd.DataFrame(X, columns=feature_names, index=variant_ids)
    
    # Add target values
    y_df = pd.DataFrame({
        'pathogenicity': pathogenicity,
        'expression_impact': expression_impact,
        'structural_impact': structural_impact,
        'conservation': conservation,
        'disease_association': disease_assoc
    }, index=variant_ids)
    
    # Add variant metadata
    metadata = pd.DataFrame({
        'variant_id': variant_ids,
        'chromosome': np.random.choice(['chr1', 'chr2', 'chr3', 'chr4', 'chr5'], n_samples),
        'position': np.random.randint(1, 250000000, n_samples),
        'reference': np.random.choice(['A', 'C', 'G', 'T'], n_samples),
        'alternate': np.random.choice(['A', 'C', 'G', 'T'], n_samples)
    }, index=variant_ids)
    
    # Make sure ref != alt
    for i, row in metadata.iterrows():
        while row['reference'] == row['alternate']:
            row['alternate'] = np.random.choice(['A', 'C', 'G', 'T'])
    
    # Create a feature importance reference for later evaluation
    feature_importance = {
        'pathogenicity': {f: 1.0 if i in pathogenicity_features else 0.1 for i, f in enumerate(feature_names)},
        'expression_impact': {f: 1.0 if i in expression_features else 0.1 for i, f in enumerate(feature_names)},
        'structural_impact': {f: 1.0 if i in structure_features else 0.1 for i, f in enumerate(feature_names)},
        'conservation': {f: 1.0 if i in conservation_features else 0.1 for i, f in enumerate(feature_names)}
    }
    
    return X_df, y_df, metadata, feature_importance

# Preprocess data
def preprocess_data(X, y):
    """
    Preprocess variant data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Build a traditional single-objective model (for comparison)
def build_single_objective_model(input_dim, output_dim=1, output_activation='sigmoid'):
    """
    Build a traditional neural network model for a single objective
    """
    inputs = Input(shape=(input_dim,))
    
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(output_dim, activation=output_activation)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build MOLI model for multi-objective genomic variant interpretation
def build_moli_model(input_dim):
    """
    Build a Multi-Objective Learning through Inference (MOLI) model
    for genomic variant interpretation with four objectives:
    1. Pathogenicity score prediction (regression)
    2. Expression impact prediction (regression)
    3. Structural impact prediction (regression)
    4. Conservation score prediction (regression)
    """
    # Shared input
    inputs = Input(shape=(input_dim,))
    
    # Shared representation layers
    shared = Dense(256, activation='relu')(inputs)
    shared = Dropout(0.3)(shared)
    shared = Dense(128, activation='relu')(shared)
    shared = Dropout(0.3)(shared)
    
    # Task-specific branches with private parameters
    
    # 1. Pathogenicity branch
    pathogenicity_branch = Dense(64, activation='relu')(shared)
    pathogenicity_branch = Dense(32, activation='relu')(pathogenicity_branch)
    pathogenicity_output = Dense(1, activation='sigmoid', name='pathogenicity')(pathogenicity_branch)
    
    # 2. Expression impact branch
    expression_branch = Dense(64, activation='relu')(shared)
    expression_branch = Dense(32, activation='relu')(expression_branch)
    expression_output = Dense(1, activation='sigmoid', name='expression_impact')(expression_branch)
    
    # 3. Structural impact branch
    structure_branch = Dense(64, activation='relu')(shared)
    structure_branch = Dense(32, activation='relu')(structure_branch)
    structure_output = Dense(1, activation='sigmoid', name='structural_impact')(structure_branch)
    
    # 4. Conservation branch
    conservation_branch = Dense(64, activation='relu')(shared)
    conservation_branch = Dense(32, activation='relu')(conservation_branch)
    conservation_output = Dense(1, activation='sigmoid', name='conservation')(conservation_branch)
    
    # 5. Disease association branch (using information from other branches)
    # This is where the inference part of MOLI comes in - using predictions from other tasks
    combined_features = Concatenate()([
        pathogenicity_branch,
        expression_branch,
        structure_branch,
        conservation_branch
    ])
    
    disease_branch = Dense(64, activation='relu')(combined_features)
    disease_branch = Dense(32, activation='relu')(disease_branch)
    disease_output = Dense(1, activation='sigmoid', name='disease_association')(disease_branch)
    
    # MOLI model with multiple outputs
    model = Model(
        inputs=inputs, 
        outputs=[
            pathogenicity_output,
            expression_output,
            structure_output,
            conservation_output,
            disease_output
        ]
    )
    
    # Compile with appropriate loss functions and metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'pathogenicity': 'mse',
            'expression_impact': 'mse',
            'structural_impact': 'mse',
            'conservation': 'mse',
            'disease_association': 'binary_crossentropy'
        },
        metrics={
            'pathogenicity': ['mae', 'mse'],
            'expression_impact': ['mae', 'mse'],
            'structural_impact': ['mae', 'mse'],
            'conservation': ['mae', 'mse'],
            'disease_association': ['accuracy', 'AUC']
        }
    )
    
    return model

# Train and evaluate MOLI model
def train_and_evaluate_moli(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the MOLI model and evaluate its performance
    """
    # Prepare target data for training
    y_train_dict = {
        'pathogenicity': y_train['pathogenicity'].values,
        'expression_impact': y_train['expression_impact'].values,
        'structural_impact': y_train['structural_impact'].values,
        'conservation': y_train['conservation'].values,
        'disease_association': y_train['disease_association'].values
    }
    
    # Prepare target data for testing
    y_test_dict = {
        'pathogenicity': y_test['pathogenicity'].values,
        'expression_impact': y_test['expression_impact'].values,
        'structural_impact': y_test['structural_impact'].values,
        'conservation': y_test['conservation'].values,
        'disease_association': y_test['disease_association'].values
    }
    
    # Early stopping based on validation loss
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    print("\nModel Evaluation:")
    results = model.evaluate(X_test, y_test_dict, verbose=0)
    
    # Get predictions for more detailed evaluation
    predictions = model.predict(X_test)
    
    # Create prediction dictionary for easier access
    pred_dict = {
        'pathogenicity': predictions[0].flatten(),
        'expression_impact': predictions[1].flatten(),
        'structural_impact': predictions[2].flatten(), 
        'conservation': predictions[3].flatten(),
        'disease_association': predictions[4].flatten()
    }
    
    # Calculate additional metrics for each objective
    for objective in ['pathogenicity', 'expression_impact', 'structural_impact', 'conservation']:
        y_true = y_test_dict[objective]
        y_pred = pred_dict[objective]
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{objective.replace('_', ' ').title()} Scores:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    # For disease association (binary classification)
    y_true = y_test_dict['disease_association']
    y_pred = pred_dict['disease_association']
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = np.mean(y_pred_binary == y_true)
    auc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    print("\nDisease Association Scores:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 3, 1)
    for key in history.history.keys():
        if 'loss' in key and 'val' not in key and 'disease' not in key:
            plt.plot(history.history[key], label=key)
    plt.title('Training Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot disease association metrics
    plt.subplot(2, 3, 2)
    plt.plot(history.history['disease_association_accuracy'], label='accuracy')
    plt.plot(history.history['val_disease_association_accuracy'], label='val_accuracy')
    plt.title('Disease Association Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot correlation between objectives
    plt.subplot(2, 3, 3)
    objectives = ['pathogenicity', 'expression_impact', 'structural_impact', 'conservation']
    corr_matrix = y_test[objectives].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Objectives')
    
    # Plot predictions vs actual for pathogenicity
    plt.subplot(2, 3, 4)
    plt.scatter(y_test['pathogenicity'], pred_dict['pathogenicity'], alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Pathogenicity')
    plt.ylabel('Predicted Pathogenicity')
    plt.title('Pathogenicity Prediction')
    
    # Plot disease association distribution by pathogenicity
    plt.subplot(2, 3, 5)
    sns.boxplot(x='disease_association', y='pathogenicity', data=y_test)
    plt.title('Disease Association vs Pathogenicity')
    
    # Plot comparing model predicting accuracy across objectives
    plt.subplot(2, 3, 6)
    objectives = ['pathogenicity', 'expression_impact', 'structural_impact', 'conservation']
    r2_scores = []
    
    for obj in objectives:
        y_true = y_test_dict[obj]
        y_pred = pred_dict[obj]
        r2_scores.append(r2_score(y_true, y_pred))
    
    plt.bar(objectives, r2_scores)
    plt.ylim(0, 1)
    plt.title('R² Score by Objective')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return model, history, pred_dict

# Compare with single-objective approach (train separate models)
def train_single_objective_models(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train separate models for each objective
    """
    objectives = ['pathogenicity', 'expression_impact', 'structural_impact', 'conservation', 'disease_association']
    models = {}
    predictions = {}
    metrics = {}
    
    for objective in objectives:
        print(f"\nTraining model for {objective}...")
        
        # Binary classification for disease association
        if objective == 'disease_association':
            model = build_single_objective_model(X_train.shape[1], output_activation='sigmoid')
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Regression for other objectives
            model = build_single_objective_model(X_train.shape[1], output_activation='sigmoid')
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train, y_train[objective].values,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Save model
        models[objective] = model
        
        # Get predictions
        preds = model.predict(X_test).flatten()
        predictions[objective] = preds
        
        # Calculate metrics
        if objective == 'disease_association':
            pred_binary = (preds > 0.5).astype(int)
            accuracy = np.mean(pred_binary == y_test[objective].values)
            auc = roc_auc_score(y_test[objective].values, preds)
            
            metrics[objective] = {
                'accuracy': accuracy,
                'auc': auc
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
        else:
            mse = mean_squared_error(y_test[objective].values, preds)
            r2 = r2_score(y_test[objective].values, preds)
            
            metrics[objective] = {
                'mse': mse,
                'r2': r2
            }
            
            print(f"  MSE: {mse:.4f}")
            print(f"  R²: {r2:.4f}")
    
    return models, predictions, metrics

# Compare MOLI vs single-objective approach
def compare_approaches(y_test, moli_preds, single_preds):
    """
    Compare the performance of MOLI vs single-objective models
    """
    objectives = ['pathogenicity', 'expression_impact', 'structural_impact', 'conservation']
    
    plt.figure(figsize=(12, 8))
    
    # Compare R² scores for regression objectives
    plt.subplot(2, 2, 1)
    moli_r2 = []
    single_r2 = []
    
    for obj in objectives:
        moli_r2.append(r2_score(y_test[obj].values, moli_preds[obj]))
        single_r2.append(r2_score(y_test[obj].values, single_preds[obj]))
    
    x = np.arange(len(objectives))
    width = 0.35
    
    plt.bar(x - width/2, moli_r2, width, label='MOLI')
    plt.bar(x + width/2, single_r2, width, label='Single-Objective')
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison')
    plt.xticks(x, [obj.replace('_', ' ').title() for obj in objectives])
    plt.legend()
    
    # Compare disease association prediction
    plt.subplot(2, 2, 2)
    moli_auc = roc_auc_score(y_test['disease_association'].values, moli_preds['disease_association'])
    single_auc = roc_auc_score(y_test['disease_association'].values, single_preds['disease_association'])
    
    plt.bar(['MOLI', 'Single-Objective'], [moli_auc, single_auc])
    plt.ylabel('AUC Score')
    plt.title('Disease Association AUC Comparison')
    
    # Compare MSE for regression objectives
    plt.subplot(2, 2, 3)
    moli_mse = []
    single_mse = []
    
    for obj in objectives:
        moli_mse.append(mean_squared_error(y_test[obj].values, moli_preds[obj]))
        single_mse.append(mean_squared_error(y_test[obj].values, single_preds[obj]))
    
    plt.bar(x - width/2, moli_mse, width, label='MOLI')
    plt.bar(x + width/2, single_mse, width, label='Single-Objective')
    plt.ylabel('MSE (lower is better)')
    plt.title('MSE Comparison')
    plt.xticks(x, [obj.replace('_', ' ').title() for obj in objectives])
    plt.legend()
    
    # Compare pathogenicity predictions scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(y_test['pathogenicity'], moli_preds['pathogenicity'], alpha=0.3, label='MOLI')
    plt.scatter(y_test['pathogenicity'], single_preds['pathogenicity'], alpha=0.3, label='Single')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Pathogenicity')
    plt.ylabel('Predicted Pathogenicity')
    plt.title('Prediction Comparison for Pathogenicity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nPerformance Summary:")
    print("                    MOLI    Single-Objective")
    
    # R² for regression tasks
    for obj in objectives:
        moli_r2 = r2_score(y_test[obj].values, moli_preds[obj])
        single_r2 = r2_score(y_test[obj].values, single_preds[obj])
        print(f"{obj.replace('_', ' ').title()} R²:    {moli_r2:.4f}    {single_r2:.4f}")
    
    # AUC for classification task
    moli_auc = roc_auc_score(y_test['disease_association'].values, moli_preds['disease_association'])
    single_auc = roc_auc_score(y_test['disease_association'].values, single_preds['disease_association'])
    print(f"Disease AUC:        {moli_auc:.4f}    {single_auc:.4f}")

# Analyze a specific variant
def analyze_variant(model, variant_id, X, feature_names, scaler):
    """
    Analyze a specific variant using the MOLI model
    """
    # Get variant features
    variant_features = X.loc[variant_id].values.reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(variant_features)
    
    # Get predictions
    predictions = model.predict(scaled_features)
    
    # Format predictions
    results = {
        'pathogenicity': predictions[0][0][0],
        'expression_impact': predictions[1][0][0],
        'structural_impact': predictions[2][0][0],
        'conservation': predictions[3][0][0],
        'disease_association': predictions[4][0][0]
    }
    
    print(f"\nVariant Analysis: {variant_id}")
    print("-" * 40)
    
    for objective, score in results.items():
        if objective == 'disease_association':
            print(f"{objective.replace('_', ' ').title()}: {score:.4f} ({'High' if score > 0.5 else 'Low'} probability)")
        else:
            impact_level = 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
            print(f"{objective.replace('_', ' ').title()}: {score:.4f} ({impact_level} impact)")
    
    # MOLI-specific interpretation
    print("\nIntegrated Interpretation:")
    
    # Summarize the overall impact based on all objectives
    if results['pathogenicity'] > 0.7 and results['disease_association'] > 0.6:
        print("- This variant is likely pathogenic and disease-associated")
    elif results['pathogenicity'] > 0.5 and results['disease_association'] > 0.3:
        print("- This variant is possibly pathogenic with some disease relevance")
    else:
        print("- This variant is likely benign with low disease relevance")
    
    # Provide mechanism insights based on the different scores
    print("\nPossible Mechanism:")
    
    # Expression effect
    if results['expression_impact'] > 0.7:
        print("- Strong effect on gene expression")
    elif results['expression_impact'] > 0.4:
        print("- Moderate effect on gene expression")
    
    # Structural effect
    if results['structural_impact'] > 0.7:
        print("- Significant impact on protein structure")
    elif results['structural_impact'] > 0.4:
        print("- Moderate impact on protein structure")
    
    # Conservation insights
    if results['conservation'] > 0.7:
        print("- Located in a highly conserved region, suggesting functional importance")
    elif results['conservation'] > 0.4:
        print("- Located in a moderately conserved region")
    
    # Plot relative impact across objectives
    plt.figure(figsize=(10, 6))
    
    # Bar chart of scores
    plt.subplot(1, 2, 1)
    objectives = list(results.keys())
    scores = list(results.values())
    
    colors = ['#ff9999' if score > 0.7 else '#ffcc99' if score > 0.4 else '#99ff99' for score in scores]
    
    plt.bar(range(len(objectives)), scores, color=colors)
    plt.xticks(range(len(objectives)), [obj.replace('_', ' ').title() for obj in objectives], rotation=45)
    plt.ylim(0, 1)
    plt.title(f'Variant {variant_id} Impact Scores')
    
    # Radar chart for multi-objective visualization
    plt.subplot(1, 2, 2)
    
    # Prepare radar chart
    N = len(objectives) - 1  # Exclude disease_association for the radar
    radar_objectives = objectives[:-1]
    radar_scores = scores[:-1]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    radar_scores += radar_scores[:1]  # Close the loop
    
    ax = plt.subplot(1, 2, 2, polar=True)
    ax.plot(angles, radar_scores, 'o-', linewidth=2)
    ax.fill(angles, radar_scores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), [obj.replace('_', ' ').title() for obj in radar_objectives])
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Objective Impact Profile')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Main workflow
def main():
    print("Creating synthetic genomic variant data...")
    X, y, metadata, feature_importance = create_synthetic_variant_data(n_samples=1000)
    
    print(f"Dataset: {X.shape[0]} variants, {X.shape[1]} features")
    print(f"Objectives: {', '.join(y.columns)}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train MOLI model
    print("\nBuilding and training MOLI model...")
    moli_model = build_moli_model(input_dim=X.shape[1])
    moli_model, history, moli_preds = train_and_evaluate_moli(
        moli_model, X_train, y_train, X_test, y_test, epochs=30
    )
    
    # Train single-objective models for comparison
    print("\nTraining single-objective models for comparison...")
    single_models, single_preds, single_metrics = train_single_objective_models(
        X_train, y_train, X_test, y_test, epochs=30
    )
    
    # Compare approaches
    print("\nComparing MOLI vs single-objective approach...")
    compare_approaches(y_test, moli_preds, single_preds)
    
    # Analyze a specific variant using MOLI model
    variant_id = X.index[0]  # Choose first variant for demonstration
    analyze_variant(moli_model, variant_id, X, X.columns, scaler)
    
    print("\nMOLI provides an integrated approach to variant interpretation")
    print("by simultaneously considering multiple biological factors")
    print("and producing a holistic assessment of variant impact.")

if __name__ == "__main__":
    main()
