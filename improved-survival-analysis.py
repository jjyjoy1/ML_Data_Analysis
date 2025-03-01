import os
import warnings
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
from joblib import delayed, dump, Parallel
from tabulate import tabulate

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestSurvivalAnalysis

# Import survival analysis models
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# Import custom cross-validation methods
from sksurv_extensions.model_selection import (
    SurvivalStratifiedShuffleSplit,
    SurvivalStratifiedSampleFromGroupShuffleSplit
)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_survival_data(csv_file):
    """
    Load survival data from CSV file.
    
    Parameters:
    csv_file (str): Path to CSV file
    
    Returns:
    tuple: (X, y, groups, group_weights)
        X: Feature dataframe
        y: Survival target
        groups: Group assignments for stratified CV (or None)
        group_weights: Weights for groups (or None)
    """
    print(f"Loading data from {csv_file}")
    
    # Load data from CSV
    data = pd.read_csv(csv_file, sep='\t', index=False)
    
    # Extract survival information
    # Assuming 'Status' is 1 for event, 0 for censored
    # and 'Survival_in_days' is the time to event/censoring
    try:
        y = Surv.from_dataframe('Status', 'Survival_in_days', data)
    except Exception as e:
        print(f"Error creating survival target: {e}")
        # Try different column names if standard ones aren't found
        if 'vital_status' in data.columns and 'days_to_death' in data.columns:
            # Convert vital_status to binary (1 for dead, 0 for alive)
            data['Status'] = (data['vital_status'] == 'Dead').astype(int)
            # Use days_to_death or days_to_last_followup based on status
            data['Survival_in_days'] = np.where(
                data['Status'] == 1,
                data['days_to_death'],
                data['days_to_last_followup']
            )
            y = Surv.from_dataframe('Status', 'Survival_in_days', data)
        else:
            raise ValueError("Could not identify survival columns in the data")
    
    # Create feature matrix, excluding survival information and potential group columns
    exclude_cols = ['Status', 'Survival_in_days', 'Group', 'GroupWeight', 
                   'patient_id', 'sample_id', 'vital_status', 'days_to_death', 
                   'days_to_last_followup']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    
    # Extract group information if available
    if 'Group' in data.columns:
        groups = np.array(data['Group'], dtype=int)
        if 'GroupWeight' in data.columns and data['GroupWeight'].nunique() > 1:
            group_weights = np.array(data['GroupWeight'], dtype=float)
        else:
            group_weights = None
    else:
        groups = None
        group_weights = None
    
    # Print data summary
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Event rate: {sum(y[0]) / len(y[0]):.2f}")
    
    return X, y, groups, group_weights


def preprocess_features(X):
    """
    Preprocess features, handling categorical variables and missing values.
    
    Parameters:
    X (pd.DataFrame): Feature dataframe
    
    Returns:
    pd.DataFrame: Preprocessed features
    """
    X_processed = X.copy()
    
    # Handle missing values
    numeric_cols = X_processed.select_dtypes(include=['number']).columns
    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    
    # Impute missing numeric values with median
    for col in numeric_cols:
        X_processed[col] = X_processed[col].fillna(X_processed[col].median())
    
    # Impute missing categorical values with most frequent value
    for col in categorical_cols:
        X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
    
    # One-hot encode categorical variables
    for col in categorical_cols:
        if X_processed[col].nunique() > 1:
            # Create dummy variables
            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
            # Add dummy variables to processed dataframe
            X_processed = pd.concat([X_processed, dummies], axis=1)
    
    # Drop original categorical columns
    X_processed = X_processed.drop(categorical_cols, axis=1)
    
    return X_processed


def create_model(model_type='cox', feature_selection='k_best', k=10, alpha=0.01):
    """
    Create a pipeline with preprocessing, feature selection, and survival model.
    
    Parameters:
    model_type (str): Type of survival model ('cox', 'gbm', 'rsf')
    feature_selection (str): Feature selection method ('k_best', 'model_based', None)
    k (int): Number of features to select if using k_best
    alpha (float): Regularization parameter for Cox model
    
    Returns:
    Pipeline: Scikit-learn pipeline
    """
    # Define preprocessing steps
    steps = [('scaler', StandardScaler())]
    
    # Add feature selection step
    if feature_selection == 'k_best':
        steps.append(('feature_selection', SelectKBest(k=k)))
    elif feature_selection == 'model_based':
        # Use Random Forest to select features
        estimator = RandomForestSurvivalAnalysis(
            n_estimators=100, 
            min_samples_split=10, 
            min_samples_leaf=15, 
            random_state=777
        )
        steps.append(('feature_selection', SelectFromModel(estimator)))
    
    # Add survival model
    if model_type == 'cox':
        steps.append(('model', CoxPHSurvivalAnalysis(
            alpha=alpha, n_iter=10000, ties='efron', tol=1e-09)))
    elif model_type == 'gbm':
        steps.append(('model', GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=777)))
    elif model_type == 'rsf':
        steps.append(('model', RandomForestSurvivalAnalysis(
            n_estimators=200, min_samples_split=10, min_samples_leaf=15, random_state=777)))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return Pipeline(steps)


def fit_models(X, y, groups, group_weights, test_splits, test_size, 
               random_seed=777, model_type='cox', feature_selection='k_best', k=10):
    """
    Fit survival models using cross-validation.
    
    Parameters:
    X (pd.DataFrame): Feature dataframe
    y (structured array): Survival target
    groups (array-like): Group assignments for stratified CV
    group_weights (array-like): Weights for groups
    test_splits (int): Number of test splits for CV
    test_size (float): Proportion of test set
    random_seed (int): Random seed
    model_type (str): Type of survival model
    feature_selection (str): Feature selection method
    k (int): Number of features to select if using k_best
    
    Returns:
    tuple: (split_models, split_results)
    """
    # Preprocess features
    X_processed = preprocess_features(X)
    
    # Create model pipeline
    pipe = create_model(model_type, feature_selection, k)
    
    # Set up cross-validation
    if groups is None:
        cv = SurvivalStratifiedShuffleSplit(
            n_splits=test_splits, test_size=test_size,
            random_state=random_seed)
        cv_split_params = {}
    else:
        cv = SurvivalStratifiedSampleFromGroupShuffleSplit(
            n_splits=test_splits, test_size=test_size,
            random_state=random_seed)
        cv_split_params = {'weights': group_weights}
    
    split_models, split_results = [], []
    for split_idx, (train_idxs, test_idxs) in enumerate(cv.split(X_processed, y, groups, **cv_split_params)):
        print(f"Fitting model for split {split_idx+1}/{test_splits}")
        try:
            # Get train and test sets
            X_train = X_processed.iloc[train_idxs]
            X_test = X_processed.iloc[test_idxs]
            y_train = y[train_idxs]
            y_test = y[test_idxs]
            
            # Fit model
            pipe.fit(X_train, y_train)
            
            # Calculate concordance index (C-index)
            score = pipe.score(X_test, y_test)
            
            # Make predictions
            y_pred = pipe.predict(X_test)
            
            # Generate survival functions if available
            try:
                surv_funcs = pipe.predict_survival_function(X_test)
            except:
                surv_funcs = None
            
            # Get feature importances if available
            feature_importances = None
            if hasattr(pipe.named_steps.get('model', None), 'feature_importances_'):
                feature_importances = pipe.named_steps['model'].feature_importances_
            elif hasattr(pipe.named_steps.get('model', None), 'coef_'):
                feature_importances = pipe.named_steps['model'].coef_
            
            if feature_importances is not None and feature_selection == 'k_best':
                selected_features = pipe.named_steps['feature_selection'].get_support()
                feature_names = X_processed.columns[selected_features]
                importances = dict(zip(feature_names, feature_importances))
            elif feature_importances is not None:
                importances = dict(zip(X_processed.columns, feature_importances))
            else:
                importances = None
            
        except Exception as e:
            print(f"Error in split {split_idx+1}: {e}")
            split_models.append(None)
            split_results.append(None)
        else:
            split_models.append(pipe)
            split_scores = {'test': {'score': score, 'y_pred': y_pred}}
            split_results.append({
                'scores': split_scores,
                'surv_funcs': surv_funcs,
                'feature_importances': importances
            })
    
    return split_models, split_results


def main():
    parser = ArgumentParser(description="Survival analysis with multiple models and feature selection")
    parser.add_argument('--data-dir', type=str, default='data', help='data directory')
    parser.add_argument('--results-dir', type=str, default='results/models', help='results directory')
    parser.add_argument('--model-type', type=str, default='cox', choices=['cox', 'gbm', 'rsf'], 
                        help='survival model type')
    parser.add_argument('--feature-selection', type=str, default='k_best', 
                        choices=['k_best', 'model_based', 'none'], help='feature selection method')
    parser.add_argument('--num-features', type=int, default=10, help='number of features to select')
    parser.add_argument('--test-splits', type=int, default=100, help='number of test splits')
    parser.add_argument('--test-size', type=float, default=0.25, help='test split size')
    parser.add_argument('--n-jobs', type=int, default=-1, help='number of parallel jobs')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
    parser.add_argument('--random-seed', type=int, default=777, help='random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = f"{args.results_dir}/surv_{args.model_type}"
    os.makedirs(out_dir, mode=0o755, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(glob(f"{args.data_dir}/tcga_*_surv_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each dataset
    all_models = []
    all_results = []
    mean_scores = []
    
    for csv_file in csv_files:
        file_basename = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\nProcessing {file_basename}")
        
        # Load data
        try:
            X, y, groups, group_weights = load_survival_data(csv_file)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
        
        # Fit models
        split_models, split_results = fit_models(
            X, y, groups, group_weights, 
            test_splits=args.test_splits,
            test_size=args.test_size,
            random_seed=args.random_seed,
            model_type=args.model_type,
            feature_selection=args.feature_selection if args.feature_selection != 'none' else None,
            k=args.num_features
        )
        
        all_models.append(split_models)
        all_results.append(split_results)
        
        # Calculate mean score
        scores = []
        for split_result in split_results:
            if split_result is not None:
                scores.append(split_result['scores']['test']['score'])
            else:
                scores.append(np.nan)
        
        # Parse dataset name components
        parts = file_basename.split('_')
        if len(parts) >= 4:
            cancer = parts[1]
            analysis = parts[2]
            target = parts[3]
            data_type = parts[4] if len(parts) > 4 else 'unknown'
        else:
            cancer = file_basename
            analysis = 'unknown'
            target = 'unknown'
            data_type = 'unknown'
        
        model_name = f"{file_basename}_{args.model_type}"
        if args.feature_selection != 'none':
            model_name += f"_{args.feature_selection}{args.num_features}"
        
        mean_score = np.nanmean(scores)
        mean_scores.append([analysis, cancer, target, data_type, args.model_type, 
                           args.feature_selection, args.num_features, mean_score])
        
        # Save results
        results_dir = f"{out_dir}/{model_name}"
        os.makedirs(results_dir, mode=0o755, exist_ok=True)
        
        dump(split_models, f"{results_dir}/{model_name}_split_models.pkl")
        dump(split_results, f"{results_dir}/{model_name}_split_results.pkl")
        
        # Save feature importances
        feature_importances = {}
        for i, split_result in enumerate(split_results):
            if split_result is not None and 'feature_importances' in split_result:
                if split_result['feature_importances'] is not None:
                    for feature, importance in split_result['feature_importances'].items():
                        if feature not in feature_importances:
                            feature_importances[feature] = []
                        feature_importances[feature].append(importance)
        
        if feature_importances:
            # Calculate mean importance for each feature
            mean_importances = {f: np.mean(imps) for f, imps in feature_importances.items()}
            # Sort by importance
            sorted_importances = sorted(mean_importances.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Save to file
            with open(f"{results_dir}/{model_name}_feature_importances.tsv", 'w') as f:
                f.write("Feature\tImportance\n")
                for feature, importance in sorted_importances:
                    f.write(f"{feature}\t{importance}\n")
    
    # Save overall results
    mean_scores_df = pd.DataFrame(mean_scores, columns=[
        'Analysis', 'Cancer', 'Target', 'Data Type', 'Model Type', 
        'Feature Selection', 'Num Features', 'Mean C-index'])
    
    mean_scores_df.to_csv(f"{out_dir}/survival_model_mean_scores.tsv", 
                         index=False, sep='\t')
    
    # Print summary
    if args.verbose > 0:
        print("\nModel Performance Summary:")
        print(tabulate(mean_scores_df.sort_values(
            ['Analysis', 'Cancer', 'Target', 'Data Type']),
            floatfmt='.4f', showindex=False, headers='keys'))


if __name__ == "__main__":
    main()
