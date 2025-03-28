import os
import warnings
from argparse import ArgumentParser
from glob import glob
from itertools import product

import numpy as np
import pandas as pd
from joblib import delayed, dump, Parallel
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, balanced_accuracy_score,
    precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn_extensions.model_selection import RepeatedStratifiedGroupKFold

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning,
                      message='The max_iter was reached which means the coef_ did not converge')


def calculate_test_scores(pipe, X_test, y_test, pipe_predict_params=None,
                        test_sample_weights=None):
    """
    Calculate performance metrics for the model on test data.
    
    Parameters:
    -----------
    pipe : sklearn.pipeline.Pipeline
        Fitted model pipeline
    X_test : pandas.DataFrame
        Test features
    y_test : numpy.ndarray
        Test labels
    pipe_predict_params : dict, optional
        Additional parameters for prediction
    test_sample_weights : numpy.ndarray, optional
        Sample weights for test data
        
    Returns:
    --------
    dict
        Dictionary containing calculated scores
    """
    if pipe_predict_params is None:
        pipe_predict_params = {}
    
    scores = {}
    
    # Get prediction scores
    if hasattr(pipe, 'decision_function'):
        y_score = pipe.decision_function(X_test, **pipe_predict_params)
    else:
        y_score = pipe.predict_proba(X_test, **pipe_predict_params)[:, 1]
    
    scores['y_score'] = y_score
    
    # Calculate metrics
    for metric in metrics:
        if metric == 'roc_auc':
            scores[metric] = roc_auc_score(
                y_test, y_score, sample_weight=test_sample_weights)
            scores['fpr'], scores['tpr'], _ = roc_curve(
                y_test, y_score, pos_label=1,
                sample_weight=test_sample_weights)
        elif metric == 'balanced_accuracy':
            y_pred = pipe.predict(X_test, **pipe_predict_params)
            scores['y_pred'] = y_pred
            scores[metric] = balanced_accuracy_score(
                y_test, y_pred, sample_weight=test_sample_weights)
        elif metric == 'average_precision':
            scores[metric] = average_precision_score(
                y_test, y_score, sample_weight=test_sample_weights)
            scores['pre'], scores['rec'], _ = precision_recall_curve(
                y_test, y_score, pos_label=1,
                sample_weight=test_sample_weights)
            scores['pr_auc'] = auc(scores['rec'], scores['pre'])
    
    return scores


def fit_models(pipe, X, y, groups, sample_weights, test_splits, test_repeats):
    """
    Fit models using cross-validation and calculate performance metrics.
    
    Parameters:
    -----------
    pipe : sklearn.pipeline.Pipeline
        Model pipeline to fit
    X : pandas.DataFrame
        Features
    y : numpy.ndarray
        Labels
    groups : numpy.ndarray or None
        Groups for grouped cross-validation
    sample_weights : numpy.ndarray or None
        Sample weights
    test_splits : int
        Number of CV splits
    test_repeats : int
        Number of CV repeats
        
    Returns:
    --------
    list
        List of dictionaries containing scores for each split
    """
    # Set up cross-validation
    if groups is None:
        cv = RepeatedStratifiedKFold(n_splits=test_splits,
                                   n_repeats=test_repeats,
                                   random_state=random_seed)
    else:
        cv = RepeatedStratifiedGroupKFold(n_splits=test_splits,
                                        n_repeats=test_repeats,
                                        random_state=random_seed)

    split_results = []
    
    # For each CV split
    for train_idxs, test_idxs in cv.split(X, y, groups):
        train_sample_weights = None
        test_sample_weights = None
        
        if sample_weights is not None:
            train_sample_weights = sample_weights[train_idxs]
            test_sample_weights = sample_weights[test_idxs]
        
        # Extract dataset for this split
        X_train, X_test = X.iloc[train_idxs], X.iloc[test_idxs]
        y_train, y_test = y[train_idxs], y[test_idxs]
        
        # Fit model - use appropriate sample_weight parameter based on the model type
        if 'sample_weight' in pipe.fit_params_names:
            pipe.fit(X_train, y_train, sample_weight=train_sample_weights)
        else:
            # For pipelines where the classifier is not the first step
            fit_params = {}
            if train_sample_weights is not None:
                # Find the classifier step name
                for step_name, step in pipe.steps:
                    if hasattr(step, 'fit') and hasattr(step, 'predict'):
                        fit_params[f'{step_name}__sample_weight'] = train_sample_weights
                        break
            pipe.fit(X_train, y_train, **fit_params)
        
        # Calculate test scores
        split_scores = {'te': calculate_test_scores(
            pipe, X_test, y_test, pipe_predict_params={},
            test_sample_weights=test_sample_weights)}
        
        # Record important features if model supports it
        if hasattr(pipe.named_steps.get('feature_selection', None), 'get_support'):
            selected_features = np.array(X.columns)[pipe.named_steps['feature_selection'].get_support()]
            split_scores['selected_features'] = selected_features.tolist()
        
        split_results.append({'scores': split_scores})

    return split_results


def load_dataset(file_path):
    """
    Load dataset from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
        
    Returns:
    --------
    tuple
        (X, y, groups, sample_weights, filename_parts)
    """
    # Extract information from filename
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    filename_parts = file_basename.split('_')
    
    # Read CSV data
    df = pd.read_csv(file_path)
    
    # Extract target variable
    y = np.array(df['Class'], dtype=int)
    
    # Extract features (remove non-feature columns)
    non_feature_cols = ['Class']
    
    # Handle groups for stratified group k-fold if present
    groups = None
    sample_weights = None
    
    if 'Group' in df.columns:
        groups = np.array(df['Group'], dtype=int)
        non_feature_cols.append('Group')
        
        # Calculate sample weights based on group sizes
        _, group_indices, group_counts = np.unique(
            groups, return_inverse=True, return_counts=True)
        sample_weights = (np.max(group_counts) / group_counts)[group_indices]
    
    # Create feature dataframe (excluding non-feature columns)
    X = df.drop(columns=[col for col in non_feature_cols if col in df.columns])
    
    # Return data and metadata
    return X, y, groups, sample_weights, filename_parts


def main():
    parser = ArgumentParser(description='Drug response prediction from clinical data')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing CSV data files')
    parser.add_argument('--results-dir', type=str, default='results/models', help='Directory to save results')
    parser.add_argument('--test-splits', type=int, help='Number of CV test splits')
    parser.add_argument('--test-repeats', type=int, help='Number of CV test repeats')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--parallel-backend', type=str, default='loky', help='Joblib parallel backend')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--feature-selection', type=str, default='rfe', 
                      choices=['rfe', 'model_based', 'none'], 
                      help='Feature selection method')
    args = parser.parse_args()

    # Create output directory
    out_dir = f'{args.results_dir}/resp'
    os.makedirs(out_dir, mode=0o755, exist_ok=True)

    # Process all CSV files in data directory
    csv_files = sorted(glob(f'{args.data_dir}/tcga_*_resp_*.csv'))
    num_files = len(csv_files)
    
    if num_files == 0:
        print(f"No matching CSV files found in {args.data_dir}")
        return
    
    print(f"Found {num_files} CSV files to process")
    
    # Load all datasets
    datasets = []
    for file_idx, file_path in enumerate(csv_files):
        if args.verbose:
            print(f'Loading {file_idx + 1}/{num_files}: {os.path.basename(file_path)}')
        
        X, y, groups, sample_weights, filename_parts = load_dataset(file_path)
        
        # Set cross-validation parameters based on dataset or command line args
        _, cancer, _, target, *_ = filename_parts
        cancer_target = f'{cancer}_{target}'
        
        if args.test_splits is None:
            test_splits = 3 if cancer_target == 'stad_oxaliplatin' else 4
        else:
            test_splits = args.test_splits
            
        if args.test_repeats is None:
            test_repeats = 33 if cancer_target == 'stad_oxaliplatin' else 25
        else:
            test_repeats = args.test_repeats
        
        datasets.append((X, y, groups, sample_weights, test_splits, test_repeats, file_path))
    
    print(f'Running drug response clinical models using {args.feature_selection} feature selection')
    
    # Define model pipelines
    pipes = []
    
    # SVM with feature selection
    if args.feature_selection == 'rfe':
        # RFE feature selection
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(
                estimator=SVC(kernel='linear', class_weight='balanced', random_state=random_seed),
                n_features_to_select=5,  # Select top 5 features
                step=1
            )),
            ('classifier', SVC(kernel='linear', class_weight='balanced', 
                             probability=True, random_state=random_seed))
        ]))
    elif args.feature_selection == 'model_based':
        # Model-based feature selection
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                estimator=SVC(kernel='linear', class_weight='balanced', random_state=random_seed),
                threshold='mean'
            )),
            ('classifier', SVC(kernel='linear', class_weight='balanced', 
                             probability=True, random_state=random_seed))
        ]))
    else:
        # No feature selection
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', class_weight='balanced', 
                             probability=True, random_state=random_seed))
        ]))
    
    # Logistic Regression with feature selection
    if args.feature_selection == 'rfe':
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(
                estimator=LogisticRegression(penalty='l2', class_weight='balanced', 
                                           random_state=random_seed),
                n_features_to_select=5,
                step=1
            )),
            ('classifier', LogisticRegression(
                penalty='l2', solver='saga', max_iter=5000,
                class_weight='balanced', random_state=random_seed))
        ]))
    elif args.feature_selection == 'model_based':
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                estimator=LogisticRegression(penalty='l2', class_weight='balanced', 
                                           random_state=random_seed),
                threshold='mean'
            )),
            ('classifier', LogisticRegression(
                penalty='l2', solver='saga', max_iter=5000,
                class_weight='balanced', random_state=random_seed))
        ]))
    else:
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l2', solver='saga', max_iter=5000,
                class_weight='balanced', random_state=random_seed))
        ]))
    
    # Random Forest (new model)
    if args.feature_selection != 'none':
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced',
                random_state=random_seed))
        ]))
    
    # Gradient Boosting (new model)
    if args.feature_selection != 'none':
        pipes.append(Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100, max_depth=3, 
                random_state=random_seed))
        ]))
    
    # Run all models in parallel
    all_results = Parallel(n_jobs=args.n_jobs, backend=args.parallel_backend,
                         verbose=args.verbose)(
        delayed(fit_models)(pipe, X, y, groups, sample_weights, test_splits, test_repeats)
        for pipe, (X, y, groups, sample_weights, test_splits, test_repeats, _) in 
        product(pipes, datasets)
    )
    
    # Process results
    mean_scores = []
    feature_importance_summary = []
    
    for (pipe, (_, _, _, _, _, _, file_path)), split_results in zip(
            product(pipes, datasets), all_results):
        
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        _, cancer, analysis, target, data_type, *rest = file_basename.split('_')
        
        # Extract scores
        roc_scores, pr_scores = [], []
        for split_result in split_results:
            roc_scores.append(split_result['scores']['te']['roc_auc'])
            pr_scores.append(split_result['scores']['te']['pr_auc'])
        
        # Determine model type
        model_code = None
        if isinstance(pipe.named_steps.get('classifier', None), SVC):
            model_code = 'svm'
        elif isinstance(pipe.named_steps.get('classifier', None), LogisticRegression):
            model_code = 'lgr'
        elif isinstance(pipe.named_steps.get('classifier', None), RandomForestClassifier):
            model_code = 'rf'
        elif isinstance(pipe.named_steps.get('classifier', None), GradientBoostingClassifier):
            model_code = 'gbm'
            
        # Feature selection method
        if args.feature_selection == 'rfe':
            fs_code = 'rfe'
        elif args.feature_selection == 'model_based':
            fs_code = 'sfm'
        else:
            fs_code = 'none'
        
        dataset_name = '_'.join(file_basename.split('_')[:-1])
        model_name = f'{dataset_name}_{model_code}_{fs_code}_clinical'
        
        # Calculate mean scores
        mean_roc = np.nanmean(roc_scores)
        mean_pr = np.nanmean(pr_scores)
        
        mean_scores.append([
            analysis, cancer, target, data_type, 
            model_code, fs_code, mean_roc, mean_pr
        ])
        
        # Save split results
        results_dir = f'{out_dir}/{model_name}'
        os.makedirs(results_dir, mode=0o755, exist_ok=True)
        dump(split_results, f'{results_dir}/{model_name}_split_results.pkl')
        
        # Collect feature importance information if available
        if 'selected_features' in split_results[0]['scores']:
            # Get all selected features across splits
            all_selected_features = []
            for split_result in split_results:
                if 'selected_features' in split_result['scores']:
                    all_selected_features.extend(split_result['scores']['selected_features'])
            
            # Count feature occurrences
            feature_counts = {}
            for feature in all_selected_features:
                if feature not in feature_counts:
                    feature_counts[feature] = 0
                feature_counts[feature] += 1
            
            # Sort by frequency
            sorted_features = sorted(feature_counts.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            # Add to summary
            feature_importance_summary.append({
                'model_name': model_name,
                'top_features': sorted_features[:10],  # Top 10 features
                'total_splits': len(split_results)
            })
    
    # Create results dataframe
    mean_scores_df = pd.DataFrame(mean_scores, columns=[
        'Analysis', 'Cancer', 'Target', 'Data Type', 'Model Code', 
        'Feature Selection', 'Mean ROC AUC', 'Mean PR AUC'
    ])
    
    # Save results
    mean_scores_df.to_csv(f'{out_dir}/resp_clinical_model_mean_scores.tsv',
                        index=False, sep='\t')
    
    # Print summary table
    if args.verbose > 0:
        print("\nModel Performance Summary:")
        print(tabulate(
            mean_scores_df.sort_values(['Analysis', 'Cancer', 'Target', 'Data Type', 'Model Code']),
            floatfmt='.4f', showindex=False, headers='keys'
        ))
        
        # Print feature importance summary if available
        if feature_importance_summary:
            print("\nTop Selected Features Summary:")
            for model_info in feature_importance_summary:
                print(f"\nModel: {model_info['model_name']}")
                print(f"Feature selection frequency across {model_info['total_splits']} CV splits:")
                for feature, count in model_info['top_features']:
                    percent = (count / model_info['total_splits']) * 100
                    print(f"  {feature}: {count} times ({percent:.1f}%)")


if __name__ == '__main__':
    # Global configuration
    random_seed = 777
    metrics = ['roc_auc', 'average_precision', 'balanced_accuracy']
    
    main()
