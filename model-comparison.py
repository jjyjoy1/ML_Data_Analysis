import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from scipy import stats
from pathlib import Path
from tabulate import tabulate
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from matplotlib.gridspec import GridSpec

def load_result_files(results_dir):
    """
    Load all model result files from the directory
    
    Parameters:
    -----------
    results_dir : str
        Directory containing model results
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with combined model results
    """
    print(f"Loading results from {results_dir}")
    
    # Load the TSV file with summary results
    summary_file = os.path.join(results_dir, 'resp_clinical_model_mean_scores.tsv')
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    summary_df = pd.read_csv(summary_file, sep='\t')
    
    # Filter for RFE feature selection if specified
    feature_selection = 'rfe'
    summary_df = summary_df[summary_df['Feature Selection'] == feature_selection]
    
    # Map model codes to readable names
    model_name_map = {
        'svm': 'Support Vector Machine',
        'lgr': 'Logistic Regression',
        'rf': 'Random Forest',
        'gbm': 'Gradient Boosting'
    }
    
    # Add readable model names
    summary_df['Model Name'] = summary_df['Model Code'].map(model_name_map)
    
    print(f"Found {len(summary_df)} model results with RFE feature selection")
    
    return summary_df

def load_detailed_results(results_dir, summary_df):
    """
    Load detailed results for each model from pickle files
    
    Parameters:
    -----------
    results_dir : str
        Base directory containing model results
    summary_df : pd.DataFrame
        Summary dataframe with model metadata
        
    Returns:
    --------
    dict
        Dictionary with detailed results for each model
    """
    detailed_results = {}
    resp_dir = os.path.join(results_dir, 'resp')
    
    for _, row in summary_df.iterrows():
        # Construct model directory and result file name
        analysis = row['Analysis']
        cancer = row['Cancer']
        target = row['Target']
        data_type = row['Data Type']
        model_code = row['Model Code']
        fs_code = row['Feature Selection']
        
        dataset_name = f"tcga_{cancer}_{analysis}_{target}_{data_type}"
        model_name = f"{dataset_name}_{model_code}_{fs_code}_clinical"
        model_dir = os.path.join(resp_dir, model_name)
        
        # Check if directory exists
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found: {model_dir}")
            continue
        
        # Find result file
        result_file = os.path.join(model_dir, f"{model_name}_split_results.pkl")
        if not os.path.exists(result_file):
            print(f"Warning: Result file not found: {result_file}")
            continue
        
        # Load detailed results
        try:
            split_results = load(result_file)
            detailed_results[model_name] = {
                'summary': row,
                'split_results': split_results
            }
            print(f"Loaded detailed results for {model_name}")
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return detailed_results

def create_performance_comparison(summary_df):
    """
    Create bar plots comparing model performance across cancer types
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe with model results
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with performance comparison
    """
    plt.figure(figsize=(14, 10))
    
    # Create subplots based on cancer type
    cancer_types = summary_df['Cancer'].unique()
    n_cancers = len(cancer_types)
    
    gs = GridSpec(n_cancers, 2, figure=plt.gcf(), hspace=0.4, wspace=0.3)
    
    # Set up color palette
    palette = sns.color_palette("viridis", 4)
    
    for i, cancer in enumerate(sorted(cancer_types)):
        cancer_data = summary_df[summary_df['Cancer'] == cancer]
        
        # ROC AUC plot
        ax1 = plt.subplot(gs[i, 0])
        sns.barplot(x='Target', y='Mean ROC AUC', hue='Model Name', data=cancer_data, 
                   palette=palette, ax=ax1)
        ax1.set_title(f"{cancer.upper()} - ROC AUC")
        ax1.set_ylim(0.5, 1.0)  # AUC is typically 0.5-1.0
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        if i < n_cancers - 1:
            ax1.set_xlabel('')
            ax1.legend([])
        else:
            ax1.set_xlabel('Drug Target')
            ax1.legend(title='Model', loc='lower right')
        
        # PR AUC plot
        ax2 = plt.subplot(gs[i, 1])
        sns.barplot(x='Target', y='Mean PR AUC', hue='Model Name', data=cancer_data, 
                   palette=palette, ax=ax2)
        ax2.set_title(f"{cancer.upper()} - PR AUC")
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        if i < n_cancers - 1:
            ax2.set_xlabel('')
        else:
            ax2.set_xlabel('Drug Target')
        
        # Only show legend in last row
        if i < n_cancers - 1:
            ax2.legend([])
        else:
            ax2.legend(title='Model', loc='lower right')
    
    plt.suptitle('Model Performance Comparison by Cancer Type and Drug Target (with RFE Feature Selection)', 
                fontsize=16, y=0.98)
    
    return plt.gcf()

def plot_model_rankings(summary_df):
    """
    Plot model rankings across all datasets
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe with model results
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with model rankings
    """
    # Create a dataset identifier
    summary_df['Dataset'] = summary_df.apply(
        lambda row: f"{row['Cancer']}_{row['Target']}", axis=1)
    
    # Rank models for each dataset based on ROC AUC
    ranking_df = pd.DataFrame()
    
    for dataset in summary_df['Dataset'].unique():
        dataset_df = summary_df[summary_df['Dataset'] == dataset].copy()
        dataset_df['ROC_Rank'] = dataset_df['Mean ROC AUC'].rank(ascending=False)
        dataset_df['PR_Rank'] = dataset_df['Mean PR AUC'].rank(ascending=False)
        ranking_df = pd.concat([ranking_df, dataset_df])
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    # Plot average ranks
    plt.subplot(1, 2, 1)
    average_ranks = ranking_df.groupby('Model Name')[['ROC_Rank', 'PR_Rank']].mean().reset_index()
    
    # Melt for easier plotting
    avg_ranks_melted = pd.melt(average_ranks, 
                              id_vars=['Model Name'], 
                              value_vars=['ROC_Rank', 'PR_Rank'],
                              var_name='Metric', value_name='Avg Rank')
    
    # Replace metric names for better readability
    avg_ranks_melted['Metric'] = avg_ranks_melted['Metric'].replace({
        'ROC_Rank': 'ROC AUC Rank', 
        'PR_Rank': 'PR AUC Rank'
    })
    
    sns.barplot(x='Model Name', y='Avg Rank', hue='Metric', data=avg_ranks_melted, palette='Set2')
    plt.axhline(y=2.5, linestyle='--', color='gray', alpha=0.7)  # Midpoint for 4 models
    plt.ylim(0.5, 4.5)
    plt.gca().invert_yaxis()  # Lower rank is better
    plt.title('Average Model Ranking Across All Datasets')
    plt.ylabel('Average Rank (lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot rank distribution
    plt.subplot(1, 2, 2)
    
    # Count rank occurrences
    rank_counts = pd.DataFrame({
        'ROC Rank 1': ranking_df.groupby('Model Name')['ROC_Rank'].apply(lambda x: sum(x == 1)),
        'ROC Rank 2': ranking_df.groupby('Model Name')['ROC_Rank'].apply(lambda x: sum(x == 2)),
        'ROC Rank 3': ranking_df.groupby('Model Name')['ROC_Rank'].apply(lambda x: sum(x == 3)),
        'ROC Rank 4': ranking_df.groupby('Model Name')['ROC_Rank'].apply(lambda x: sum(x == 4)),
    }).reset_index()
    
    # Melt for easier plotting
    rank_counts_melted = pd.melt(rank_counts, 
                                id_vars=['Model Name'], 
                                var_name='Rank', 
                                value_name='Count')
    
    sns.barplot(x='Model Name', y='Count', hue='Rank', data=rank_counts_melted, palette='YlOrRd_r')
    plt.title('Distribution of ROC AUC Ranks Across Datasets')
    plt.ylabel('Number of Datasets')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.suptitle('Model Ranking Analysis', fontsize=16, y=1.02)
    
    return plt.gcf()

def statistical_comparison(summary_df):
    """
    Perform statistical comparisons between models
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe with model results
        
    Returns:
    --------
    dict
        Dictionary with statistical test results
    """
    # Create a dataset identifier
    summary_df['Dataset'] = summary_df.apply(
        lambda row: f"{row['Cancer']}_{row['Target']}", axis=1)
    
    # Prepare data for paired tests
    model_names = sorted(summary_df['Model Name'].unique())
    n_models = len(model_names)
    
    # Prepare results table for ROC AUC
    roc_p_values = np.zeros((n_models, n_models))
    
    # For each pair of models
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                continue
                
            # Get paired samples
            paired_data = []
            for dataset in summary_df['Dataset'].unique():
                model1_val = summary_df[(summary_df['Dataset'] == dataset) & 
                                      (summary_df['Model Name'] == model1)]['Mean ROC AUC'].values
                
                model2_val = summary_df[(summary_df['Dataset'] == dataset) & 
                                      (summary_df['Model Name'] == model2)]['Mean ROC AUC'].values
                
                if len(model1_val) == 1 and len(model2_val) == 1:
                    paired_data.append((model1_val[0], model2_val[0]))
            
            if len(paired_data) > 1:
                # Convert to arrays
                model1_values = np.array([x[0] for x in paired_data])
                model2_values = np.array([x[1] for x in paired_data])
                
                # Perform paired t-test
                _, p_value = stats.ttest_rel(model1_values, model2_values)
                roc_p_values[i, j] = p_value
    
    # Create p-value table
    p_value_table = pd.DataFrame(roc_p_values, 
                               index=model_names, 
                               columns=model_names)
    
    # Calculate average performance across datasets
    avg_performance = summary_df.groupby('Model Name')[['Mean ROC AUC', 'Mean PR AUC']].mean()
    std_performance = summary_df.groupby('Model Name')[['Mean ROC AUC', 'Mean PR AUC']].std()
    
    performance_table = pd.DataFrame({
        'Avg ROC AUC': avg_performance['Mean ROC AUC'],
        'Std ROC AUC': std_performance['Mean ROC AUC'],
        'Avg PR AUC': avg_performance['Mean PR AUC'],
        'Std PR AUC': std_performance['Mean PR AUC']
    })
    
    return {
        'p_values': p_value_table,
        'performance': performance_table
    }

def plot_roc_curves(detailed_results, selected_datasets=None):
    """
    Plot ROC curves for models across different datasets
    
    Parameters:
    -----------
    detailed_results : dict
        Dictionary with detailed model results
    selected_datasets : list or None
        List of dataset names to plot, or None for all datasets
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with ROC curves
    """
    # Group results by dataset
    datasets = {}
    
    for model_name, results in detailed_results.items():
        parts = model_name.split('_')
        cancer = parts[1]
        target = parts[3]
        model_code = parts[4]
        
        dataset_key = f"{cancer}_{target}"
        
        if selected_datasets and dataset_key not in selected_datasets:
            continue
            
        if dataset_key not in datasets:
            datasets[dataset_key] = {}
            
        # Map model codes to readable names
        model_name_map = {
            'svm': 'Support Vector Machine',
            'lgr': 'Logistic Regression',
            'rf': 'Random Forest',
            'gbm': 'Gradient Boosting'
        }
        
        readable_name = model_name_map.get(model_code, model_code)
        datasets[dataset_key][readable_name] = results
    
    # Plot ROC curves
    n_datasets = len(datasets)
    if n_datasets == 0:
        print("No datasets to plot")
        return None
        
    # Determine grid layout
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i, (dataset_key, models) in enumerate(sorted(datasets.items())):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Random guess line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        # For each model
        for (model_name, results), color in zip(models.items(), colors):
            # Average ROC curve across CV splits
            mean_fpr = np.linspace(0, 1, 100)
            tprs = []
            aucs = []
            
            # For each CV split
            for split_result in results['split_results']:
                if 'fpr' in split_result['scores']['te'] and 'tpr' in split_result['scores']['te']:
                    fpr = split_result['scores']['te']['fpr']
                    tpr = split_result['scores']['te']['tpr']
                    
                    # Interpolate to standard grid
                    if len(fpr) > 1 and len(tpr) > 1:
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        aucs.append(split_result['scores']['te']['roc_auc'])
            
            # Calculate mean and standard deviation
            if tprs:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                
                # Plot the mean ROC curve
                plt.plot(mean_fpr, mean_tpr, color=color, 
                       label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
                
                # Plot the standard deviation
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                               color=color, alpha=0.2)
        
        # Formatting
        cancer, target = dataset_key.split('_')
        plt.title(f'{cancer.upper()} - {target}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='lower right', fontsize='small')
    
    plt.tight_layout()
    plt.suptitle('ROC Curves Comparison Across Cancer Types and Drug Targets', 
               fontsize=16, y=1.02)
    
    return plt.gcf()

def plot_feature_importance(detailed_results):
    """
    Plot feature importance across models
    
    Parameters:
    -----------
    detailed_results : dict
        Dictionary with detailed model results
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with feature importance
    """
    # Collect feature importance data
    feature_data = []
    
    for model_name, results in detailed_results.items():
        parts = model_name.split('_')
        cancer = parts[1]
        target = parts[3]
        model_code = parts[4]
        
        # Map model codes to readable names
        model_name_map = {
            'svm': 'Support Vector Machine',
            'lgr': 'Logistic Regression',
            'rf': 'Random Forest',
            'gbm': 'Gradient Boosting'
        }
        
        readable_name = model_name_map.get(model_code, model_code)
        
        # For each CV split, collect selected features
        all_features = {}
        total_splits = len(results['split_results'])
        
        for split_idx, split_result in enumerate(results['split_results']):
            if 'selected_features' in split_result['scores']:
                for feature in split_result['scores']['selected_features']:
                    if feature not in all_features:
                        all_features[feature] = 0
                    all_features[feature] += 1
        
        # Calculate feature importance as percentage of splits
        for feature, count in all_features.items():
            percentage = (count / total_splits) * 100
            feature_data.append({
                'Cancer': cancer,
                'Target': target,
                'Model': readable_name,
                'Feature': feature,
                'Importance': percentage
            })
    
    if not feature_data:
        print("No feature importance data available")
        return None
        
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    # Get the top features across all models
    top_features = feature_df.groupby('Feature')['Importance'].mean().nlargest(15).index.tolist()
    
    # Filter for top features
    plot_df = feature_df[feature_df['Feature'].isin(top_features)]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot feature importance
    g = sns.catplot(
        data=plot_df, 
        kind="bar",
        x="Feature", y="Importance", hue="Model",
        palette="viridis", alpha=.8, height=6, aspect=2
    )
    
    g.set_xticklabels(rotation=45, ha="right")
    g.set(xlabel='Feature', ylabel='Selection Frequency (%)')
    g.fig.suptitle('Top Features Selected by RFE Across Models', fontsize=16, y=1.02)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    return g.fig

def generate_comparison_report(results_dir, output_dir='comparison_results'):
    """
    Generate a comprehensive comparison report for all models
    
    Parameters:
    -----------
    results_dir : str
        Directory containing model results
    output_dir : str
        Directory to save comparison results
        
    Returns:
    --------
    None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model results
    summary_df = load_result_files(results_dir)
    detailed_results = load_detailed_results(results_dir, summary_df)
    
    # 1. Generate performance comparison bar plots
    print("Generating performance comparison plots...")
    perf_fig = create_performance_comparison(summary_df)
    perf_fig.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. Generate model ranking plots
    print("Generating model ranking plots...")
    rank_fig = plot_model_rankings(summary_df)
    rank_fig.savefig(os.path.join(output_dir, 'model_rankings.png'), dpi=300, bbox_inches='tight')
    
    # 3. Generate ROC curves
    print("Generating ROC curves...")
    roc_fig = plot_roc_curves(detailed_results)
    if roc_fig:
        roc_fig.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    
    # 4. Generate feature importance plots
    print("Generating feature importance plots...")
    feat_fig = plot_feature_importance(detailed_results)
    if feat_fig:
        feat_fig.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # 5. Statistical comparison
    print("Performing statistical comparison...")
    stats_results = statistical_comparison(summary_df)
    
    # Save statistical results to CSV
    stats_results['p_values'].to_csv(os.path.join(output_dir, 'pairwise_pvalues.csv'))
    stats_results['performance'].to_csv(os.path.join(output_dir, 'model_performance_stats.csv'))
    
    # Create a summary text file
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=======================\n\n")
        
        f.write("Average Performance Across Datasets:\n")
        f.write(tabulate(stats_results['performance'], headers='keys', floatfmt='.4f'))
        f.write("\n\n")
        
        f.write("Pairwise Statistical Comparison (p-values from paired t-tests):\n")
        f.write(tabulate(stats_results['p_values'], headers='keys', floatfmt='.4f'))
        f.write("\n\n")
        
        # Add best model for each dataset
        f.write("Best Model for Each Dataset (by ROC AUC):\n")
        for dataset in summary_df['Dataset'].unique():
            dataset_df = summary_df[summary_df['Dataset'] == dataset]
            best_idx = dataset_df['Mean ROC AUC'].idxmax()
            best_model = dataset_df.loc[best_idx]
            f.write(f"{dataset}: {best_model['Model Name']} (ROC AUC: {best_model['Mean ROC AUC']:.4f})\n")
    
    print(f"Comparison report generated and saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model comparison visualizations')
    parser.add_argument('--results-dir', type=str, default='results/models',
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Directory to save comparison results')
    args = parser.parse_args()
    
    generate_comparison_report(args.results_dir, args.output_dir)
