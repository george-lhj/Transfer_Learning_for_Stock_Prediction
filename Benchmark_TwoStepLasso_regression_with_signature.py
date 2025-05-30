import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)

def train_test_split_by_time(df, train_ratio=0.7):
    """
    Split dataset into training and testing sets by time order
    
    Parameters:
    df: DataFrame, containing date and feature data
    train_ratio: float, proportion of training set
    
    Returns:
    df_train, df_test: training and testing sets
    """
    # Ensure data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df) * train_ratio)
    split_date = df.iloc[split_idx]['date']
    
    print(f"Training set end date: {split_date}")
    print(f"Training samples: {split_idx}")
    print(f"Testing samples: {len(df) - split_idx}")
    
    # Display return statistics
    print("\nTraining set return statistics:")
    print(df.iloc[:split_idx]['RET'].describe())
    print("\nTesting set return statistics:")
    print(df.iloc[split_idx:]['RET'].describe())
    
    return df.iloc[:split_idx], df.iloc[split_idx:]

def evaluate_predictions(y_true, y_pred):
    """
    Calculate multiple regression evaluation metrics
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate predicted and true directions
    pred_direction = y_pred > 0
    true_direction = y_true > 0
    
    # Calculate direction accuracy
    direction_accuracy = np.mean(pred_direction == true_direction)
    
    # Calculate detailed metrics
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Direction_Accuracy': direction_accuracy,
        # Add more detailed direction prediction information
        'True_Positive_Rate': np.mean(pred_direction[true_direction]),  # Proportion of correctly predicted increases
        'True_Negative_Rate': np.mean(~pred_direction[~true_direction]),  # Proportion of correctly predicted decreases
    }
    return metrics

def two_step_lasso_regression_by_group(df_train, df_test, group_col='Sector', use_weights=False):
    """
    Perform two-step lasso regression by specified column
    Step 1: Use Lasso for feature selection
    Step 2: Use OLS (Linear Regression) on selected features
    
    Parameters:
    df_train: DataFrame, training set
    df_test: DataFrame, testing set
    group_col: str, grouping column name, e.g., 'Sector' or 'Industry'
    use_weights: bool, whether to use adaptive weights in training
    
    Returns:
    results: DataFrame, containing results for each group
    y_true_all: list, all true values
    y_pred_all: list, all predicted values
    """
    # Set parameter grid for Lasso (first step)
    param_grid = {"alpha": np.logspace(-5, 1, num=10)}
    
    results = []
    y_true_all = []
    y_pred_all = []
    
    # Get all feature columns
    feature_cols = [col for col in df_train.columns if col.startswith('SIG_')]
    
    # Train for each group
    for group in tqdm(df_train[group_col].unique(), desc=f"Processing {group_col}s"):
        try:
            # Get current group data
            df_group_train = df_train[df_train[group_col] == group]
            df_group_test = df_test[df_test[group_col] == group]
            
            # If using weights, first remove samples with NaN weights
            if use_weights and 'adaptive_weight' in df_group_train.columns:
                # Record sample count before removal
                original_size = len(df_group_train)
                # Remove samples with NaN weights
                df_group_train = df_group_train.dropna(subset=['adaptive_weight'])
                # Print number of removed samples
                removed_samples = original_size - len(df_group_train)
                if removed_samples > 0:
                    print(f"Removed {removed_samples} samples with NaN weights from {group}")
            
            if len(df_group_train) < 10 or len(df_group_test) < 5:
                print(f"Warning: {group_col} {group} has insufficient samples, skipping")
                continue
            
            # Prepare features and labels
            X_train = df_group_train[feature_cols].values
            y_train = df_group_train['RET'].values
            X_test = df_group_test[feature_cols].values
            y_test = df_group_test['RET'].values
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # STEP 1: Grid search for Lasso (feature selection)
            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(
                estimator=Lasso(random_state=42, max_iter=100000, tol=1e-3),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Get training weights (if using)
            if use_weights and 'adaptive_weight' in df_group_train.columns:
                weights_train = df_group_train['adaptive_weight'].values
                # Grid search with sample weights
                gs.fit(X_train_scaled, y_train, sample_weight=weights_train)
            else:
                # Normal training without weights
                gs.fit(X_train_scaled, y_train)
            
            # Get best Lasso model from grid search
            lasso_model = gs.best_estimator_
            
            # Get non-zero coefficient indices and their values
            non_zero_indices = np.where(lasso_model.coef_ != 0)[0]
            non_zero_coeffs = np.sum(lasso_model.coef_ != 0)
            sparsity = 1.0 - (non_zero_coeffs / len(lasso_model.coef_))
            
            # 增强型应急措施 - 当没有特征被选择时
            if len(non_zero_indices) == 0:
                print(f"Warning: No features selected for {group_col} {group}. Attempting enhanced recovery...")
                
                # 方法1: 尝试一系列更小的alpha值
                alpha_values = np.logspace(-10, -4, 7)  # 比标准网格中的值更小
                best_alpha = None
                max_non_zero = 0
                
                for alpha in alpha_values:
                    recovery_lasso = Lasso(alpha=alpha, max_iter=100000, tol=1e-3, random_state=42)
                    recovery_lasso.fit(X_train_scaled, y_train)
                    n_nonzero = np.sum(recovery_lasso.coef_ != 0)
                    
                    # 如果找到非零特征，保存最好的结果
                    if n_nonzero > max_non_zero:
                        max_non_zero = n_nonzero
                        best_alpha = alpha
                        temp_indices = np.where(recovery_lasso.coef_ != 0)[0]
                        # 如果找到至少3个特征，就停止搜索
                        if len(temp_indices) >= 3:
                            non_zero_indices = temp_indices
                            lasso_model = recovery_lasso
                            print(f"  Recovery successful with alpha={alpha}: found {len(non_zero_indices)} features")
                            break
                
                # 如果仍然没有找到足够的特征，使用相关性方法
                if len(non_zero_indices) < 3:
                    print(f"  Still insufficient features ({len(non_zero_indices)}). Using correlation-based selection.")
                    
                    # 计算每个特征与目标的相关性
                    correlations = []
                    for i, feature in enumerate(feature_cols):
                        corr = np.corrcoef(df_group_train[feature], df_group_train['RET'])[0, 1]
                        if not np.isnan(corr):  # 忽略NaN相关性
                            correlations.append((i, abs(corr)))
                    
                    # 按相关性排序并选择前k个特征
                    if correlations:
                        k = min(5, len(correlations))  # 最多选择5个特征
                        top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:k]
                        non_zero_indices = np.array([idx for idx, _ in top_features])
                        print(f"  Selected {len(non_zero_indices)} features based on correlation")
                    else:
                        # 最后的后备方案：选择方差最大的几个特征
                        variances = np.var(X_train_scaled, axis=0)
                        non_zero_indices = np.argsort(variances)[-5:]  # 选择方差最大的5个特征
                        print(f"  Selected {len(non_zero_indices)} features based on variance")
                
                # 更新稀疏性指标
                non_zero_coeffs = len(non_zero_indices)
                sparsity = 1.0 - (non_zero_coeffs / len(feature_cols))
            
            # Create feature names for selected features for later interpretation
            selected_feature_names = [feature_cols[i] for i in non_zero_indices]
            
            # STEP 2: OLS on selected features
            # Extract selected features
            X_train_selected = X_train_scaled[:, non_zero_indices]
            X_test_selected = X_test_scaled[:, non_zero_indices]
            
            # Train OLS model (LinearRegression)
            ols_model = LinearRegression()
            
            # Use sample weights if available
            if use_weights and 'adaptive_weight' in df_group_train.columns:
                ols_model.fit(X_train_selected, y_train, sample_weight=weights_train)
            else:
                ols_model.fit(X_train_selected, y_train)
            
            # Predict using OLS model
            y_pred = ols_model.predict(X_test_selected)
            
            # Evaluate predictions
            metrics = evaluate_predictions(y_test, y_pred)
            
            # Save results
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            # Calculate return statistics
            train_ret_stats = df_group_train['RET'].describe()
            test_ret_stats = df_group_test['RET'].describe()
            
            result_dict = {
                group_col: group,
                "Train_Samples": len(df_group_train),
                "Test_Samples": len(df_group_test),
                "Best_Alpha": gs.best_params_["alpha"] if len(non_zero_indices) > 0 else (best_alpha if best_alpha else 1e-10),
                "Non_Zero_Coeffs": non_zero_coeffs,
                "Total_Coeffs": len(lasso_model.coef_),
                "Sparsity": sparsity,
                "MSE": metrics['MSE'],
                "RMSE": metrics['RMSE'],
                "MAE": metrics['MAE'],
                "R2": metrics['R2'],
                "Direction_Accuracy": metrics['Direction_Accuracy'],
                "Train_Return_Mean": train_ret_stats['mean'],
                "Train_Return_Std": train_ret_stats['std'],
                "Test_Return_Mean": test_ret_stats['mean'],
                "Test_Return_Std": test_ret_stats['std'],
                "Selected_Features": selected_feature_names,
                "Lasso_Model": lasso_model,
                "OLS_Model": ols_model,
                "Used_Emergency_Procedure": len(non_zero_indices) == 0
            }
            
            # Add Sector information if grouping by Industry
            if group_col == 'Industry':
                result_dict["Sector"] = df_group_train['Sector'].iloc[0]
                
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing {group_col} {group}: {str(e)}")
            continue
    
    return pd.DataFrame(results), y_true_all, y_pred_all

def extract_params_from_filename(filename):
    """
    Extract parameters from filename
    Example: sig_data_SP500_w30_o3_d3_weighted_gamma0.5.parquet -> 
    {'w': 30, 'o': 3, 'd': 3, 'weighted': True, 'gamma': 0.5}
    """
    params = {}
    # Remove .parquet extension and split
    parts = filename.replace('.parquet', '').split('_')
    
    # Only process parts containing parameters
    param_parts = [p for p in parts if any(p.startswith(x) and len(p) > 1 and p[1:].replace('.', '').isdigit() 
                                         for x in ['w', 'o', 'd'])]
    
    for part in param_parts:
        if part.startswith('w'):
            params['w'] = int(part[1:])
        elif part.startswith('o'):
            params['o'] = int(part[1:])
        elif part.startswith('d'):
            params['d'] = int(part[1:])
    
    # Check if weighted parameter is included
    params['weighted'] = 'weighted' in filename
    if params['weighted']:
        # Extract gamma value
        gamma_part = [p for p in parts if p.startswith('gamma')][0]
        params['gamma'] = float(gamma_part.replace('gamma', ''))
    
    # Verify basic parameters exist
    if not all(key in params for key in ['w', 'o', 'd']):
        raise ValueError(f"Could not extract all parameters from filename: {filename}")
    
    return params

def main():
    # Define signature parameters to test
    sig_files = [
        # "sig_data_SP500_w5_o3_d3.parquet",
        # "sig_data_SP500_w10_o3_d3.parquet",
        # "sig_data_SP500_w15_o3_d3.parquet",
        # "sig_data_SP500_w30_o3_d3.parquet",
        # "sig_data_SP500_w40_o3_d3.parquet",
        # "sig_data_SP500_w50_o3_d3.parquet",
        # "sig_data_SP500_w60_o3_d3.parquet",

        # "sig_data_SP500_w5_o4_d3.parquet",
        # "sig_data_SP500_w10_o4_d3.parquet",
        # "sig_data_SP500_w15_o4_d3.parquet",
        # "sig_data_SP500_w30_o4_d3.parquet",
        # "sig_data_SP500_w40_o4_d3.parquet",
        # "sig_data_SP500_w50_o4_d3.parquet",
        # "sig_data_SP500_w60_o4_d3.parquet",

        # "sig_data_SP500_w5_o5_d3.parquet",
        # "sig_data_SP500_w10_o5_d3.parquet",
        # "sig_data_SP500_w15_o5_d3.parquet",
        # "sig_data_SP500_w30_o5_d3.parquet",
        # "sig_data_SP500_w40_o5_d3.parquet",
        # "sig_data_SP500_w50_o5_d3.parquet",
        # "sig_data_SP500_w60_o5_d3.parquet",

        # "sig_data_SP500_w10_o3_d3_clusters.parquet",
        # "sig_data_SP500_w10_o4_d3_clusters.parquet",
        # "sig_data_SP500_w10_o5_d3_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_clusters.parquet",
        # "sig_data_SP500_w15_o4_d3_clusters.parquet",
        # "sig_data_SP500_w15_o5_d3_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_clusters.parquet",
        # "sig_data_SP500_w30_o4_d3_clusters.parquet",
        # "sig_data_SP500_w30_o5_d3_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_clusters.parquet",
        # "sig_data_SP500_w50_o3_d3_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_clusters.parquet",

        # "sig_data_SP500_w10_o3_d3_weighted_gamma0.125_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_weighted_gamma0.125_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_weighted_gamma0.125_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_weighted_gamma0.125_clusters.parquet",
        "sig_data_SP500_w50_o3_d3_weighted_gamma0.125_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_weighted_gamma0.125_clusters.parquet",

        # "sig_data_SP500_w10_o3_d3_weighted_gamma0.25_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_weighted_gamma0.25_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_weighted_gamma0.25_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_weighted_gamma0.25_clusters.parquet",
        # "sig_data_SP500_w50_o3_d3_weighted_gamma0.25_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_weighted_gamma0.25_clusters.parquet",

        # "sig_data_SP500_w10_o3_d3_weighted_gamma0.5_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_weighted_gamma0.5_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_weighted_gamma0.5_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_weighted_gamma0.5_clusters.parquet",
        # "sig_data_SP500_w50_o3_d3_weighted_gamma0.5_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_weighted_gamma0.5_clusters.parquet",

        # "sig_data_SP500_w10_o3_d3_weighted_gamma1_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_weighted_gamma1_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_weighted_gamma1_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_weighted_gamma1_clusters.parquet",
        # "sig_data_SP500_w50_o3_d3_weighted_gamma1_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_weighted_gamma1_clusters.parquet",

        # "sig_data_SP500_w10_o3_d3_weighted_gamma5_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_weighted_gamma5_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_weighted_gamma5_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_weighted_gamma5_clusters.parquet",
        # "sig_data_SP500_w50_o3_d3_weighted_gamma5_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_weighted_gamma5_clusters.parquet",

        # "sig_data_SP500_w10_o3_d3_weighted_gamma10_clusters.parquet",
        # "sig_data_SP500_w15_o3_d3_weighted_gamma10_clusters.parquet",
        # "sig_data_SP500_w30_o3_d3_weighted_gamma10_clusters.parquet",
        # "sig_data_SP500_w40_o3_d3_weighted_gamma10_clusters.parquet",
        # "sig_data_SP500_w50_o3_d3_weighted_gamma10_clusters.parquet",
        # "sig_data_SP500_w60_o3_d3_weighted_gamma10_clusters.parquet",
    ]
    
    for sig_file in sig_files:
        print(f"\nProcessing {sig_file}...")
        
        # Extract parameters from filename
        params = extract_params_from_filename(sig_file)
        
        # Load data
        file_path = f"./datasets/{sig_file}"
        if not os.path.exists(file_path):
            print(f"Error: Signature data file {file_path} not found! Skipping...")
            continue
            
        print(f"Loading signature data from: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded signature data, shape: {df.shape}")
        
        # Convert date to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Split dataset by time
        print("\nSplitting dataset by time...")
        df_train, df_test = train_test_split_by_time(df, train_ratio=0.7)
        
        # Define grouping columns to test - prioritize CLUSTER if it exists
        if 'CLUSTER' in df.columns:
            print("\nCLUSTER column found. Using CLUSTER as the grouping variable.")
            group_columns = ['CLUSTER']
        else:
            print("\nNo CLUSTER column found. Using Sector as the grouping variable.")
            group_columns = ['Sector']
        
        # Test each grouping column
        for group_col in group_columns:
            print(f"\nStarting two-step Lasso model training by {group_col}...")
            results_df, y_true_all, y_pred_all = two_step_lasso_regression_by_group(
                df_train, 
                df_test, 
                group_col,
                use_weights=params.get('weighted', False)
            )
            
            if len(results_df) == 0:
                print(f"No results generated for {group_col}. Skipping...")
                continue
            
            # Output results
            print(f"\nTraining results for each {group_col}:")
            available_cols = [col for col in results_df.columns 
                            if col in [group_col, 'Sector', 'Train_Samples', 'Test_Samples', 
                                    'Best_Alpha', 'Non_Zero_Coeffs', 'Sparsity', 'MSE', 'RMSE', 'MAE', 'R2', 
                                    'Direction_Accuracy']]
            
            # Select display columns based on grouping type
            if group_col == 'Industry' and 'Sector' in available_cols:
                display_cols = ['Industry', 'Sector'] + [col for col in available_cols 
                              if col not in ['Industry', 'Sector']]
            else:
                display_cols = [group_col] + [col for col in available_cols 
                              if col != group_col]
            
            print(results_df[display_cols])
            
            # Calculate overall evaluation metrics
            overall_metrics = evaluate_predictions(y_true_all, y_pred_all)
            print(f"\n{group_col} overall evaluation metrics:")
            for metric_name, metric_value in overall_metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            
            # Calculate number of unique stocks in each group
            group_symbol_counts = df[['symbol', group_col]].groupby(group_col)['symbol'].nunique()
            
            # Only keep symbol counts for groups that successfully ran regression
            group_symbol_counts = group_symbol_counts[group_symbol_counts.index.isin(results_df[group_col])]
            
            # Add symbol counts to results_df
            results_df = results_df.copy()
            results_df['Symbol_Count'] = results_df[group_col].map(group_symbol_counts)
            
            # If CLUSTER column, reset index (reorder)
            if group_col == 'CLUSTER':
                results_df[group_col] = range(len(results_df))
            
            # Save results with complete parameter suffix
            if params.get('weighted', False):
                param_suffix = f"_w{params['w']}_o{params['o']}_d{params['d']}_weighted_gamma{params['gamma']}"
            else:
                param_suffix = f"_w{params['w']}_o{params['o']}_d{params['d']}"
            
            # If input filename contains "cluster", add _cluster suffix to output filename
            if "cluster" in sig_file:
                param_suffix += "_cluster"
                
            output_file = f"./datasets/twostep_lasso_results_{group_col.lower()}{param_suffix}.csv"
            
            # Drop model objects before saving to CSV
            results_to_save = results_df.drop(columns=['Lasso_Model', 'OLS_Model', 'Selected_Features'], errors='ignore')
            results_to_save.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Save selected features to a separate file for interpretation
            if 'Selected_Features' in results_df.columns:
                features_file = f"./datasets/twostep_lasso_selected_features_{group_col.lower()}{param_suffix}.pkl"
                features_data = {row[group_col]: row['Selected_Features'] for _, row in results_df.iterrows()}
                pd.to_pickle(features_data, features_file)
                print(f"Selected features saved to: {features_file}")
            
            # Create and save accuracy plot
            plot_path = f"./datasets/{group_col.lower()}_accuracy_distribution{param_suffix}_twostep_lasso.png"
            plot_accuracy_distribution(
                results_df,
                group_col,
                group_col,
                plot_path,
                params,
                overall_metrics=overall_metrics
            )
            
            # Create and save sparsity plot (Lasso specific)
            sparsity_path = f"./datasets/{group_col.lower()}_sparsity_distribution{param_suffix}_twostep.png"
            plot_sparsity_distribution(
                results_df,
                group_col,
                group_col,
                sparsity_path,
                params,
                overall_metrics=overall_metrics
            )

def plot_accuracy_distribution(results_df, group_col, title, save_path, sig_params, overall_metrics=None):
    """
    Plot accuracy distribution with accuracy values on top of bars
    
    Parameters:
    results_df: DataFrame containing results
    group_col: Group column name ('Sector' or 'Industry')
    title: Chart title
    save_path: Path to save the plot
    sig_params: Dictionary containing signature parameters (w, o, d)
    overall_metrics: Dictionary containing overall evaluation metrics
    """
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    bars = plt.bar(results_df[group_col], results_df['Direction_Accuracy'])
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Add symbol counts at the bottom of bars
    for i, row in results_df.iterrows():
        plt.text(row[group_col], 0.01,  # Position slightly above bar bottom
                f'n={row["Symbol_Count"]}',
                ha='center', va='bottom',
                rotation=0)
    
    # Add horizontal line for overall accuracy
    if overall_metrics is not None:
        overall_acc = overall_metrics['Direction_Accuracy']
    else:
        overall_acc = results_df['Direction_Accuracy'].mean()
    plt.axhline(y=overall_acc, color='r', linestyle='--', label=f'Overall: {overall_acc:.3f}')
    
    # Add signature parameters text
    param_text = f'w={sig_params["w"]}, o={sig_params["o"]}, d={sig_params["d"]}'
    if sig_params.get('weighted', False):
        param_text += f'\nweighted (γ={sig_params["gamma"]})'
        
    plt.text(0.98, 0.98, 
            param_text,
            transform=plt.gca().transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Customize plot
    plt.title(f'Two-step Lasso - Accuracy Distribution by {title}', fontsize=14, pad=20)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel('Accuracy Rate', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")

def plot_sparsity_distribution(results_df, group_col, title, save_path, sig_params, overall_metrics=None):
    """
    Plot sparsity distribution for Two-step Lasso models
    
    Parameters:
    results_df: DataFrame containing results
    group_col: Group column name
    title: Chart title
    save_path: Path to save the plot
    sig_params: Dictionary containing signature parameters (w, o, d)
    overall_metrics: Dictionary containing overall evaluation metrics
    """
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Sort by sparsity (descending)
    results_df_sorted = results_df.sort_values('Sparsity', ascending=False)
    
    # Create bar plot
    bars = plt.bar(results_df_sorted[group_col], results_df_sorted['Sparsity'])
    
    # Add sparsity values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Add non-zero coefficient counts at the bottom of bars
    for i, row in results_df_sorted.iterrows():
        plt.text(row[group_col], 0.01,  # Position slightly above bar bottom
                f'{int(row["Non_Zero_Coeffs"])}/{int(row["Total_Coeffs"])}',
                ha='center', va='bottom',
                rotation=90)
    
    # Add average sparsity line
    avg_sparsity = results_df['Sparsity'].mean()
    plt.axhline(y=avg_sparsity, color='r', linestyle='--', label=f'Avg Sparsity: {avg_sparsity:.3f}')
    
    # Add signature parameters text
    param_text = f'w={sig_params["w"]}, o={sig_params["o"]}, d={sig_params["d"]}'
    if sig_params.get('weighted', False):
        param_text += f'\nweighted (γ={sig_params["gamma"]})'
        
    plt.text(0.98, 0.98, 
            param_text,
            transform=plt.gca().transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Customize plot
    plt.title(f'Two-step Lasso - Sparsity Distribution by {title}', fontsize=14, pad=20)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel('Sparsity (% of Zero Coefficients)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main() 