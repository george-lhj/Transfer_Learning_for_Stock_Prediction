import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from calc_signature import *
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        # Add direction accuracy (whether prediction direction is correct)
        'Direction_Accuracy': np.mean((y_true * y_pred) > 0)
    }
    return metrics

def ridge_regression_by_group_adaptive_weight(df_train, df_test, group_col='Sector'):
    """
    Perform ridge regression by specified column,
    incorporating adaptive weights computed for each stock.
    """
    # Set parameter grid
    param_grid = {"alpha": np.logspace(-3, 3, num=10)}
    
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
            
            if len(df_group_train) < 10 or len(df_group_test) < 5:
                print(f"Warning: {group_col} {group} has insufficient samples, skipping")
                continue
            
            # Prepare features and labels
            X_train = df_group_train[feature_cols].values
            y_train = df_group_train['RET'].values
            X_test = df_group_test[feature_cols].values
            y_test = df_group_test['RET'].values
            
            # Get the adaptive weights for training samples
            weights_train = df_group_train['adaptive_weight'].values
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Grid search with sample weights passed to fit
            gs = GridSearchCV(
                estimator=Ridge(),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Fit the model with sample_weight
            gs.fit(X_train_scaled, y_train, sample_weight=weights_train)
            
            # Predict on test data
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            
            # Evaluate predictions
            metrics = evaluate_predictions(y_test, y_pred)
            
            # Save results
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            train_ret_stats = df_group_train['RET'].describe()
            test_ret_stats = df_group_test['RET'].describe()
            
            result_dict = {
                group_col: group,
                "Train_Samples": len(df_group_train),
                "Test_Samples": len(df_group_test),
                "Best_Alpha": gs.best_params_["alpha"],
                "MSE": metrics['MSE'],
                "RMSE": metrics['RMSE'],
                "MAE": metrics['MAE'],
                "R2": metrics['R2'],
                "Direction_Accuracy": metrics['Direction_Accuracy'],
                "Train_Return_Mean": train_ret_stats['mean'],
                "Train_Return_Std": train_ret_stats['std'],
                "Test_Return_Mean": test_ret_stats['mean'],
                "Test_Return_Std": test_ret_stats['std'],
                "Model": best_model
            }
            
            if group_col == 'Industry':
                result_dict["Sector"] = df_group_train['Sector'].iloc[0]
                
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing {group_col} {group}: {str(e)}")
            continue
    
    return pd.DataFrame(results), y_true_all, y_pred_all

def ridge_regression_by_group(df_train, df_test, group_col='Sector'):
    """
    Perform ridge regression by specified column
    
    Parameters:
    df_train: DataFrame, training set
    df_test: DataFrame, testing set
    group_col: str, grouping column name, e.g., 'Sector' or 'Industry'
    
    Returns:
    results: DataFrame, containing results for each group
    y_true_all: list, all true values
    y_pred_all: list, all predicted values
    """
    # Set parameter grid
    param_grid = {"alpha": np.logspace(-3, 3, num=10)}
    
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
            
            # Grid search
            gs = GridSearchCV(
                estimator=Ridge(),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',  # Use MSE as evaluation metric
                n_jobs=-1
            )
            
            # Train model
            gs.fit(X_train_scaled, y_train)
            
            # Predict
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            
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
                "Best_Alpha": gs.best_params_["alpha"],
                "MSE": metrics['MSE'],
                "RMSE": metrics['RMSE'],
                "MAE": metrics['MAE'],
                "R2": metrics['R2'],
                "Direction_Accuracy": metrics['Direction_Accuracy'],
                "Train_Return_Mean": train_ret_stats['mean'],
                "Train_Return_Std": train_ret_stats['std'],
                "Test_Return_Mean": test_ret_stats['mean'],
                "Test_Return_Std": test_ret_stats['std'],
                "Model": best_model
            }
            
            # Add Sector information if grouping by Industry
            if group_col == 'Industry':
                result_dict["Sector"] = df_group_train['Sector'].iloc[0]
                
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing {group_col} {group}: {str(e)}")
            continue
    
    return pd.DataFrame(results), y_true_all, y_pred_all


def main():
    # Load data
    print("Loading data...")
    df = pd.read_parquet("./datasets/sig_data_SP500_w10_o3.parquet")
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Split dataset by time
    print("\nSplitting dataset by time...")
    df_train, df_test = train_test_split_by_time(df, train_ratio=0.7)
    
    # Define grouping columns to test
    group_columns = ['Sector', 'Industry']
    
    # Test each grouping column
    for group_col in group_columns:
        print(f"\nStarting model training by {group_col}...")
        results_df, y_true_all, y_pred_all = ridge_regression_by_group(df_train, df_test, group_col)
        
        # Output results
        print(f"\nTraining results for each {group_col}:")
        # Ensure selected columns exist in results DataFrame
        available_cols = [col for col in results_df.columns 
                        if col in [group_col, 'Sector', 'Train_Samples', 'Test_Samples', 
                                 'Best_Alpha', 'MSE', 'RMSE', 'MAE', 'R2', 
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
        
        # Save results
        output_file = f"./datasets/ridge_regression_results_{group_col.lower()}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()