import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
from joblib import Parallel, delayed
import multiprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

np.random.seed(42)

def prepare_historical_features(df, window_size, stock_id, id_column='symbol', 
                               return_col='Return', trate_col='Transaction_Rate'):
    """
    Prepare historical features for a specific stock
    
    Parameters:
    df: DataFrame, containing date, stock and return data
    window_size: int, historical window size
    stock_id: str, stock code
    id_column: str, stock identifier column name
    return_col: str, return rate column name
    trate_col: str, transaction rate column name
    
    Returns:
    DataFrame containing historical features
    """
    # Get data for this stock
    stock_data = df[df[id_column] == stock_id].sort_values('date')
    
    if len(stock_data) <= window_size:
        return None
    
    feature_list = []
    
    # Create historical features for each day
    for i in range(window_size, len(stock_data)):
        try:
            current_date = stock_data.iloc[i]['date']
            current_row = stock_data.iloc[i].copy()
            
            # Extract historical data from previous window_size days
            historical_data = stock_data.iloc[i-window_size:i]
            
            # Check if there is enough non-NaN data
            if historical_data[return_col].isna().sum() > window_size * 0.3:  # If more than 30% of data is NaN
                continue
                
            # Create features: historical returns
            for lag in range(1, window_size + 1):
                value = historical_data.iloc[-lag][return_col]
                current_row[f'RET_LAG_{lag}'] = value if not pd.isna(value) else 0
            
            # Create features: historical transaction rates
            if trate_col in historical_data.columns:
                for lag in range(1, window_size + 1):
                    value = historical_data.iloc[-lag][trate_col]
                    current_row[f'TRATE_LAG_{lag}'] = value if not pd.isna(value) else 0
                
            # Add some statistical features (using dropna() to ignore NaN values)
            current_row['RET_MEAN'] = historical_data[return_col].dropna().mean()
            current_row['RET_STD'] = historical_data[return_col].dropna().std()
            
            if trate_col in historical_data.columns:
                current_row['TRATE_MEAN'] = historical_data[trate_col].dropna().mean()
                current_row['TRATE_STD'] = historical_data[trate_col].dropna().std()
            
            # Check for NaNs and fill with 0
            for col in current_row.index:
                if pd.isna(current_row[col]) and col not in [id_column, 'date', return_col, trate_col, 'Ticker', 'Sector', 'Industry']:
                    current_row[col] = 0
            
            feature_list.append(current_row)
        except Exception as e:
            print(f"Error processing {stock_id} on date {current_date}: {str(e)}")
            continue
    
    if len(feature_list) == 0:
        return None
        
    return pd.DataFrame(feature_list)

def train_test_split_by_time(df, train_ratio=0.7, return_col='Return'):
    """
    Split dataset into training and testing sets by time order
    
    Parameters:
    df: DataFrame, containing date and feature data
    train_ratio: float, proportion of training set
    return_col: str, return rate column name
    
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
    
    # Show return statistics
    print(f"\nTraining set return statistics:")
    print(df.iloc[:split_idx][return_col].describe())
    print(f"\nTest set return statistics:")
    print(df.iloc[split_idx:][return_col].describe())
    
    return df.iloc[:split_idx], df.iloc[split_idx:]

def evaluate_predictions(y_true, y_pred):
    """
    Calculate multiple regression evaluation metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate predicted and actual directions
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
        'True_Positive_Rate': np.mean(pred_direction[true_direction]),  # Ratio of correctly predicted increases
        'True_Negative_Rate': np.mean(~pred_direction[~true_direction]),  # Ratio of correctly predicted decreases
    }
    return metrics

def benchmark_regression_by_stock(df_train, df_test, id_column='symbol', return_col='Return'):
    """
    Execute benchmark linear regression for each stock
    
    Parameters:
    df_train: DataFrame, training set
    df_test: DataFrame, test set
    id_column: str, stock identifier column name
    return_col: str, return rate column name
    
    Returns:
    results: DataFrame, containing results for each stock
    y_true_all: list, all actual values
    y_pred_all: list, all predicted values
    """
    results = []
    y_true_all = []
    y_pred_all = []
    
    # Get all stock IDs
    all_stocks = sorted(df_train[id_column].unique())
    
    # Train model for each stock
    for stock in tqdm(all_stocks, desc="Processing stocks"):
        try:
            # Get current stock data
            df_stock_train = df_train[df_train[id_column] == stock]
            df_stock_test = df_test[df_test[id_column] == stock]
            
            if len(df_stock_train) < 10 or len(df_stock_test) < 5:
                print(f"Warning: Stock {stock} has insufficient samples, skipping")
                continue
            
            # Prepare features and labels
            # Get all feature columns (RET_LAG_* and TRATE_LAG_* columns)
            feature_cols = [col for col in df_stock_train.columns 
                           if col.startswith('RET_LAG_') or col.startswith('TRATE_LAG_')
                           or col in ['RET_MEAN', 'RET_STD', 'TRATE_MEAN', 'TRATE_STD']]
            
            X_train = df_stock_train[feature_cols].values
            y_train = df_stock_train[return_col].values
            X_test = df_stock_test[feature_cols].values
            y_test = df_stock_test[return_col].values
            
            # Check for NaN values and handle them
            if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
                # If there are NaN values, fill with mean
                imputer = SimpleImputer(strategy='mean')
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
                
                # Handle NaNs in target variable
                if np.isnan(y_train).any():
                    y_imputer = SimpleImputer(strategy='mean')
                    y_train = y_imputer.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                if np.isnan(y_test).any():
                    # To avoid leakage, we use training set mean to fill missing values in test set
                    if not np.isnan(y_train).any():  # If training set has been filled
                        mean_value = np.mean(y_train)
                        y_test = np.nan_to_num(y_test, nan=mean_value)
                    else:
                        y_imputer = SimpleImputer(strategy='mean')
                        y_test = y_imputer.fit_transform(y_test.reshape(-1, 1)).ravel()
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train simple linear regression model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Add direction prediction column
            df_stock_test_with_pred = df_stock_test.copy()
            df_stock_test_with_pred['PRED_RET'] = y_pred
            df_stock_test_with_pred['PRED_DIR'] = (y_pred > 0).astype(int)
            df_stock_test_with_pred['TRUE_DIR'] = (y_test > 0).astype(int)
            
            # Evaluate predictions
            metrics = evaluate_predictions(y_test, y_pred)
            
            # Save results
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            # Calculate return statistics
            train_ret_stats = pd.Series(y_train).describe()
            test_ret_stats = pd.Series(y_test).describe()
            
            # Get group information (if it exists)
            ticker = df_stock_train['Ticker'].iloc[0] if 'Ticker' in df_stock_train.columns else None
            sector = df_stock_train['Sector'].iloc[0] if 'Sector' in df_stock_train.columns else None
            industry = df_stock_train['Industry'].iloc[0] if 'Industry' in df_stock_train.columns else None
            
            result_dict = {
                id_column: stock,
                "Ticker": ticker,
                "Train_Samples": len(df_stock_train),
                "Test_Samples": len(df_stock_test),
                "MSE": metrics['MSE'],
                "RMSE": metrics['RMSE'],
                "MAE": metrics['MAE'],
                "R2": metrics['R2'],
                "Direction_Accuracy": metrics['Direction_Accuracy'],
                "Train_Return_Mean": train_ret_stats['mean'],
                "Train_Return_Std": train_ret_stats['std'],
                "Test_Return_Mean": test_ret_stats['mean'],
                "Test_Return_Std": test_ret_stats['std']
            }
            
            # Add group information (if it exists)
            if sector is not None:
                result_dict["Sector"] = sector
            if industry is not None:
                result_dict["Industry"] = industry
                
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing stock {stock}: {str(e)}")
            continue
    
    return pd.DataFrame(results), y_true_all, y_pred_all

def linear_regression_by_group(df_train, df_test, group_col='Sector', return_col='Return'):
    """
    Perform linear regression by group (Sector or Industry)
    
    Parameters:
    df_train: DataFrame, training set
    df_test: DataFrame, test set
    group_col: str, grouping column name ('Sector' or 'Industry')
    return_col: str, return rate column name
    
    Returns:
    results: DataFrame, containing results for each group
    y_true_all: list, all true values
    y_pred_all: list, all predicted values
    """
    results = []
    y_true_all = []
    y_pred_all = []
    
    # Get all feature columns
    feature_cols = [col for col in df_train.columns 
                   if col.startswith('RET_LAG_') or col.startswith('TRATE_LAG_')
                   or col in ['RET_MEAN', 'RET_STD', 'TRATE_MEAN', 'TRATE_STD']]
    
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
            y_train = df_group_train[return_col].values
            X_test = df_group_test[feature_cols].values
            y_test = df_group_test[return_col].values
            
            # Check for NaN values and handle them
            if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
                # If there are NaN values, fill with mean
                imputer = SimpleImputer(strategy='mean')
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
                
                # Handle NaNs in target variable
                if np.isnan(y_train).any():
                    y_imputer = SimpleImputer(strategy='mean')
                    y_train = y_imputer.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                if np.isnan(y_test).any():
                    # To avoid leakage, we use training set mean to fill missing values in test set
                    if not np.isnan(y_train).any():  # If training set has been filled
                        mean_value = np.mean(y_train)
                        y_test = np.nan_to_num(y_test, nan=mean_value)
                    else:
                        y_imputer = SimpleImputer(strategy='mean')
                        y_test = y_imputer.fit_transform(y_test.reshape(-1, 1)).ravel()
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train simple linear regression model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate predictions
            metrics = evaluate_predictions(y_test, y_pred)
            
            # Save results
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            # Calculate return statistics
            train_ret_stats = pd.Series(y_train).describe()
            test_ret_stats = pd.Series(y_test).describe()
            
            # Count unique symbols in this group
            symbol_count = len(df_group_train['symbol'].unique())
            
            result_dict = {
                group_col: group,
                "Train_Samples": len(df_group_train),
                "Test_Samples": len(df_group_test),
                "Symbol_Count": symbol_count,
                "MSE": metrics['MSE'],
                "RMSE": metrics['RMSE'],
                "MAE": metrics['MAE'],
                "R2": metrics['R2'],
                "Direction_Accuracy": metrics['Direction_Accuracy'],
                "Train_Return_Mean": train_ret_stats['mean'],
                "Train_Return_Std": train_ret_stats['std'],
                "Test_Return_Mean": test_ret_stats['mean'],
                "Test_Return_Std": test_ret_stats['std']
            }
            
            # Add sector info if grouping by industry
            if group_col == 'Industry' and 'Sector' in df_group_train.columns:
                # Find the most common sector for this industry
                result_dict['Sector'] = df_group_train['Sector'].mode()[0]
                
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing {group_col} {group}: {str(e)}")
            continue
    
    return pd.DataFrame(results), y_true_all, y_pred_all

def process_stock(stock, df, window_size, id_column, return_col, trate_col):
    """Wrapper function designed for parallel processing"""
    try:
        features = prepare_historical_features(df, window_size, stock, id_column, return_col, trate_col)
        return features
    except Exception as e:
        print(f"Error processing stock {stock} features: {str(e)}")
        return None

def main():
    # Define window size
    window_size = 50
    
    # Feature file path
    features_file = f"./datasets/features_df_w{window_size}.parquet"
    
    # Check if saved feature file exists
    if os.path.exists(features_file):
        print(f"Found saved feature file: {features_file}, loading directly...")
        df_with_features = pd.read_parquet(features_file)
        print(f"Successfully loaded feature data, shape: {df_with_features.shape}")
    else:
        # If feature file doesn't exist, perform complete feature processing
        print(f"Feature file not found, processing feature data...")
        
        # Load original data
        input_file = "sp500_final_data_5y.csv"
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(f"./datasets/{input_file}")
        
        # Map column names
        return_col = 'Return'  # Return rate column
        trate_col = 'Transaction_Rate'  # Transaction rate column
        id_column = 'symbol'  # Stock identifier column
        
        print("Data columns:", df.columns.tolist())
        
        # Data preprocessing according to calc_signature.py
        print("Preprocessing data...")
        
        # 1. Keep only needed columns
        keep_columns = [id_column, 'date', 'volume', trate_col, return_col, 'Sector', 'Industry', 'Ticker']
        available_columns = [col for col in keep_columns if col in df.columns]
        df = df[available_columns]
        
        # 2. Convert date to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # 3. Handle missing values - drop rows with NaN
        original_rows = len(df)
        df = df.dropna()
        dropped_rows = original_rows - len(df)
        print(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/original_rows:.2%})")
        
        # 4. Optional: Apply logarithmic transformation to volume
        if 'volume' in df.columns:
            df['volume'] = np.log1p(df['volume'])
            
        # 5. Optional: Apply logarithmic transformation to return rate (here we don't do log transformation, as linear regression directly predicts original returns)
        # df['Return_original'] = df[return_col]
        # df[return_col] = np.log1p(df[return_col])
        
        # Prepare feature data
        print("Preparing historical features...")
        all_stocks = df[id_column].unique()
        print(f"Total of {len(all_stocks)} stocks")
        
        # Determine CPU count for parallel processing
        n_jobs = multiprocessing.cpu_count() - 1  # Reserve one core for system
        print(f"Using {n_jobs} CPUs for parallel processing")
        
        # Use parallel processing to accelerate feature generation
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_stock)(stock, df, window_size, id_column, return_col, trate_col) 
            for stock in tqdm(all_stocks, desc="Parallel processing stock features")
        )
        
        # Filter out non-None results
        processed_data = [result for result in results if result is not None]
        
        if not processed_data:
            print("No processable data. Please check data format and window size.")
            return
        
        # Combine all processed data into one DataFrame
        df_with_features = pd.concat(processed_data, ignore_index=True)
        
        # Save processed feature DataFrame
        print(f"Saving feature data to: {features_file}")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(features_file), exist_ok=True)
        df_with_features.to_parquet(features_file, engine='pyarrow', compression='snappy')
        print(f"Feature data saved, shape: {df_with_features.shape}")
    
    # Split dataset by time
    print("\nSplitting dataset by time...")
    return_col = 'Return'  # Ensure this variable exists in the file reading branch
    id_column = 'symbol'   # Ensure this variable exists in the file reading branch
    df_train, df_test = train_test_split_by_time(df_with_features, train_ratio=0.7, return_col=return_col)
    
    # Run benchmark regression per stock
    print("\nRunning benchmark linear regression per stock...")
    results_df, y_true_all, y_pred_all = benchmark_regression_by_stock(df_train, df_test, 
                                                                     id_column=id_column,
                                                                     return_col=return_col)
    
    if len(results_df) == 0:
        print("No results generated. Skipping...")
        return
    
    # Output results for per-stock models
    print("\nTraining results for each stock:")
    display_cols = [id_column, 'Ticker', 'Train_Samples', 'Test_Samples', 
                  'MSE', 'RMSE', 'MAE', 'R2', 'Direction_Accuracy']
    
    # If group information exists, add to display columns
    for col in ['Sector', 'Industry']:
        if col in results_df.columns:
            display_cols.insert(2, col)
    
    print(results_df[display_cols])
    
    # Calculate overall evaluation metrics for per-stock models
    overall_metrics = evaluate_predictions(y_true_all, y_pred_all)
    print(f"\nOverall evaluation metrics (per-stock models):")
    for metric_name, metric_value in overall_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save per-stock results
    output_file = f"./datasets/linear_regression_results_per_stock_w{window_size}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nPer-stock results saved to: {output_file}")
    
    # Simple average by sector/industry (original method)
    for group_col in ['Sector', 'Industry']:
        if group_col in results_df.columns:
            try:
                # Calculate accuracy for each group
                group_accuracy = results_df.groupby(group_col)['Direction_Accuracy'].agg(['mean', 'count'])
                group_accuracy.columns = ['Accuracy', 'Count']
                group_accuracy = group_accuracy.sort_values('Accuracy', ascending=False)
                
                print(f"\nAverage accuracy by {group_col} (simple average):")
                print(group_accuracy)
                
                # Create and save accuracy distribution chart
                plot_accuracy_distribution(
                    group_accuracy.reset_index(),
                    group_col,
                    group_col,
                    f'./datasets/linear_{group_col.lower()}_accuracy_simple_avg_w{window_size}.png',
                    window_size,
                    overall_metrics=overall_metrics
                )
            except Exception as e:
                print(f"Error when grouping by {group_col}: {str(e)}")
                continue
    
    # Run linear regression by group (new method)
    for group_col in ['Sector', 'Industry']:
        if group_col in df_train.columns and group_col in df_test.columns:
            try:
                print(f"\nRunning linear regression by {group_col}...")
                group_results_df, group_y_true_all, group_y_pred_all = linear_regression_by_group(
                    df_train, df_test, group_col=group_col, return_col=return_col
                )
                
                if len(group_results_df) == 0:
                    print(f"No results generated for {group_col} regression. Skipping...")
                    continue
                
                # Output results for group models
                print(f"\nTraining results for each {group_col}:")
                group_display_cols = [group_col, 'Train_Samples', 'Test_Samples', 'Symbol_Count',
                                    'MSE', 'RMSE', 'MAE', 'R2', 'Direction_Accuracy']
                
                # If Sector column exists when grouping by Industry, add it to display
                if group_col == 'Industry' and 'Sector' in group_results_df.columns:
                    group_display_cols.insert(1, 'Sector')
                
                print(group_results_df[group_display_cols])
                
                # Calculate overall evaluation metrics for group models
                group_overall_metrics = evaluate_predictions(group_y_true_all, group_y_pred_all)
                print(f"\nOverall evaluation metrics (by {group_col}):")
                for metric_name, metric_value in group_overall_metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")
                
                # Save group results
                group_output_file = f"./datasets/linear_regression_results_by_{group_col.lower()}_w{window_size}.csv"
                group_results_df.to_csv(group_output_file, index=False)
                print(f"\nResults by {group_col} saved to: {group_output_file}")
                
                # Create and save accuracy distribution chart for group models
                plot_accuracy_distribution(
                    group_results_df,
                    group_col,
                    group_col,
                    f'./datasets/linear_{group_col.lower()}_accuracy_group_model_w{window_size}.png',
                    window_size,
                    overall_metrics=group_overall_metrics
                )
                
                # Compare with simple average approach
                print(f"\nComparison of approaches for {group_col}:")
                print(f"Simple average approach - Overall Accuracy: {overall_metrics['Direction_Accuracy']:.4f}")
                print(f"Group model approach - Overall Accuracy: {group_overall_metrics['Direction_Accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error when running regression by {group_col}: {str(e)}")
                continue

def plot_accuracy_distribution(results_df, group_col, title, save_path, window_size, overall_metrics=None):
    """
    Plot accuracy distribution and display accuracy values at the top of the bar chart
    
    Parameters:
    results_df: DataFrame containing results
    group_col: Group column name
    title: Chart title
    save_path: Save path
    window_size: Window size
    overall_metrics: Dictionary containing overall evaluation metrics
    """
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar chart
    bars = plt.bar(results_df[group_col].astype(str), results_df['Accuracy'])
    
    # Add accuracy values at the top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Add sample count at the bottom of bars
    for i, row in results_df.iterrows():
        plt.text(i, 0.01,  # Position slightly above bottom of bar
                f'n={row["Count"]}',
                ha='center', va='bottom',
                rotation=0)
    
    # Add horizontal line for overall accuracy
    if overall_metrics is not None:
        overall_acc = overall_metrics['Direction_Accuracy']
    else:
        overall_acc = results_df['Accuracy'].mean()
    plt.axhline(y=overall_acc, color='r', linestyle='--', label=f'Overall: {overall_acc:.3f}')
    
    # Add window size text
    param_text = f'Historical Window = {window_size} days'
    plt.text(0.98, 0.98, 
            param_text,
            transform=plt.gca().transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Customize chart
    plt.title(f'Linear Regression - Accuracy Distribution by {title}', fontsize=14, pad=20)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label truncation
    plt.tight_layout()
    
    # Save chart
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {save_path}")

if __name__ == "__main__":
    main() 