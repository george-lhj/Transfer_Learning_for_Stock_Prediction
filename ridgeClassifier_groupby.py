import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from calc_signature import *
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
warnings.filterwarnings('ignore')

def preprocess_returns(df):
    """
    Convert returns to binary classification labels
    1: Positive return (Up)
    0: Negative return (Down)
    """
    # Save original return value
    df['RET_original'] = df['RET']
    # Convert to binary classification labels
    df['RET'] = (df['RET'] > 0).astype(int)
    return df

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
    
    # Check class distribution
    print("\nTraining set class distribution:")
    print(df.iloc[:split_idx]['RET'].value_counts(normalize=True))
    print("\nTesting set class distribution:")
    print(df.iloc[split_idx:]['RET'].value_counts(normalize=True))
    
    return df.iloc[:split_idx], df.iloc[split_idx:]

def ridge_regression_by_group(df_train, df_test, group_col='Sector'):
    """
    Perform ridge classification by specified column
    
    Parameters:
    df_train: DataFrame, training set
    df_test: DataFrame, testing set
    group_col: str, grouping column name, e.g., 'Sector' or 'Industry'
    
    Returns:
    results: DataFrame, containing results for each group
    y_true_all: list, all true labels
    y_pred_all: list, all predicted labels
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
            
            # Check class distribution
            train_class_dist = df_group_train['RET'].value_counts(normalize=True)
            test_class_dist = df_group_test['RET'].value_counts(normalize=True)
            
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
                estimator=RidgeClassifier(),
                param_grid=param_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1
            )
            
            # Train model
            gs.fit(X_train_scaled, y_train)
            
            # Predict
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            
            # Calculate accuracy
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            
            # Save results
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            result_dict = {
                group_col: group,
                "Train_Samples": len(df_group_train),
                "Test_Samples": len(df_group_test),
                "Best_Alpha": gs.best_params_["alpha"],
                "Test_Accuracy_Score": acc,
                "Train_Class_Distribution": dict(train_class_dist),
                "Test_Class_Distribution": dict(test_class_dist),
                "Report": report,
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

def plot_accuracy_distribution(results_df, group_col, title, save_path):
    """
    Plot accuracy distribution histogram with accuracy values on top of bars
    
    Parameters:
    results_df: DataFrame containing results
    group_col: Group column name ('Sector' or 'Industry')
    title: Chart title
    save_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    bars = plt.bar(results_df[group_col], results_df['Test_Accuracy_Score'])
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',  # Only show the overall accuracy
                ha='center', va='bottom')
    
    # Add horizontal line for mean accuracy
    mean_acc = results_df['Test_Accuracy_Score'].mean()
    plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Mean: {mean_acc:.3f}')
    
    # Customize plot
    plt.title(f'Accuracy Distribution by {title}', fontsize=14, pad=20)
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

def plot_accuracy_comparison():
    """
    Plot accuracy distribution comparison for Sector and Industry
    """
    # Read the saved CSV files
    sector_results = pd.read_csv('./datasets/ridge_classification_results_sector.csv')
    industry_results = pd.read_csv('./datasets/ridge_classification_results_industry.csv')
    
    # Ensure datasets folder exists
    os.makedirs('./datasets', exist_ok=True)
    
    # Plot for Sector
    plot_accuracy_distribution(
        sector_results,
        'Sector',
        'Sector',
        './datasets/sector_accuracy_distribution.png'
    )
    
    # Plot for Industry
    plot_accuracy_distribution(
        industry_results,
        'Industry',
        'Industry',
        './datasets/industry_accuracy_distribution.png'
    )

def main():
    # Load data
    print("Loading data...")
    df = pd.read_parquet("./datasets/sig_data_SP500_w10_o3.parquet")
    
    # Preprocess: convert returns to binary classification labels
    print("Converting returns to binary classification labels...")
    df = preprocess_returns(df)
    
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
        display_cols = [group_col, "Train_Samples", "Test_Samples", 
                       "Best_Alpha", "Test_Accuracy_Score",
                       "Train_Class_Distribution", "Test_Class_Distribution"]
        
        # Select display columns based on grouping type
        if group_col == 'Industry' and 'Sector' in results_df.columns:
            display_cols = ['Industry', 'Sector'] + [col for col in display_cols 
                          if col not in ['Industry', 'Sector']]
        
        # Print results in a nice format
        print(results_df[display_cols].to_string())
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        print(f"\n{group_col} overall accuracy: {overall_accuracy:.4f}")
        print(f"\n{group_col} overall classification report:")
        print(classification_report(y_true_all, y_pred_all))
        
        # Save results
        output_file = f"./datasets/ridge_classification_results_{group_col.lower()}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # After saving all results, call the plotting function
    plot_accuracy_comparison()
    
    # Calculate statistics for each group
    print("\nCalculating statistics for each group...")
    sector_stats = df.groupby('Sector')['symbol'].nunique().reset_index()
    industry_stats = df.groupby('Industry')['symbol'].nunique().reset_index()
    
    # Calculate basic statistics
    basic_stats = pd.DataFrame({
        'Metric': ['Total_Sectors', 'Total_Industries', 'Avg_Stocks_per_Sector', 
                  'Avg_Stocks_per_Industry', 'Max_Stocks_in_Sector', 'Max_Stocks_in_Industry'],
        'Value': [
            len(sector_stats),
            len(industry_stats),
            sector_stats['symbol'].mean(),
            industry_stats['symbol'].mean(),
            sector_stats['symbol'].max(),
            industry_stats['symbol'].max()
        ]
    })
    
    # Save CSV files
    sector_stats.to_csv('./datasets/sector_stock_counts.csv', index=False)
    industry_stats.to_csv('./datasets/industry_stock_counts.csv', index=False)
    basic_stats.to_csv('./datasets/basic_statistics.csv', index=False)
    
    print("Statistics and plots have been saved to the datasets folder")

if __name__ == "__main__":
    main()