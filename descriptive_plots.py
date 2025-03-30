import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_label_distribution(train_path, test_path, label_column='SECTOR'):
    """
    Analyze the distribution of a specified label in training and test datasets
    using both pie charts and bar charts.
    
    Parameters:
    train_path (str): Path to training set parquet file
    test_path (str): Path to test set parquet file
    label_column (str): The column name of the classification label to analyze
    """
    # Load data
    print(f"Loading datasets and analyzing {label_column} distribution...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    # Verify the label column exists in both datasets
    if label_column not in df_train.columns or label_column not in df_test.columns:
        raise ValueError(f"Label column '{label_column}' not found in one or both datasets")
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("pastel")
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ----- 1. Training Set Label Proportion Pie Chart -----
    print(f"Analyzing training set {label_column} distribution...")
    label_counts_train = df_train[label_column].value_counts().sort_index()
    label_props_train = label_counts_train / len(df_train) * 100
    
    wedges, texts, autotexts = axes[0, 0].pie(
        label_props_train, 
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Add legend
    axes[0, 0].legend(
        wedges, 
        [f"{label_column} {s}: {p:.1f}%" for s, p in zip(label_props_train.index, label_props_train)],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    axes[0, 0].set_title(f'Training Set {label_column} Distribution (Pie Chart)', fontsize=14)
    
    # ----- 2. Test Set Label Proportion Pie Chart -----
    print(f"Analyzing test set {label_column} distribution...")
    label_counts_test = df_test[label_column].value_counts().sort_index()
    label_props_test = label_counts_test / len(df_test) * 100
    
    wedges, texts, autotexts = axes[0, 1].pie(
        label_props_test, 
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Add legend
    axes[0, 1].legend(
        wedges, 
        [f"{label_column} {s}: {p:.1f}%" for s, p in zip(label_props_test.index, label_props_test)],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    axes[0, 1].set_title(f'Test Set {label_column} Distribution (Pie Chart)', fontsize=14)
    
    # ----- 3. Training Set Label Count Bar Chart -----
    print(f"Creating training set {label_column} bar chart...")
    
    sns.barplot(x=label_counts_train.index, y=label_counts_train.values, ax=axes[1, 0])
    
    # Add count labels
    for i, v in enumerate(label_counts_train.values):
        axes[1, 0].text(i, v + max(label_counts_train.values)*0.01, str(v), ha='center')
    
    axes[1, 0].set_title(f'Training Set {label_column} Sample Counts', fontsize=14)
    axes[1, 0].set_xlabel(f'{label_column} ID')
    axes[1, 0].set_ylabel('Sample Count')
    
    # ----- 4. Test Set Label Count Bar Chart -----
    print(f"Creating test set {label_column} bar chart...")
    
    sns.barplot(x=label_counts_test.index, y=label_counts_test.values, ax=axes[1, 1])
    
    # Add count labels
    for i, v in enumerate(label_counts_test.values):
        axes[1, 1].text(i, v + max(label_counts_test.values)*0.01, str(v), ha='center')
    
    axes[1, 1].set_title(f'Test Set {label_column} Sample Counts', fontsize=14)
    axes[1, 1].set_xlabel(f'{label_column} ID')
    axes[1, 1].set_ylabel('Sample Count')
    
    # Adjust layout and save
    plt.tight_layout()
    output_filename = f'{label_column.lower()}_analysis.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis chart saved as '{output_filename}'")
    
    # Print statistical summary
    print(f"\n----- {label_column} Distribution Summary -----")
    print(f"Training set size: {len(df_train)} samples")
    print(f"Test set size: {len(df_test)} samples")
    print(f"Number of unique {label_column} values in training set: {len(label_counts_train)}")
    print(f"Number of unique {label_column} values in test set: {len(label_counts_test)}")
    
    # Check for missing values in test set
    missing_in_test = set(label_counts_train.index) - set(label_counts_test.index)
    if missing_in_test:
        print(f"\nWARNING: {len(missing_in_test)} {label_column} values found in training set but missing in test set:")
        print(missing_in_test)


# Example usage - default to SECTOR analysis
if __name__ == "__main__":
    analyze_label_distribution(
        "./datasets/df_train_wdate_wclusters.parquet",
        "./datasets/df_test_wdate_wclusters.parquet",
        label_column="CLUSTER"  # Default label to analyze
    )