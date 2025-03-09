import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(42)

# Load datasets and the trained model
print("Loading datasets and trained model...")
df_train = pd.read_parquet("./datasets/df_train_wdate.parquet")
df_test = pd.read_parquet("./datasets/df_test_wdate.parquet")

kmeans_final = joblib.load("./datasets/kmeans_final.joblib")
all_stock_ids = joblib.load("./datasets/all_stock_ids.joblib")

# Extract SIG feature columns
feature_cols = [c for c in df_train.columns if c.startswith("SIG_")]

# Compute average training features grouped by stock ID
train_features_df = df_train.groupby('STOCK')[feature_cols].mean().loc[all_stock_ids]
train_features = train_features_df.values.astype(np.float32)

# Compute diagonal of train kernel (||x||^2)
diag_train = np.einsum('ij,ij->i', train_features, train_features)

# Function to compute test-to-train distance matrix per day
def compute_kernel_test_train(df_test_day, train_features):
    test_features_day = df_test_day[feature_cols].values.astype(np.float32)
    kernel = np.einsum('ij,kj->ik', test_features_day, train_features)
    diag_test_day = np.einsum('ij,ij->i', test_features_day, test_features_day)
    dist_test_train_day = np.sqrt(np.abs(diag_test_day[:, None] - 2 * kernel + diag_train[None, :]))
    return pd.DataFrame(dist_test_train_day, index=df_test_day['STOCK'], columns=all_stock_ids)

# Compute daily distance matrices (test vs. train)
print("Computing distance matrices (test vs. train) for each test date...")
results_test = Parallel(n_jobs=-1)(
    delayed(compute_kernel_test_train)(group, train_features)
    for _, group in tqdm(df_test.groupby('DATE'), desc="Processing Test Dates")
)

# Initialize matrices for accumulating distances and counts
distance_sum_test = pd.DataFrame(0.0, index=sorted(df_test['STOCK'].unique()), columns=all_stock_ids, dtype=np.float32)
counts_test = pd.DataFrame(0, index=distance_sum_test.index, columns=distance_sum_test.columns, dtype=np.int32)

# Accumulate distances across dates
print("Accumulating distances...")
for dist_df in tqdm(results_test, desc="Accumulating distances"):
    dist_reindexed = dist_df.reindex(index=distance_sum_test.index, columns=all_stock_ids)
    mask = ~dist_reindexed.isna()
    distance_sum_test[mask] += dist_reindexed[mask].astype(np.float32)
    counts_test[mask] += 1

# Compute average distance matrix (test stocks vs train stocks)
print("Calculating average test-to-train distances...")
avg_dist_test_to_train = distance_sum_test / counts_test.replace(0, np.nan)
avg_dist_test_to_train.fillna(float(avg_dist_test_to_train.mean().mean()), inplace=True)
avg_dist_test_to_train_32 = avg_dist_test_to_train.values.astype(np.float32)

# Predict clusters using the averaged distance matrix
print("Predicting clusters...")
labels_test = kmeans_final.predict(avg_dist_test_to_train_32)

# Map predicted cluster labels back to test dataset
stock_to_cluster_test = dict(zip(distance_sum_test.index, labels_test))
df_test["CLUSTER"] = df_test["STOCK"].map(stock_to_cluster_test)

# Save test dataset with cluster labels
df_test.to_parquet("./datasets/df_test_wdate_wclusters.parquet", engine='pyarrow', compression='snappy')
print("✅ Updated test dataset saved with averaged distances.")
