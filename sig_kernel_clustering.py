import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from kneed import KneeLocator
from joblib import Parallel, delayed
import joblib
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

################################################################################
#                          CONFIGURATION
################################################################################
time_window = 10   # Match your calc_signature_new setting
order = 3          # Signature order used in calc_signature_new
input_prefix = "sig_data_SP500"
input_path = f"./datasets/{input_prefix}_w{time_window}_o{order}.parquet"

# Output paths
clustered_data_path = "./datasets/df_train_wclusters.parquet"
model_path = "./datasets/kmeans_final.joblib"
stock_list_path = "./datasets/all_stock_ids.joblib"
centroids_path = "./datasets/cluster_centroids.joblib"

################################################################################
#                            LOAD DATA
################################################################################
print(f"Loading signature data from: {input_path}")
df_train = pd.read_parquet(input_path)

# Rename to expected format
if 'symbol' in df_train.columns:
    df_train.rename(columns={'symbol': 'STOCK'}, inplace=True)
if 'date' in df_train.columns:
    df_train.rename(columns={'date': 'DATE'}, inplace=True)

# Make sure stock/date are strings for consistent indexing
df_train["STOCK"] = df_train["STOCK"].astype(str)
df_train["DATE"] = pd.to_datetime(df_train["DATE"])

# Get stock list
all_stock_ids = sorted(df_train["STOCK"].unique())

################################################################################
#                      KERNEL AND DISTANCE FUNCTIONS
################################################################################
def compute_kernel(df):
    feature_cols = [c for c in df.columns if c.startswith("SIG_")]
    features = df[feature_cols].values.astype(np.float32)
    kernel = np.einsum('ij,kj->ik', features, features)  # dot product
    return pd.DataFrame(kernel, index=df["STOCK"], columns=df["STOCK"])

def compute_signature_distance(kernel_df):
    k = kernel_df.values
    diag = np.diag(k)
    dist = np.abs(diag[:, None] - 2 * k + diag[None, :])
    return pd.DataFrame(dist, index=kernel_df.index, columns=kernel_df.columns)

################################################################################
#                  DISTANCE MATRIX ACCUMULATION (PER DATE)
################################################################################
print("Initializing accumulation of pairwise distance...")
distance_sum = pd.DataFrame(0.0, index=all_stock_ids, columns=all_stock_ids, dtype=np.float32)
counts = pd.DataFrame(0, index=all_stock_ids, columns=all_stock_ids, dtype=np.int32)

def process_group(group):
    # Remove duplicates before computing
    group = group.drop_duplicates(subset="STOCK", keep="first")
    kernel = compute_kernel(group)
    return compute_signature_distance(kernel)

print("Processing per-date distance matrices...")
results = Parallel(n_jobs=-1)(
    delayed(process_group)(group)
    for _, group in tqdm(df_train.groupby('DATE'), desc="Processing Dates")
)

print("Accumulating average pairwise distances...")
for dist_df in tqdm(results, desc="Accumulating Distances"):
    dist_df = dist_df[~dist_df.index.duplicated(keep='first')]
    dist_df = dist_df.loc[:, ~dist_df.columns.duplicated(keep='first')]
    dist_reindexed = dist_df.reindex(index=all_stock_ids, columns=all_stock_ids)
    
    mask = ~dist_reindexed.isna()
    distance_sum[mask] += dist_reindexed[mask].astype(np.float32)
    counts[mask] += 1

avg_distance = distance_sum / counts.replace(0, np.nan)
avg_distance.fillna(float(avg_distance.mean().mean()), inplace=True)
avg_distance_32 = avg_distance.values.astype(np.float32)

################################################################################
#                     ELBOW METHOD TO FIND OPTIMAL k
################################################################################
print("Using elbow method to find optimal k...")
k_range = range(2, 21)
inertia_values = []

for k in tqdm(k_range, desc="Fitting K-Means"):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(avg_distance_32)
    inertia_values.append(kmeans_temp.inertia_)

knee_locator = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
optimal_k = knee_locator.elbow or 5  # Fallback to 5 if elbow not found
print(f"Optimal number of clusters: k = {optimal_k}")

# Plot elbow
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, 'o-', label="Inertia")
plt.axvline(optimal_k, color='red', linestyle='--', label=f"Optimal k={optimal_k}")
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.legend()
plt.grid(True)
plt.xticks(list(k_range))
plt.savefig('elbow_method.png')
plt.show()

################################################################################
#                         FINAL K-MEANS CLUSTERING
################################################################################
print(f"Running final KMeans clustering (k={optimal_k})...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(avg_distance_32)
labels = kmeans_final.labels_

# Save mappings
cluster_centroids = kmeans_final.cluster_centers_
stock_to_cluster = dict(zip(all_stock_ids, labels))

df_train["CLUSTER"] = df_train["STOCK"].map(stock_to_cluster)

# Save final dataset
df_train.to_parquet(clustered_data_path, engine='pyarrow', compression='snappy')
print(f"✅ Clustered data saved to: {clustered_data_path}")

# Save model and meta
joblib.dump(kmeans_final, model_path)
print(f"✅ Saved KMeans model to: {model_path}")

joblib.dump(all_stock_ids, stock_list_path)
print(f"✅ Saved stock list to: {stock_list_path}")

joblib.dump(cluster_centroids, centroids_path)
print(f"✅ Saved centroids to: {centroids_path}")
