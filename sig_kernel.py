import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from kneed import KneeLocator
from joblib import Parallel, delayed
import joblib
import random

np.random.seed(42)
random.seed(42)

################################################################################
#                                TRAINING
################################################################################
print("Loading dataset...")
df_train = pd.read_parquet("./datasets/df_train_wdate.parquet")

# Unique training stock IDs
all_stock_ids = sorted(df_train["STOCK"].unique())

# Compute kernel & distance
def compute_kernel(df):
    feature_cols = [c for c in df.columns if c.startswith("SIG_")]
    features = df[feature_cols].values.astype(np.float32)
    kernel = np.einsum('ij,kj->ik', features, features)  # float32 kernel
    return pd.DataFrame(kernel, index=df["STOCK"], columns=df["STOCK"])

def compute_signature_distance(kernel_df):
    k = kernel_df.values
    diag = np.diag(k)
    dist = np.abs(diag[:, None] - 2 * k + diag[None, :])
    return pd.DataFrame(dist, index=kernel_df.index, columns=kernel_df.columns)

# Initialize distance accumulation
print("Initializing distance accumulation...")
distance_sum = pd.DataFrame(0.0, index=all_stock_ids, columns=all_stock_ids, dtype=np.float32)
counts = pd.DataFrame(0, index=all_stock_ids, columns=all_stock_ids, dtype=np.int32)

# Compute distance matrices for each date
print("Computing distance matrices for each date...")
results = Parallel(n_jobs=-1)(
    delayed(lambda g: compute_signature_distance(compute_kernel(g)))(group)
    for _, group in tqdm(df_train.groupby('DATE'), desc="Processing Dates")
)

# Accumulate results
print("Accumulating distance matrices...")
for dist_df in tqdm(results, desc="Accumulating distances"):
    dist_reindexed = dist_df.reindex(index=all_stock_ids, columns=all_stock_ids)
    mask = ~dist_reindexed.isna()
    distance_sum[mask] += dist_reindexed[mask].astype(np.float32)
    counts[mask] += 1

# Compute average distance
print("Calculating average distances...")
avg_distance = distance_sum / counts.replace(0, np.nan)
avg_distance.fillna(float(avg_distance.mean().mean()), inplace=True)  # Fill NaNs with mean
avg_distance_32 = avg_distance.values.astype(np.float32)

# Elbow method to determine optimal k
print("Finding optimal k using the elbow method...")
k_range = range(2, 21)
inertia_values = []
for k in tqdm(k_range, desc="Evaluating K-Means"):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(avg_distance_32)
    inertia_values.append(kmeans_temp.inertia_)

knee_locator = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
optimal_k = knee_locator.elbow
print(f"Optimal number of clusters found: k={optimal_k}")

# Plot elbow method
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

# Final K-Means clustering
print(f"Performing final K-Means clustering with k={optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(avg_distance_32)  # Train on float32 array
labels = kmeans_final.labels_
print("Clustering completed.")

# Store centroids
cluster_centroids = kmeans_final.cluster_centers_

# Assign clusters to training set
stock_to_cluster = dict(zip(all_stock_ids, labels))
df_train["CLUSTER"] = df_train["STOCK"].map(stock_to_cluster)
df_train.to_parquet("./datasets/df_train_wdate_wclusters.parquet", engine='pyarrow', compression='snappy')
print("Updated dataset saved to 'df_train_wdate_wclusters.parquet'.")
print("✅ Training completed.")

joblib.dump(kmeans_final, "./datasets/kmeans_final.joblib")
print("✅ KMeans model saved to './datasets/kmeans_final.joblib'")

joblib.dump(all_stock_ids, "./datasets/all_stock_ids.joblib")
print("✅ Saved all_stock_ids.")

joblib.dump(cluster_centroids, "./datasets/cluster_centroids.joblib") 
print("✅ Saved cluster_centroids.")
