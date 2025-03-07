import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from kneed import KneeLocator
from joblib import Parallel, delayed
import random

np.random.seed(40)
random.seed(40)
################################################################################
#                                TRAINING
################################################################################
print("Loading dataset...")
df_train = pd.read_parquet("./datasets/df_train_wdate.parquet")

# Unique training stock IDs
all_stock_ids = sorted(df_train["STOCK"].unique())

# Kernel & distance computations
def compute_kernel(df):
    cols = [c for c in df.columns if c.startswith("SIG_")]
    features = df[cols].values.astype(np.float32)
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

# Parallel function for each date
def process_date(group):
    kernel_df = compute_kernel(group)
    dist_df = compute_signature_distance(kernel_df)
    return dist_df

print("Computing distance matrices for each date...")
results = Parallel(n_jobs=-1)(
    delayed(process_date)(group) for _, group in tqdm(df_train.groupby('DATE'), desc="Processing Dates")
)

# Accumulate results
print("Accumulating distance matrices...")
for dist_df in tqdm(results, desc="Accumulating distances"):
    dist_reindexed = dist_df.reindex(index=all_stock_ids, columns=all_stock_ids)
    mask = ~dist_reindexed.isna()
    # Convert slice to float32 here to ensure consistency
    distance_sum[mask] += dist_reindexed[mask].astype(np.float32)
    counts[mask] += 1

# Compute average distance
print("Calculating average distances...")
avg_distance = distance_sum / counts.replace(0, np.nan)  # Avoid division by 0

# Replace any remaining NaNs with the mean
mean_val = float(avg_distance.mean().mean())  # float() helps avoid potential Series dtypes
avg_distance.fillna(mean_val, inplace=True)

# Ensure avg_distance is float32
avg_distance_32 = avg_distance.values.astype(np.float32)

# Elbow method
print("Finding optimal k using the elbow method...")
k_range = range(2, 21)
inertia_values = []
for k in tqdm(k_range, desc="Evaluating K-Means"):
    kmeans_temp = KMeans(n_clusters=k, random_state=40, n_init=10)
    kmeans_temp.fit(avg_distance_32)
    inertia_values.append(kmeans_temp.inertia_)

knee_locator = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
optimal_k = knee_locator.elbow
print(f"Optimal number of clusters found: k={optimal_k}")

# Plot elbow
print("Generating elbow method plot...")
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
print("Elbow method plot saved as 'elbow_method.png'")

# Final K-Means
print(f"Performing final K-Means clustering with k={optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=40, n_init=10)
kmeans_final.fit(avg_distance_32)  # Train on float32 array
labels = kmeans_final.labels_
print("Clustering completed.")

# (Optional) Save cluster results
# clusters_df = pd.DataFrame({"stock": all_stock_ids, "cluster": labels})
# clusters_df.to_csv("stock_clusters_kmeans.csv", index=False)
# print("Stock clusters saved to 'stock_clusters_kmeans.csv'.")

# (Optional) Save average distance matrix
# pd.DataFrame(avg_distance_32, index=all_stock_ids, columns=all_stock_ids).to_csv(
#     "avg_distance_matrix_between_stocks.csv", index=True)
# print("Average distance matrix saved to 'avg_distance_matrix_between_stocks.csv'.")

# (Optional) Store centroids in float32
# centroids_32 = kmeans_final.cluster_centers_.astype(np.float32)
# np.save("kmeans_centroids.npy", centroids_32)
# print("Centroids saved as 'kmeans_centroids.npy'.")

# Map clusters to training set
print("Adding cluster assignments to original dataset...")
stock_to_cluster = dict(zip(all_stock_ids, labels))
df_train["CLUSTER"] = df_train["STOCK"].map(stock_to_cluster)
df_train.to_parquet("./datasets/df_train_wdate_wclusters.parquet", engine='pyarrow', compression='snappy')
print("Updated dataset saved to 'df_train_wdate_wclusters.parquet'.")


################################################################################
#                                TESTING
################################################################################
print("Loading test dataset...")
df_test = pd.read_parquet("./datasets/df_test_wdate.parquet")

# Get unique stock IDs in test
test_stock_ids = sorted(df_test["STOCK"].unique())

# Compute kernel for unique test stocks
print("Computing kernel for test dataset...")
df_test_unique = df_test.drop_duplicates(subset=["STOCK"])
kernel_test = compute_kernel(df_test_unique)

# Compute distance matrix for test stocks
print("Computing distance matrix for test dataset...")
dist_test = compute_signature_distance(kernel_test)

# Reindex to match training shape
dist_test_reindexed = dist_test.reindex(index=all_stock_ids, columns=all_stock_ids)

# Fill NaNs with mean
dist_test_mean = float(dist_test_reindexed.mean().mean())
dist_test_reindexed.fillna(dist_test_mean, inplace=True)

# Convert to float32
dist_test_array_32 = dist_test_reindexed.values.astype(np.float32)

# Predict clusters on test data
print("Assigning test data to clusters...")
labels_test = kmeans_final.predict(dist_test_array_32)

# Map cluster labels back to full test data
test_stock_to_cluster = dict(zip(test_stock_ids, labels_test))
df_test["CLUSTER"] = df_test["STOCK"].map(test_stock_to_cluster)

# Save updated test set
df_test.to_parquet("./datasets/df_test_wdate_wclusters.parquet", engine='pyarrow', compression='snappy')
print("Updated test dataset saved to 'df_test_wdate_wclusters.parquet'.")

print("\n Complete!")