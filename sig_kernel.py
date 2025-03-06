import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

sig = pd.read_csv("df_train_wdate.csv")

# Get the unique stock IDs
all_stock_ids = sorted(sig["STOCK"].unique())

# Compute kernel and distance functions
def compute_kernel(df):
    cols = [c for c in df.columns if c.startswith("SIG_")]
    features = df[cols].values.astype(np.float32)
    kernel = features @ features.T
    return pd.DataFrame(kernel, index=df["STOCK"], columns=df["STOCK"])

def compute_signature_distance(kernel_df):
    k = kernel_df.values
    diag = np.diag(k)
    dist = diag[:, None] - 2 * k + diag[None, :]
    return pd.DataFrame(dist, index=kernel_df.index, columns=kernel_df.columns)

# Initialize distance accumulator
distance_sum = pd.DataFrame(0.0, index=all_stock_ids, columns=all_stock_ids)
counts = pd.DataFrame(0, index=all_stock_ids, columns=all_stock_ids)

# Accumulate distances across all dates
for date, group in tqdm(sig.groupby('DATE'), desc="Processing Dates"):
    kernel_df = compute_kernel(group)
    dist_df = compute_signature_distance(kernel_df)
    dist_reindexed = dist_df.reindex(index=all_stock_ids, columns=all_stock_ids)
    mask = ~dist_reindexed.isna()
    distance_sum[mask] += dist_reindexed[mask]
    counts[mask] += 1

# Calculate average distances
avg_distance = distance_sum / counts
avg_distance = avg_distance.fillna(avg_distance.max().max())

# Elbow method to determine best k
inertia_values = []
k_range = range(2, 21)
for k in tqdm(k_range, desc="Finding optimal k"):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(avg_distance)
    inertia_values.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.grid(True)
plt.xticks(k_range)
plt.savefig('elbow_method.png')
plt.show()

# Choose optimal k (manually identified from elbow graph, e.g., 10)
optimal_k = 10
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
labels = kmeans_final.fit_predict(avg_distance)

# Save cluster results
clusters_df = pd.DataFrame({"stock": all_stock_ids, "cluster": labels})
clusters_df.to_csv("stock_clusters_kmeans.csv", index=False)
avg_distance.to_csv("avg_distance_matrix_between_stocks.csv", index=True)
