import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

# --- Step 1: Read CSV ---
sig = pd.read_csv("df_train_wdate.csv")

# Use the 'stock' column to get the union of stocks (assumed to be ~5k stocks)
all_stock_ids = sorted(sig["STOCK"].unique())

# --- Step 2: Define helper functions ---
def compute_kernel(df):
    cols = [c for c in df.columns if c.startswith("SIG_")]
    sig_data = df[cols].values.astype(np.float32)
    kernel = np.dot(sig_data, sig_data.T)
    return pd.DataFrame(kernel, index=df["STOCK"], columns=df["STOCK"])

def compute_signature_distance(kernel_df):
    k = kernel_df.values.astype(np.float32)
    diag = np.diag(k)
    dist = diag[:, None] - 2 * k + diag[None, :]
    return pd.DataFrame(dist, index=kernel_df.index, columns=kernel_df.columns)

# --- Step 3: Process each date and accumulate distances ---
sum_distance_df = pd.DataFrame(0.0, index=all_stock_ids, columns=all_stock_ids)
count_df = pd.DataFrame(0, index=all_stock_ids, columns=all_stock_ids)

dates = sorted(sig["DATE"].unique())
for d in tqdm(dates, desc="Processing Dates"):
    sig_d = sig[sig["DATE"] == d]
    kernel_d = compute_kernel(sig_d)
    distance_d = compute_signature_distance(kernel_d)
    # Reindex to full set of stocks (filling missing with NaN)
    distance_d_reindexed = distance_d.reindex(index=all_stock_ids, columns=all_stock_ids)
    sum_distance_df = sum_distance_df.add(distance_d_reindexed.fillna(0))
    count_df = count_df.add(distance_d_reindexed.notna().astype(int))

# --- Step 4: Compute the average distance matrix ---
avg_distance_df = sum_distance_df / count_df
avg_distance_df = avg_distance_df.fillna(0)

# --- Step 5: Convert the distance matrix to an affinity matrix ---
# One common approach is to use a Gaussian kernel:
#   affinity[i,j] = exp(-gamma * distance[i,j])
# Here we choose gamma = 1 / sigma, where sigma is the median of nonzero distances.
nonzero_dist = avg_distance_df.values[avg_distance_df.values > 0]
sigma = np.median(nonzero_dist) if nonzero_dist.size > 0 else 1.0
gamma = 1 / sigma
affinity_matrix = np.exp(-avg_distance_df.values * gamma)

# --- Step 6: Cluster the stocks into 10 groups using Spectral Clustering ---
sc = SpectralClustering(n_clusters=10, affinity="precomputed", random_state=42)
labels = sc.fit_predict(affinity_matrix)

# Map each stock to its cluster label
clusters_df = pd.DataFrame({"stock": avg_distance_df.index, "cluster": labels})

# Save clusters to CSV
clusters_df.to_csv("stock_clusters_10_groups.csv", index=False)
print("Stock clusters saved to 'stock_clusters_10_groups.csv'")
