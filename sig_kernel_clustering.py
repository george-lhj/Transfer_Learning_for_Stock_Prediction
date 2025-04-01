import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from tqdm import tqdm
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
time_window = 10
order = 3
input_prefix = "sig_data_SP500"
input_path = f"./datasets/{input_prefix}_w{time_window}_o{order}.parquet"

# Fixed number of clusters
optimal_k = 10

# Output paths
clustered_data_path = "./datasets/df_train_wclusters.parquet"
model_path = "./datasets/kmeans_final.joblib"
stock_list_path = "./datasets/all_stock_ids.joblib"
centroids_path = "./datasets/cluster_centroids.joblib"
distance_matrix_path = "avg_distance_32.npy"

################################################################################
#                            LOAD DATA
################################################################################
print(f"Loading signature data from: {input_path}")
df_train = pd.read_parquet(input_path)

if 'symbol' in df_train.columns:
    df_train.rename(columns={'symbol': 'STOCK'}, inplace=True)
if 'date' in df_train.columns:
    df_train.rename(columns={'date': 'DATE'}, inplace=True)

df_train["STOCK"] = df_train["STOCK"].astype(str)
df_train["DATE"] = pd.to_datetime(df_train["DATE"])
all_stock_ids = sorted(df_train["STOCK"].unique())

################################################################################
#                      KERNEL AND DISTANCE FUNCTIONS
################################################################################
def compute_kernel(df):
    feature_cols = [c for c in df.columns if c.startswith("SIG_")]
    features = df[feature_cols].values.astype(np.float32)
    kernel = np.einsum('ij,kj->ik', features, features)
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
#                    EMBED DISTANCE MATRIX → VECTOR SPACE
################################################################################
print("Embedding distance matrix into vector space using MDS...")
mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42)
mds_embedding = mds.fit_transform(avg_distance_32)

################################################################################
#                         FINAL K-MEANS CLUSTERING
################################################################################
print(f"Running final KMeans clustering (k={optimal_k}) on MDS embedding...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(mds_embedding)
labels = kmeans_final.labels_

# Save mappings
cluster_centroids = kmeans_final.cluster_centers_
stock_to_cluster = dict(zip(all_stock_ids, labels))
df_train["CLUSTER"] = df_train["STOCK"].map(stock_to_cluster)

# Optional: print cluster sizes
cluster_sizes = pd.Series(labels).value_counts().sort_index()
print("Cluster sizes:")
print(cluster_sizes)

################################################################################
#                         SAVE OUTPUTS
################################################################################
df_train.to_parquet(clustered_data_path, engine='pyarrow', compression='snappy')
print(f"✅ Clustered data saved to: {clustered_data_path}")

joblib.dump(kmeans_final, model_path)
print(f"✅ Saved KMeans model to: {model_path}")

joblib.dump(all_stock_ids, stock_list_path)
print(f"✅ Saved stock list to: {stock_list_path}")

joblib.dump(cluster_centroids, centroids_path)
print(f"✅ Saved centroids to: {centroids_path}")

np.save(distance_matrix_path, avg_distance_32)
print(f"✅ Saved avg_distance_32.npy for visualization at: {distance_matrix_path}")
