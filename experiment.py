import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

df_train = pd.read_parquet("./datasets/df_train_wdate_wclusters.parquet")
df_test = pd.read_parquet("./datasets/df_test_wdate_wclusters.parquet")

industry_lst = df_train.INDUSTRY.unique() # 'INDUSTRY'
industry_group_lst = df_train.INDUSTRY_GROUP.unique() # 'INDUSTRY_GROUP'
sector_lst = df_train.SECTOR.unique() # 'SECTOR'
sub_industry_lst = df_train.SUB_INDUSTRY.unique() # 'SUB_INDUSTRY'
stock_lst = df_train.STOCK.unique() # 'STOCK'
cluster_lst = df_train.CLUSTER.unique() # 'CLUSTER'

param_grid = {
    "C": np.logspace(-3, 3, num=10)  # e.g., 0.001 to 1000
}

results = []
y_true_all = []
y_pred_all = []

# -----------------------------------------------------------------
# 2) TRAIN AND EVALUATE PER CLUSTER
# -----------------------------------------------------------------
for cluster_idx in tqdm(cluster_lst, total=len(cluster_lst), desc="Processing clusters"):
    try:
        # Split train/test data for this cluster
        df_cluster_train = df_train[df_train["CLUSTER"] == cluster_idx]
        df_cluster_test  = df_test[df_test["CLUSTER"] == cluster_idx]

        # Skip cluster if no data in train or test
        if len(df_cluster_train) == 0 or len(df_cluster_test) == 0:
            print(f"Skipping cluster {cluster_idx} - no data.")
            continue

        # Identify feature columns (those starting with "SIG_")
        feature_cols = [c for c in df_cluster_train.columns if c.startswith("SIG_")]

        # Extract X and y for training and testing
        X_train = df_cluster_train[feature_cols].values
        y_train = df_cluster_train["RET"].values  # assume labels are 0/1
        X_test = df_cluster_test[feature_cols].values
        y_test = df_cluster_test["RET"].values

        # Skip cluster if X or y is empty (shouldn't happen, but safety check)
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"Skipping cluster {cluster_idx} - empty X or y.")
            continue
        print(1)
        # Standardize features: fit on training data then transform both sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Use GridSearchCV to tune LogisticRegression with L2 penalty (ridge logistic regression)
        gs = GridSearchCV(
            estimator=LogisticRegression(
                penalty="l2", 
                solver="saga",     # 更快的求解器
                max_iter=100,      # 增加单次迭代
                tol=1e-2,          # 提高容忍度
                warm_start=True,   # 使用上次结果作为起点
                random_state=42    # 保证结果一致性
            ),
            param_grid=param_grid,
            cv=3,                 # 减少交叉验证折数
            scoring="accuracy",
            n_jobs=-1             # 使用所有CPU核心
        )
        print(2)
        gs.fit(X_train_scaled, y_train)
        print(3)
        # Extract the best hyperparameter and model
        best_C = gs.best_params_["C"]
        best_model = gs.best_estimator_

        y_prob = best_model.predict_proba(X_test_scaled)

        # Get predicted probabilities for class 1 from the best model
        y_val_proba = y_prob[:, 1]

        # --- Custom Threshold: Compute the median of the predicted probabilities ---
        custom_threshold = np.median(y_val_proba)
        # Create custom predictions: assign 1 if probability >= median, else 0
        y_val_pred_custom = (y_val_proba >= custom_threshold).astype(int)

        # Get default predictions using the model's built-in threshold (0.5)
        y_val_pred_default = best_model.predict(X_test_scaled)

        # Evaluate default predictions
        acc_default = accuracy_score(y_test, y_val_pred_default)
        report_default = classification_report(y_test, y_val_pred_default, zero_division=0)

        # Evaluate custom predictions (using median threshold)
        acc_custom = accuracy_score(y_test, y_val_pred_custom)
        report_custom = classification_report(y_test, y_val_pred_custom, zero_division=0)

        # Aggregate default predictions for overall evaluation
        y_true_all.extend(y_test)
        y_pred_all.extend(y_val_pred_default)

        # Save the results for this cluster
        results.append({
            "Cluster": cluster_idx,
            "Train_Samples": len(df_cluster_train),
            "Test_Samples": len(df_cluster_test),
            "Best_C": best_C,
            "Default_Accuracy": acc_default,
            "Custom_Accuracy": acc_custom,
            "Default_Report": report_default,
            "Custom_Report": report_custom,
            "Custom_Threshold": custom_threshold,
            "Model": best_model
        })

    except Exception as e:
        print(f"Error processing cluster {cluster_idx}: {e}")
        continue

# -----------------------------------------------------------------
# 3) SUMMARIZE RESULTS
# -----------------------------------------------------------------
df_results = pd.DataFrame(results)
print(df_results)

overall_accuracy = accuracy_score(y_true_all, y_pred_all)
print(f"\nOverall Default Accuracy across all clusters: {overall_accuracy:.4f}")

# Optionally, you can inspect the classification reports:
# for idx, row in df_results.iterrows():
#     print(f"\nCluster {row['Cluster']} - Default Report:")
#     print(row['Default_Report'])
#     print(f"Cluster {row['Cluster']} - Custom Report:")
#     print(row['Custom_Report'])