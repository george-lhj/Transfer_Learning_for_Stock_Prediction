import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import lightgbm as lgb

def main():
    """Main execution flow"""
    # Data loading
    X_train = np.load("/Users/runminghuang/Desktop/capstone/Data/X_train.npz")["X"]
    feature_names = np.load("/Users/runminghuang/Desktop/capstone/Data/X_train.npz")["feature_names"]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    y_train_df = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/y_train.csv")

    # Extract IDs before removal
    train_ids = X_train_df["ID"].astype(int)

    # Data preprocessing
    non_feature_cols = ["ID", "Stock", "Industry", "Industry_Group", 
                       "Sub_Industry", "Sector", "Start Time", "End Time"]
    X_train_df = X_train_df.drop(columns=[c for c in non_feature_cols if c in X_train_df.columns], errors="ignore")

    # Target variable processing
    y_train_df["RET"] = (y_train_df["RET"] > 0).astype(int)
    y_dict = dict(zip(y_train_df["ID"], y_train_df["RET"]))
    y = np.array([y_dict.get(stock_id, 0) for stock_id in train_ids])

    # Data splitting with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_df, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model configuration
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,  # No limit
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,     # Use all CPU cores
        force_col_wise=True  # Required for v4.0+
    )
    
    # Model training with callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_error",
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(20)
        ]
    )

    # Model evaluation
    y_pred = model.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # Feature importance visualization
    plt.figure(figsize=(12, 8))
    plt.barh(X_train.columns[np.argsort(model.feature_importances_)[-20:]], 
             np.sort(model.feature_importances_)[-20:])
    plt.title("Top 20 Important Features (LightGBM)")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("lgbm_feature_importance.png")
    plt.close()

if __name__ == "__main__":
    main() 