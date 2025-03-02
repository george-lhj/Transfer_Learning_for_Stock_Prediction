import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from calc_signature import *


x_train = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/x_train.csv")
y_train = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/y_train.csv")
x_test = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/x_test.csv")
y_test = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/test_rand.csv")

# df_train = calc_signature(x_train, y_train, order=3)
# df_test = calc_signature(x_test, y_test, order=3)

# df_train.to_csv("/Users/runminghuang/Desktop/capstone/Data/df_train.csv", index=False)
# df_test.to_csv("/Users/runminghuang/Desktop/capstone/Data/df_test.csv", index=False)

df_train = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/df_train.csv")
df_test = pd.read_csv("/Users/runminghuang/Desktop/capstone/Data/df_test.csv")

industry_lst = df_train.INDUSTRY.unique() # 'INDUSTRY'
industry_group_lst = df_train.INDUSTRY_GROUP.unique() # 'INDUSTRY_GROUP'
sector_lst = df_train.SECTOR.unique() # 'SECTOR'
sub_industry_lst = df_train.SUB_INDUSTRY.unique() # 'SUB_INDUSTRY'

param_grid = {"alpha": np.logspace(-3, 3, num=10)}
results = []

for sector_idx in sector_lst:
    try:
        # Clean memory
        import gc
        gc.collect()
        df_sector_train = df_train[df_train["SECTOR"] == sector_idx]
        df_sector_test = df_test[df_test["SECTOR"] == sector_idx]
        # Build feature matrix
        feature_cols = [c for c in df_sector_train.columns if c.startswith("SIG_")]
        
        X_train = df_sector_train[feature_cols].values
        y_train = np.array(df_sector_train["RET"])
        X_test = df_sector_test[feature_cols].values
        y_test = np.array(df_sector_test["RET"])
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  
        X_test_scaled = scaler.transform(X_test)

        # Train the model on the training set
        # Grid search (5-fold CV)
        gs = GridSearchCV(
            estimator=RidgeClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            # scoring="f1",
            n_jobs=1
        )
        gs.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        best_alpha = gs.best_params_["alpha"]
        best_model = gs.best_estimator_
        y_val_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_val_pred)
        report = classification_report(y_test, y_val_pred)

        # Store results
        results.append({
            "Grouped_by": sector_idx,
            "Train_Samples": len(df_sector_train),
            "Val_Samples": len(df_sector_test),
            "Best_Alpha": best_alpha,
            "Val_Accuracy": acc,
            "Report": report,
            "Model": best_model
        })

    except Exception as e:
        print(f"Error processing sector {sector_idx}: {str(e)}")
        continue

# Summarize
df_results = pd.DataFrame(results)
print(df_results[[
    "Grouped_by", "Train_Samples", "Val_Samples",
    "Best_Alpha", "Val_Accuracy"
]])