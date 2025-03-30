import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from calc_signature import *
from tqdm import tqdm

if __name__ == "__main__":
    x_train = pd.read_csv("./datasets/x_train.csv")
    y_train = pd.read_csv("./datasets/y_train.csv")
    x_test = pd.read_csv("./datasets/x_test.csv")
    y_test = pd.read_csv("./datasets/test_rand.csv")


    # # RIDGE REGRESSION BY SECTOR
    loaded_train = np.load("./datasets/df_train_signature_data.npz")
    loaded_test = np.load("./datasets/df_test_signature_data.npz")

    df_train_final = pd.DataFrame(data=loaded_train['data'], columns=loaded_train['features'])
    df_test_final = pd.DataFrame(data=loaded_test['data'], columns=loaded_test['features'])

    df_train_new_features = pd.read_parquet("./datasets/df_train_new_features.parquet")
    df_test_new_features = pd.read_parquet("./datasets/df_test_new_features.parquet")

    y_dict_train_RET = y_train.set_index('ID')['RET'].to_dict()
    y_dict_test_RET = y_test.set_index('ID')['RET'].to_dict()

    df_train_final['RET'] = df_train_final['ID'].map(y_dict_train_RET)
    df_test_final['RET'] = df_test_final['ID'].map(y_dict_test_RET)


    industry_lst = df_train_final.INDUSTRY.unique() # 'INDUSTRY'
    industry_group_lst = df_train_final.INDUSTRY_GROUP.unique() # 'INDUSTRY_GROUP'
    sector_lst = df_train_final.SECTOR.unique() # 'SECTOR'
    sub_industry_lst = df_train_final.SUB_INDUSTRY.unique() # 'SUB_INDUSTRY'
    stock_lst = df_train_final.STOCK.unique() # 'STOCK'

    param_grid = {"alpha": np.logspace(-3, 3, num=10)}

    results = []
    y_true_all = []
    y_pred_all = []

    for sub_industry_idx in tqdm(sub_industry_lst, total=len(sub_industry_lst), desc="Processing training and prediction"):
        try:
            df_sector_train = df_train_final[df_train_final["SUB_INDUSTRY"] == sub_industry_idx]
            df_sector_test = df_test_final[df_test_final["SUB_INDUSTRY"] == sub_industry_idx]
            # Build feature matrix
            # feature_cols = [c for c in df_sector_train.columns if c.startswith("SIG_")] + new_features
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
                # scoring="roc_auc",
                # scoring="recall",
                # scoring="f1",
                n_jobs=1
            )
            gs.fit(X_train_scaled, y_train)

            # gs = RandomizedSearchCV(
            #     estimator=LogisticRegression(penalty='elasticnet', solver='saga', max_iter=500, tol=1e-2, warm_start=True),
            #     param_distributions=param_grid,
            #     cv=5,
            #     scoring="f1",
            #     n_jobs=1,
            #     n_iter=20,
            # )
            gs.fit(X_train_scaled, y_train)
            # Evaluate on validation set
            best_alpha = gs.best_params_["alpha"]
            # best_alpha = gs.best_params_["C"], gs.best_params_["l1_ratio"]
            best_model = gs.best_estimator_
            y_val_pred = best_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_val_pred)
            report = classification_report(y_test, y_val_pred, zero_division=0)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_val_pred)
            # Store results
            results.append({
                "Grouped_by": sub_industry_idx,
                "Train_Samples": len(df_sector_train),
                "Val_Samples": len(df_sector_test),
                "Best_Alpha": best_alpha,
                "Val_Accuracy": acc,
                "Report": report,
                "Model": best_model
            })

        except Exception as e:
            print(f"Error processing sector {sub_industry_idx}: {str(e)}")
            continue

    # Summarize
    df_results = pd.DataFrame(results)
    print(df_results[[
        "Grouped_by", "Train_Samples", "Val_Samples",
        "Best_Alpha", "Val_Accuracy"
    ]])

    overall_accuracy = accuracy_score(y_true_all, y_pred_all)
    print(f"\nOverall Accuracy across all sectors: {overall_accuracy:.4f}")
    # df_results.to_csv("./datasets/ridge_regression_by_SECTOR_20250303.csv", index=False)