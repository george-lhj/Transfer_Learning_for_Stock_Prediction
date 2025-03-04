import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

    # # Generate df_train and df_test and save them as parquet files
    new_features_train, df_train_new_features, df_train_sig_prepared = prepare_df_for_signature_computation(x_train, y_train, save=True, filename="df_train")
    # new_features_test, df_test_new_features, df_test_sig_prepared = prepare_df_for_signature_computation(x_test, y_test, save=True, filename="df_test")

    # # Calculate signature for df_train and df_test and save them as npz files
    df_train_sig_prepared = pd.read_parquet("./datasets/df_train.parquet")
    # df_test_sig_prepared = pd.read_parquet("./datasets/df_test.parquet")
    with open("./datasets/df_train_new_features.txt", "r") as f:
        new_features = eval(f.read())

    calc_signature(df_train_sig_prepared, order=3, new_features=new_features, filename="df_train", save=True)
    # calc_signature(df_test_sig_prepared, order=3, new_features=new_features, filename="df_test", save=True)

    # # RIDGE REGRESSION BY SECTOR
    loaded_train = np.load("./datasets/df_train_signature_data.npz")
    # loaded_test = np.load("./datasets/df_test_signature_data.npz")

    df_train_final = pd.DataFrame(data=loaded_train['data'], columns=loaded_train['features'])
    # df_test_final = pd.DataFrame(data=loaded_test['data'], columns=loaded_test['features'])

    df_train_new_features = pd.read_parquet("./datasets/df_train_new_features.parquet")
    # df_test_new_features = pd.read_parquet("./datasets/df_test_new_features.parquet")

    y_dict_train_RET = y_train.set_index('ID')['RET'].to_dict()
    # y_dict_test_RET = y_test.set_index('ID')['RET'].to_dict()
    y_dict_train_new_features = df_train_new_features.set_index('ID')[new_features].to_dict()
    # y_dict_test_new_features = df_test_new_features.set_index('ID')[new_features].to_dict()

    df_train_final['RET'] = df_train_final['ID'].map(y_dict_train_RET)
    # df_test_final['RET'] = df_test_final['ID'].map(y_dict_test_RET)
    df_train_final[new_features] = df_train_final['ID'].map(y_dict_train_new_features)
    # df_test_final[new_features] = df_test_final['ID'].map(y_dict_test_new_features)
    print(111)




    # industry_lst = df_train_final.INDUSTRY.unique() # 'INDUSTRY'
    # industry_group_lst = df_train_final.INDUSTRY_GROUP.unique() # 'INDUSTRY_GROUP'
    # sector_lst = df_train_final.SECTOR.unique() # 'SECTOR'
    # sub_industry_lst = df_train_final.SUB_INDUSTRY.unique() # 'SUB_INDUSTRY'

    # param_grid = {"alpha": np.logspace(-3, 3, num=10)}
    # results = []

    # for sector_idx in tqdm(sector_lst, total=len(sector_lst), desc="Processing training and prediction"):
    #     try:
    #         # Clean memory
    #         import gc
    #         gc.collect()
    #         df_sector_train = df_train_final[df_train_final["SECTOR"] == sector_idx]
    #         df_sector_test = df_test_final[df_test_final["SECTOR"] == sector_idx]
    #         # Build feature matrix
    #         feature_cols = [c for c in df_sector_train.columns if c.startswith("SIG_")]
            
    #         print(1)
    #         X_train = df_sector_train[feature_cols].values
    #         y_train = np.array(df_sector_train["RET"])
    #         X_test = df_sector_test[feature_cols].values
    #         y_test = np.array(df_sector_test["RET"])
    #         print(2)
    #         # Scale the features
    #         scaler = StandardScaler()
    #         X_train_scaled = scaler.fit_transform(X_train)  
    #         X_test_scaled = scaler.transform(X_test)
    #         print(3)
    #         # Train the model on the training set
    #         # Grid search (3-fold CV)
    #         gs = GridSearchCV(
    #             estimator=RidgeClassifier(solver="svg"),
    #             param_grid=param_grid,
    #             cv=3,
    #             scoring="accuracy",
    #             # scoring="f1",
    #             n_jobs=-1
    #         )
    #         gs.fit(X_train_scaled, y_train)

    #         print(4)
    #         # Evaluate on validation set
    #         best_alpha = gs.best_params_["alpha"]
    #         best_model = gs.best_estimator_
    #         y_val_pred = best_model.predict(X_test_scaled)
    #         acc = accuracy_score(y_test, y_val_pred)
    #         report = classification_report(y_test, y_val_pred)
    #         print(5)
    #         # Store results
    #         results.append({
    #             "Grouped_by": sector_idx,
    #             "Train_Samples": len(df_sector_train),
    #             "Val_Samples": len(df_sector_test),
    #             "Best_Alpha": best_alpha,
    #             "Val_Accuracy": acc,
    #             "Report": report,
    #             "Model": best_model
    #         })

    #     except Exception as e:
    #         print(f"Error processing sector {sector_idx}: {str(e)}")
    #         continue

    # # Summarize
    # df_results = pd.DataFrame(results)
    # print(df_results[[
    #     "Grouped_by", "Train_Samples", "Val_Samples",
    #     "Best_Alpha", "Val_Accuracy"
    # ]])
    # # df_results.to_csv("/Users/runminghuang/Desktop/capstone/Data/ridge_regression_by_SECTOR_20250303.csv", index=False)