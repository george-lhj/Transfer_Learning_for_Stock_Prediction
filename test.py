import numpy as np
import pandas as pd
import iisignature
from iisignature import sig, prepare, logsig, logsiglength
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from calc_signature import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, ttest_ind
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon
import os

if __name__ == "__main__":
    # 加载只有训练数据
    print("加载并准备数据...")
    x_train = pd.read_csv("./datasets/x_train.csv")
    y_train = pd.read_csv("./datasets/y_train.csv")
    
    # 将训练数据拆分为训练集和测试集 (80% 训练, 20% 测试)
    # 首先确保x_train和y_train有相同的ID顺序
    merged_data = pd.merge(x_train, y_train, on='ID')
    
    # 按照ID拆分数据
    train_ids, test_ids = train_test_split(
        merged_data['ID'].unique(), 
        test_size=0.2, 
        random_state=42
    )
    
    # 创建训练集和测试集
    train_data = merged_data[merged_data['ID'].isin(train_ids)]
    test_data = merged_data[merged_data['ID'].isin(test_ids)]
    
    # 将拆分后的数据还原为x和y
    x_train_split = train_data.drop(['RET'], axis=1)
    y_train_split = train_data[['ID', 'RET']]
    
    x_test_split = test_data.drop(['RET'], axis=1)
    y_test_split = test_data[['ID', 'RET']]
    
    print(f"训练集大小: {len(x_train_split)}, 测试集大小: {len(x_test_split)}")
    
    # # # 以下代码是为了首次运行时生成特征数据
    # # # 生成训练和测试特征
    new_features_train, df_train_new_features, df_train_sig_prepared = prepare_df_for_signature_computation(
        x_train_split, y_train_split, save=True, filename="df_train_split")
    new_features_test, df_test_new_features, df_test_sig_prepared = prepare_df_for_signature_computation(
        x_test_split, y_test_split, save=True, filename="df_test_split")
    
    # # # 处理新特征
    with open("./datasets/df_train_split_new_features.txt", "w") as f:
        f.write(str(new_features_train))
    
    # # 计算签名特征
    calc_signature(df_train_sig_prepared, order=3, new_features=new_features_train, filename="df_train_split", save=True)
    calc_signature(df_test_sig_prepared, order=3, new_features=new_features_train, filename="df_test_split", save=True)
    
    # 以下代码假设特征数据已经生成完毕
    # 如果您首次运行，请取消上面的注释进行特征生成
    
    # 加载特征数据
    with open("./datasets/df_train_new_features.txt", "r") as f:
        new_features = eval(f.read())
    
    try:
        # 尝试加载拆分后的特征数据
        loaded_train = np.load("./datasets/df_train_split_signature_data.npz")
        loaded_test = np.load("./datasets/df_test_split_signature_data.npz")
        
        df_train_new_features = pd.read_parquet("./datasets/df_train_split_new_features.parquet")
        df_test_new_features = pd.read_parquet("./datasets/df_test_split_new_features.parquet")
    except FileNotFoundError:
        # 如果文件不存在，使用原始数据集
        print("找不到拆分后的特征数据，将使用原始数据集并进行拆分...")
        loaded_train = np.load("./datasets/df_train_signature_data.npz")
        
        df_train_final = pd.DataFrame(data=loaded_train['data'], columns=loaded_train['features'])
        df_train_new_features = pd.read_parquet("./datasets/df_train_new_features.parquet")
        
        # 映射RET值
        y_dict_train_RET = y_train.set_index('ID')['RET'].to_dict()
        df_train_final['RET'] = df_train_final['ID'].map(y_dict_train_RET)
        
        # 添加新特征
        for feature in new_features:
            try:
                y_dict_train_new_feature = df_train_new_features.set_index('ID')[feature].to_dict()
                df_train_final[feature] = df_train_final['ID'].map(y_dict_train_new_feature)
            except KeyError:
                print(f"警告: 特征 {feature} 不在训练数据中")
        
        # 按ID拆分
        df_train_final_split = df_train_final[df_train_final['ID'].isin(train_ids)]
        df_test_final_split = df_train_final[df_train_final['ID'].isin(test_ids)]
        
        # 确保没有NaN值
        df_train_final_split = df_train_final_split.fillna(0)
        df_test_final_split = df_test_final_split.fillna(0)
    else:
        # 如果文件存在，直接使用
        df_train_final_split = pd.DataFrame(data=loaded_train['data'], columns=loaded_train['features'])
        df_test_final_split = pd.DataFrame(data=loaded_test['data'], columns=loaded_test['features'])
        
        # 映射RET值
        y_dict_train_RET = y_train_split.set_index('ID')['RET'].to_dict()
        y_dict_test_RET = y_test_split.set_index('ID')['RET'].to_dict()
        
        df_train_final_split['RET'] = df_train_final_split['ID'].map(y_dict_train_RET)
        df_test_final_split['RET'] = df_test_final_split['ID'].map(y_dict_test_RET)
        
        # 添加新特征
        for feature in new_features:
            try:
                y_dict_train_new_feature = df_train_new_features.set_index('ID')[feature].to_dict()
                y_dict_test_new_features = df_test_new_features.set_index('ID')[feature].to_dict()
                
                df_train_final_split[feature] = df_train_final_split['ID'].map(y_dict_train_new_feature)
                df_test_final_split[feature] = df_test_final_split['ID'].map(y_dict_test_new_features)
            except KeyError:
                print(f"警告: 特征 {feature} 不在数据中")
        
        # 确保没有NaN值
        df_train_final_split = df_train_final_split.fillna(0)
        df_test_final_split = df_test_final_split.fillna(0)
    
    print("数据准备完成，开始按行业分组训练模型...")
    print(df_train_final_split.head())
    print(df_test_final_split.head())
    
    # 确保行业分类列存在
    if 'SECTOR' not in df_train_final_split.columns or 'INDUSTRY' not in df_train_final_split.columns:
        # 加载行业分类数据
        sector_data = pd.read_csv("./datasets/stock_info.csv")
        sector_dict = sector_data.set_index('Symbol')['Sector'].to_dict()
        industry_dict = sector_data.set_index('Symbol')['Industry'].to_dict()
        sub_industry_dict = sector_data.set_index('Symbol')['Sub-Industry'].to_dict() if 'Sub-Industry' in sector_data.columns else {}
        
        # 获取股票代码
        def extract_symbol(id_str):
            # 确保id_str是字符串类型
            id_str = str(id_str)
            return id_str.split('_')[0] if '_' in id_str else id_str
        
        # 首先检查是否已存在STOCK列
        if 'STOCK' not in df_train_final_split.columns:
            df_train_final_split['STOCK'] = df_train_final_split['ID'].apply(extract_symbol)
        if 'STOCK' not in df_test_final_split.columns:
            df_test_final_split['STOCK'] = df_test_final_split['ID'].apply(extract_symbol)
        
        # 映射行业分类 - 使用STOCK列直接映射
        df_train_final_split['SECTOR'] = df_train_final_split['STOCK'].map(sector_dict)
        df_train_final_split['INDUSTRY'] = df_train_final_split['STOCK'].map(industry_dict)
        if sub_industry_dict:
            df_train_final_split['SUB_INDUSTRY'] = df_train_final_split['STOCK'].map(sub_industry_dict)
        
        df_test_final_split['SECTOR'] = df_test_final_split['STOCK'].map(sector_dict)
        df_test_final_split['INDUSTRY'] = df_test_final_split['STOCK'].map(industry_dict)
        if sub_industry_dict:
            df_test_final_split['SUB_INDUSTRY'] = df_test_final_split['STOCK'].map(sub_industry_dict)
    
    # 获取所有行业分类
    industries = df_train_final_split.INDUSTRY.dropna().unique()
    sectors = df_train_final_split.SECTOR.dropna().unique()
    
    # 选择使用哪个分类进行分组
    group_by = 'SECTOR'  # 使用SECTOR进行分组
    group_list = sectors if group_by == 'SECTOR' else industries
    
    # 设置Grid Search参数
    param_grid = {"alpha": np.logspace(-3, 3, num=10)}
    
    # 存储结果
    results = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    
    # 按行业分组训练
    for group_idx in tqdm(group_list, total=len(group_list), desc=f"按{group_by}分组训练和预测"):
        try:
            # 获取该组的训练和测试数据
            df_group_train = df_train_final_split[df_train_final_split[group_by] == group_idx]
            df_group_test = df_test_final_split[df_test_final_split[group_by] == group_idx]
            
            # 如果该组数据太少，跳过
            if len(df_group_train) < 10 or len(df_group_test) < 5:
                print(f"跳过 {group_idx} 因为样本数量不足 (训练: {len(df_group_train)}, 测试: {len(df_group_test)})")
                continue
            
            # 特征选择
            feature_cols = [c for c in df_group_train.columns if c.startswith("SIG_")]
            valid_new_features = [f for f in new_features if f in df_group_train.columns]
            
            # 可以选择是否使用新特征
            # feature_cols.extend(valid_new_features)
            
            # 准备训练和测试数据
            X_train = df_group_train[feature_cols].values
            y_train = (df_group_train['RET'] > 0).astype(int).values
            
            X_test = df_group_test[feature_cols].values
            y_test = (df_group_test['RET'] > 0).astype(int).values
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Grid Search
            gs = GridSearchCV(
                estimator=RidgeClassifier(),
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            gs.fit(X_train_scaled, y_train)
            
            # 获取最佳模型
            best_alpha = gs.best_params_["alpha"]
            best_model = gs.best_estimator_
            
            # 预测
            y_val_pred = best_model.predict(X_test_scaled)
            
            # 计算预测概率 (对于RidgeClassifier需要特殊处理)
            try:
                y_val_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            except AttributeError:
                # RidgeClassifier没有predict_proba方法
                decision_values = best_model.decision_function(X_test_scaled)
                y_val_prob = 1 / (1 + np.exp(-decision_values))
            
            # 计算指标
            acc = accuracy_score(y_test, y_val_pred)
            roc_auc = roc_auc_score(y_test, y_val_prob)
            
            # 计算PR AUC
            precision, recall, _ = precision_recall_curve(y_test, y_val_prob)
            pr_auc = auc(recall, precision)
            
            # 生成分类报告
            report = classification_report(y_test, y_val_pred, zero_division=0)
            
            # 保存预测结果
            y_true_all.extend(y_test)
            y_pred_all.extend(y_val_pred)
            y_prob_all.extend(y_val_prob)
            
            # 保存该组结果
            results.append({
                "Group": group_idx,
                "Train_Samples": len(df_group_train),
                "Test_Samples": len(df_group_test),
                "Best_Alpha": best_alpha,
                "Accuracy": acc,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "Report": report,
                "Model": best_model
            })
            
        except Exception as e:
            print(f"处理 {group_idx} 时出错: {str(e)}")
            continue
    
    # 汇总结果
    df_results = pd.DataFrame(results)
    print("\n按组别性能汇总:")
    print(df_results[["Group", "Train_Samples", "Test_Samples", "Best_Alpha", "Accuracy", "ROC_AUC", "PR_AUC"]])
    
    # 计算整体性能
    overall_accuracy = accuracy_score(y_true_all, y_pred_all)
    overall_roc_auc = roc_auc_score(y_true_all, y_prob_all)
    
    # 计算整体PR AUC
    overall_precision, overall_recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    overall_pr_auc = auc(overall_recall, overall_precision)
    
    print(f"\n整体性能:")
    print(f"准确率: {overall_accuracy:.4f}")
    print(f"ROC AUC: {overall_roc_auc:.4f}")
    print(f"PR AUC: {overall_pr_auc:.4f}")
    
    # 保存结果
    df_results.to_csv("./results/industry_ridge_regression_train_only.csv", index=False)
    
    # 计算各组样本数的分布
    group_counts = df_results.groupby('Group')['Train_Samples'].sum().reset_index()
    group_counts = group_counts.sort_values('Train_Samples', ascending=False)
    
    print("\n各组样本数分布:")
    print(group_counts.head(10))
    
    # 计算性能最好的几个组
    best_groups = df_results.sort_values('Accuracy', ascending=False).head(10)
    print("\n性能最好的10个组:")
    print(best_groups[["Group", "Train_Samples", "Accuracy", "ROC_AUC"]])
    
    print("\n完成!")

    # 第二部分：仅使用测试集数据
    print("\n" + "="*80)
    print("第二部分：仅使用测试集数据进行相同分析")
    print("="*80)
    
    # 加载测试数据
    print("\n加载并准备测试数据...")
    x_test_only = pd.read_csv("./datasets/x_test.csv")
    y_test_only = pd.read_csv("./datasets/test_rand.csv")
    
    # 将测试数据拆分为"训练集"和"测试集" (80% 训练, 20% 测试)
    merged_test_data = pd.merge(x_test_only, y_test_only, on='ID')
    
    # 按照ID拆分数据
    test_train_ids, test_test_ids = train_test_split(
        merged_test_data['ID'].unique(), 
        test_size=0.2, 
        random_state=42
    )
    
    # 创建训练集和测试集（从测试数据中划分）
    test_train_data = merged_test_data[merged_test_data['ID'].isin(test_train_ids)]
    test_test_data = merged_test_data[merged_test_data['ID'].isin(test_test_ids)]
    
    # 将拆分后的数据还原为x和y
    x_test_train = test_train_data.drop(['RET'], axis=1)
    y_test_train = test_train_data[['ID', 'RET']]
    
    x_test_test = test_test_data.drop(['RET'], axis=1)
    y_test_test = test_test_data[['ID', 'RET']]
    
    print(f"测试数据的训练集大小: {len(x_test_train)}, 测试集大小: {len(x_test_test)}")
    
    # 生成特征数据（如有必要）
    try:
        # 尝试加载已生成的特征数据
        loaded_test_train = np.load("./datasets/df_test_train_signature_data.npz")
        loaded_test_test = np.load("./datasets/df_test_test_signature_data.npz")
        
        df_test_train_features = pd.read_parquet("./datasets/df_test_train_new_features.parquet")
        df_test_test_features = pd.read_parquet("./datasets/df_test_test_new_features.parquet")
        
        print("成功加载测试数据的特征文件")
    except FileNotFoundError:
        # 如果文件不存在，从原始测试数据中处理
        print("找不到测试数据的特征文件，将使用原始测试数据生成特征...")
        
        # 如果需要生成特征，取消下面的注释
        # new_features_test_train, df_test_train_features, df_test_train_sig_prepared = prepare_df_for_signature_computation(
        #     x_test_train, y_test_train, save=True, filename="df_test_train")
        # new_features_test_test, df_test_test_features, df_test_test_sig_prepared = prepare_df_for_signature_computation(
        #     x_test_test, y_test_test, save=True, filename="df_test_test")
        
        # with open("./datasets/df_test_train_new_features.txt", "w") as f:
        #     f.write(str(new_features_test_train))
        
        # 计算签名特征
        # calc_signature(df_test_train_sig_prepared, order=3, new_features=new_features_test_train, filename="df_test_train", save=True)
        # calc_signature(df_test_test_sig_prepared, order=3, new_features=new_features_test_train, filename="df_test_test", save=True)
        
        # 从原始测试集加载特征
        loaded_test = np.load("./datasets/df_test_signature_data.npz")
        df_test_all = pd.DataFrame(data=loaded_test['data'], columns=loaded_test['features'])
        df_test_new_features = pd.read_parquet("./datasets/df_test_new_features.parquet")
        
        # 映射RET值
        y_dict_test_RET = y_test_only.set_index('ID')['RET'].to_dict()
        df_test_all['RET'] = df_test_all['ID'].map(y_dict_test_RET)
        
        # 添加新特征
        for feature in new_features:
            try:
                y_dict_test_new_feature = df_test_new_features.set_index('ID')[feature].to_dict()
                df_test_all[feature] = df_test_all['ID'].map(y_dict_test_new_feature)
            except KeyError:
                print(f"警告: 特征 {feature} 不在测试数据中")
        
        # 根据ID分割
        df_test_train_final = df_test_all[df_test_all['ID'].isin(test_train_ids)]
        df_test_test_final = df_test_all[df_test_all['ID'].isin(test_test_ids)]
        
        # 确保没有NaN值
        df_test_train_final = df_test_train_final.fillna(0)
        df_test_test_final = df_test_test_final.fillna(0)
    else:
        # 如果已经存在处理好的特征数据
        df_test_train_final = pd.DataFrame(data=loaded_test_train['data'], columns=loaded_test_train['features'])
        df_test_test_final = pd.DataFrame(data=loaded_test_test['data'], columns=loaded_test_test['features'])
        
        # 映射RET值
        y_dict_test_train_RET = y_test_train.set_index('ID')['RET'].to_dict()
        y_dict_test_test_RET = y_test_test.set_index('ID')['RET'].to_dict()
        
        df_test_train_final['RET'] = df_test_train_final['ID'].map(y_dict_test_train_RET)
        df_test_test_final['RET'] = df_test_test_final['ID'].map(y_dict_test_test_RET)
        
        # 添加新特征
        for feature in new_features:
            try:
                y_dict_test_train_feature = df_test_train_features.set_index('ID')[feature].to_dict()
                y_dict_test_test_feature = df_test_test_features.set_index('ID')[feature].to_dict()
                
                df_test_train_final[feature] = df_test_train_final['ID'].map(y_dict_test_train_feature)
                df_test_test_final[feature] = df_test_test_final['ID'].map(y_dict_test_test_feature)
            except KeyError:
                print(f"警告: 特征 {feature} 不在测试数据特征中")
        
        # 确保没有NaN值
        df_test_train_final = df_test_train_final.fillna(0)
        df_test_test_final = df_test_test_final.fillna(0)
    
    print("测试数据准备完成，开始按行业板块分组训练模型...")
    
    # 确保行业分类列存在
    if 'SECTOR' not in df_test_train_final.columns or 'INDUSTRY' not in df_test_train_final.columns:
        # 加载行业分类数据
        sector_data = pd.read_csv("./datasets/stock_info.csv")
        sector_dict = sector_data.set_index('Symbol')['Sector'].to_dict()
        industry_dict = sector_data.set_index('Symbol')['Industry'].to_dict()
        
        # 获取股票代码
        def extract_symbol(id_str):
            # 确保id_str是字符串类型
            id_str = str(id_str)
            return id_str.split('_')[0] if '_' in id_str else id_str
        
        # 首先检查是否已存在STOCK列
        if 'STOCK' not in df_test_train_final.columns:
            df_test_train_final['STOCK'] = df_test_train_final['ID'].apply(extract_symbol)
        if 'STOCK' not in df_test_test_final.columns:
            df_test_test_final['STOCK'] = df_test_test_final['ID'].apply(extract_symbol)
        
        # 映射行业分类 - 使用STOCK列直接映射
        df_test_train_final['SECTOR'] = df_test_train_final['STOCK'].map(sector_dict)
        df_test_train_final['INDUSTRY'] = df_test_train_final['STOCK'].map(industry_dict)
        
        df_test_test_final['SECTOR'] = df_test_test_final['STOCK'].map(sector_dict)
        df_test_test_final['INDUSTRY'] = df_test_test_final['STOCK'].map(industry_dict)
    
    # 获取所有板块
    test_sectors = df_test_train_final.SECTOR.dropna().unique()
    
    # 使用SECTOR进行分组
    test_group_by = 'SECTOR'
    test_group_list = test_sectors
    
    # 存储结果
    test_results = []
    test_y_true_all = []
    test_y_pred_all = []
    test_y_prob_all = []
    
    # 按板块分组训练
    for group_idx in tqdm(test_group_list, total=len(test_group_list), desc=f"按{test_group_by}分组训练和预测测试数据"):
        try:
            # 获取该组的训练和测试数据
            df_test_group_train = df_test_train_final[df_test_train_final[test_group_by] == group_idx]
            df_test_group_test = df_test_test_final[df_test_test_final[test_group_by] == group_idx]
            
            # 如果该组数据太少，跳过
            if len(df_test_group_train) < 10 or len(df_test_group_test) < 5:
                print(f"跳过 {group_idx} 因为样本数量不足 (训练: {len(df_test_group_train)}, 测试: {len(df_test_group_test)})")
                continue
            
            # 特征选择
            feature_cols = [c for c in df_test_group_train.columns if c.startswith("SIG_")]
            
            # 准备训练和测试数据
            X_train = df_test_group_train[feature_cols].values
            y_train = (df_test_group_train['RET'] > 0).astype(int).values
            
            X_test = df_test_group_test[feature_cols].values
            y_test = (df_test_group_test['RET'] > 0).astype(int).values
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Grid Search
            gs = GridSearchCV(
                estimator=RidgeClassifier(),
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            gs.fit(X_train_scaled, y_train)
            
            # 获取最佳模型
            best_alpha = gs.best_params_["alpha"]
            best_model = gs.best_estimator_
            
            # 预测
            y_val_pred = best_model.predict(X_test_scaled)
            
            # 计算预测概率
            try:
                y_val_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            except AttributeError:
                decision_values = best_model.decision_function(X_test_scaled)
                y_val_prob = 1 / (1 + np.exp(-decision_values))
            
            # 计算指标
            acc = accuracy_score(y_test, y_val_pred)
            try:
                roc_auc = roc_auc_score(y_test, y_val_prob)
                precision, recall, _ = precision_recall_curve(y_test, y_val_prob)
                pr_auc = auc(recall, precision)
            except:
                roc_auc = 0.5
                pr_auc = 0.5
                print(f"无法为 {group_idx} 计算ROC/PR AUC")
            
            # 生成分类报告
            report = classification_report(y_test, y_val_pred, zero_division=0)
            
            # 保存预测结果
            test_y_true_all.extend(y_test)
            test_y_pred_all.extend(y_val_pred)
            test_y_prob_all.extend(y_val_prob)
            
            # 保存该组结果
            test_results.append({
                "Group": group_idx,
                "Train_Samples": len(df_test_group_train),
                "Test_Samples": len(df_test_group_test),
                "Best_Alpha": best_alpha,
                "Accuracy": acc,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "Report": report
            })
            
        except Exception as e:
            print(f"处理测试数据 {group_idx} 时出错: {str(e)}")
            continue
    
    # 汇总结果
    df_test_results = pd.DataFrame(test_results)
    print("\n按组别性能汇总 (测试数据):")
    print(df_test_results[["Group", "Train_Samples", "Test_Samples", "Best_Alpha", "Accuracy", "ROC_AUC", "PR_AUC"]])
    
    # 计算整体性能
    try:
        test_overall_accuracy = accuracy_score(test_y_true_all, test_y_pred_all)
        test_overall_roc_auc = roc_auc_score(test_y_true_all, test_y_prob_all)
        
        test_overall_precision, test_overall_recall, _ = precision_recall_curve(test_y_true_all, test_y_prob_all)
        test_overall_pr_auc = auc(test_overall_recall, test_overall_precision)
        
        print(f"\n整体性能 (测试数据):")
        print(f"准确率: {test_overall_accuracy:.4f}")
        print(f"ROC AUC: {test_overall_roc_auc:.4f}")
        print(f"PR AUC: {test_overall_pr_auc:.4f}")
    except:
        print("无法计算整体性能指标")
    
    # 保存结果
    df_test_results.to_csv("./results/industry_ridge_regression_test_only.csv", index=False)
    
    # 计算各组样本数的分布
    test_group_counts = df_test_results.groupby('Group')['Train_Samples'].sum().reset_index()
    test_group_counts = test_group_counts.sort_values('Train_Samples', ascending=False)
    
    print("\n各组样本数分布 (测试数据):")
    print(test_group_counts.head(10))
    
    # 计算性能最好的几个组
    test_best_groups = df_test_results.sort_values('Accuracy', ascending=False).head(10)
    print("\n性能最好的10个组 (测试数据):")
    print(test_best_groups[["Group", "Train_Samples", "Accuracy", "ROC_AUC"]])
    
    # 比较训练数据和测试数据的结果
    print("\n" + "="*80)
    print("训练数据和测试数据结果比较")
    print("="*80)
    
    # 计算共同的板块
    common_groups = set(df_results['Group']) & set(df_test_results['Group'])
    print(f"\n共同的板块数量: {len(common_groups)}")
    
    # 比较共同板块的性能
    if common_groups:
        train_perf = df_results[df_results['Group'].isin(common_groups)][['Group', 'Accuracy', 'ROC_AUC']]
        test_perf = df_test_results[df_test_results['Group'].isin(common_groups)][['Group', 'Accuracy', 'ROC_AUC']]
        
        comparison = pd.merge(train_perf, test_perf, on='Group', suffixes=('_训练数据', '_测试数据'))
        comparison['准确率差异'] = comparison['Accuracy_训练数据'] - comparison['Accuracy_测试数据']
        comparison['ROC_AUC差异'] = comparison['ROC_AUC_训练数据'] - comparison['ROC_AUC_测试数据']
        
        print("\n共同板块性能比较:")
        print(comparison.sort_values('准确率差异', ascending=False))
        
        # 计算平均差异
        mean_acc_diff = comparison['准确率差异'].mean()
        mean_auc_diff = comparison['ROC_AUC差异'].mean()
        
        print(f"\n平均准确率差异: {mean_acc_diff:.4f}")
        print(f"平均ROC AUC差异: {mean_auc_diff:.4f}")
    
    print("\n所有分析完成!") 

    # 第三部分：用全部训练集训练，全部测试集验证
    print("\n" + "="*80)
    print("第三部分：使用完整训练集训练，完整测试集验证")
    print("="*80)
    
    # 加载完整数据集
    print("\n加载完整数据集...")
    x_train_full = pd.read_csv("./datasets/x_train.csv")
    y_train_full = pd.read_csv("./datasets/y_train.csv")
    x_test_full = pd.read_csv("./datasets/x_test.csv")
    y_test_full = pd.read_csv("./datasets/test_rand.csv")
    
    print(f"训练集大小: {len(x_train_full)}, 测试集大小: {len(x_test_full)}")
    
    # 加载处理好的特征
    try:
        # 尝试加载训练和测试特征
        loaded_train_full = np.load("./datasets/df_train_signature_data.npz")
        loaded_test_full = np.load("./datasets/df_test_signature_data.npz")
        
        df_train_full_features = pd.DataFrame(data=loaded_train_full['data'], columns=loaded_train_full['features'])
        df_test_full_features = pd.DataFrame(data=loaded_test_full['data'], columns=loaded_test_full['features'])
        
        df_train_new_features_full = pd.read_parquet("./datasets/df_train_new_features.parquet")
        df_test_new_features_full = pd.read_parquet("./datasets/df_test_new_features.parquet")
        
        print("成功加载完整数据集的特征")
    except FileNotFoundError:
        print("错误：找不到完整数据集的特征文件")
        print("请确保先运行ridge_regression_by_sector.py生成特征文件")
        exit(1)
        
    # 映射RET值
    y_dict_train_RET = y_train_full.set_index('ID')['RET'].to_dict()
    y_dict_test_RET = y_test_full.set_index('ID')['RET'].to_dict()
    
    df_train_full_features['RET'] = df_train_full_features['ID'].map(y_dict_train_RET)
    df_test_full_features['RET'] = df_test_full_features['ID'].map(y_dict_test_RET)
    
    # 添加新特征
    with open("./datasets/df_train_new_features.txt", "r") as f:
        new_features_full = eval(f.read())
        
    for feature in new_features_full:
        try:
            y_dict_train_new_feature = df_train_new_features_full.set_index('ID')[feature].to_dict()
            y_dict_test_new_feature = df_test_new_features_full.set_index('ID')[feature].to_dict()
            
            df_train_full_features[feature] = df_train_full_features['ID'].map(y_dict_train_new_feature)
            df_test_full_features[feature] = df_test_full_features['ID'].map(y_dict_test_new_feature)
        except KeyError:
            print(f"警告: 特征 {feature} 不在数据中")
    
    # 确保没有NaN值
    df_train_full_features = df_train_full_features.fillna(0)
    df_test_full_features = df_test_full_features.fillna(0)
    
    print("完整数据准备完成，开始按行业板块分组训练模型...")
    
    # 确保行业分类列存在
    if 'SECTOR' not in df_train_full_features.columns or 'INDUSTRY' not in df_train_full_features.columns:
        # 加载行业分类数据
        sector_data = pd.read_csv("./datasets/stock_info.csv")
        sector_dict = sector_data.set_index('Symbol')['Sector'].to_dict()
        industry_dict = sector_data.set_index('Symbol')['Industry'].to_dict()
        
        # 获取股票代码
        def extract_symbol(id_str):
            # 确保id_str是字符串类型
            id_str = str(id_str)
            return id_str.split('_')[0] if '_' in id_str else id_str
        
        # 首先检查是否已存在STOCK列
        if 'STOCK' not in df_train_full_features.columns:
            df_train_full_features['STOCK'] = df_train_full_features['ID'].apply(extract_symbol)
            df_test_full_features['STOCK'] = df_test_full_features['ID'].apply(extract_symbol)
        
        # 映射行业分类 - 使用STOCK列直接映射
        df_train_full_features['SECTOR'] = df_train_full_features['STOCK'].map(sector_dict)
        df_train_full_features['INDUSTRY'] = df_train_full_features['STOCK'].map(industry_dict)
        
        df_test_full_features['SECTOR'] = df_test_full_features['STOCK'].map(sector_dict)
        df_test_full_features['INDUSTRY'] = df_test_full_features['STOCK'].map(industry_dict)
    
    # 获取所有板块
    full_sectors = df_train_full_features.SECTOR.dropna().unique()
    
    # 使用SECTOR进行分组
    full_group_by = 'SECTOR'
    full_group_list = full_sectors
    
    # 设置Grid Search参数
    param_grid = {"alpha": np.logspace(-3, 3, num=10)}
    
    # 存储结果
    full_results = []
    full_y_true_all = []
    full_y_pred_all = []
    full_y_prob_all = []
    
    # 按板块分组训练
    for group_idx in tqdm(full_group_list, total=len(full_group_list), desc=f"按{full_group_by}分组训练和验证"):
        try:
            # 获取该组的训练和测试数据
            df_full_group_train = df_train_full_features[df_train_full_features[full_group_by] == group_idx]
            df_full_group_test = df_test_full_features[df_test_full_features[full_group_by] == group_idx]
            
            # 如果该组数据太少，跳过
            if len(df_full_group_train) < 10 or len(df_full_group_test) < 5:
                print(f"跳过 {group_idx} 因为样本数量不足 (训练: {len(df_full_group_train)}, 测试: {len(df_full_group_test)})")
                continue
            
            # 特征选择
            feature_cols = [c for c in df_full_group_train.columns if c.startswith("SIG_")]
            
            # 准备训练和测试数据
            X_train = df_full_group_train[feature_cols].values
            y_train = (df_full_group_train['RET'] > 0).astype(int).values
            
            X_test = df_full_group_test[feature_cols].values
            y_test = (df_full_group_test['RET'] > 0).astype(int).values
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 使用交叉验证找到最佳参数
            gs = GridSearchCV(
                estimator=RidgeClassifier(),
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            gs.fit(X_train_scaled, y_train)
            
            # 获取最佳模型
            best_alpha = gs.best_params_["alpha"]
            best_model = gs.best_estimator_
            
            # 在测试集上预测
            y_test_pred = best_model.predict(X_test_scaled)
            
            # 获取决策值或概率
            try:
                y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            except AttributeError:
                decision_values = best_model.decision_function(X_test_scaled)
                y_test_prob = 1 / (1 + np.exp(-decision_values))
            
            # 计算性能指标
            acc = accuracy_score(y_test, y_test_pred)
            
            try:
                roc_auc = roc_auc_score(y_test, y_test_prob)
                precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
                pr_auc = auc(recall, precision)
            except:
                roc_auc = 0.5
                pr_auc = 0.5
                print(f"无法为 {group_idx} 计算ROC/PR AUC")
            
            # 生成分类报告
            report = classification_report(y_test, y_test_pred, zero_division=0)
            
            # 保存预测结果
            full_y_true_all.extend(y_test)
            full_y_pred_all.extend(y_test_pred)
            full_y_prob_all.extend(y_test_prob)
            
            # 保存该组结果
            full_results.append({
                "Group": group_idx,
                "Train_Samples": len(df_full_group_train),
                "Test_Samples": len(df_full_group_test),
                "Best_Alpha": best_alpha,
                "Test_Accuracy": acc,
                "Test_ROC_AUC": roc_auc,
                "Test_PR_AUC": pr_auc,
                "Report": report
            })
            
        except Exception as e:
            print(f"处理 {group_idx} 时出错: {str(e)}")
            continue
    
    # 汇总结果
    df_full_results = pd.DataFrame(full_results)
    print("\n按组别性能汇总 (全部训练集训练，全部测试集验证):")
    print(df_full_results[["Group", "Train_Samples", "Test_Samples", "Best_Alpha", "Test_Accuracy", "Test_ROC_AUC", "Test_PR_AUC"]])
    
    # 计算整体性能
    full_overall_accuracy = accuracy_score(full_y_true_all, full_y_pred_all)
    full_overall_roc_auc = roc_auc_score(full_y_true_all, full_y_prob_all)
    
    full_overall_precision, full_overall_recall, _ = precision_recall_curve(full_y_true_all, full_y_prob_all)
    full_overall_pr_auc = auc(full_overall_recall, full_overall_precision)
    
    print(f"\n整体性能 (全部训练集训练，全部测试集验证):")
    print(f"准确率: {full_overall_accuracy:.4f}")
    print(f"ROC AUC: {full_overall_roc_auc:.4f}")
    print(f"PR AUC: {full_overall_pr_auc:.4f}")
    
    # 保存结果
    df_full_results.to_csv("./results/industry_ridge_regression_full_train_test.csv", index=False)
    
    # 计算各组样本数的分布
    full_group_counts = df_full_results.groupby('Group')['Train_Samples'].sum().reset_index()
    full_group_counts = full_group_counts.sort_values('Train_Samples', ascending=False)
    
    print("\n各组样本数分布 (全部训练集):")
    print(full_group_counts.head(10))
    
    # 计算性能最好的几个组
    full_best_groups = df_full_results.sort_values('Test_Accuracy', ascending=False).head(10)
    print("\n性能最好的10个组 (全部训练集训练，全部测试集验证):")
    print(full_best_groups[["Group", "Train_Samples", "Test_Accuracy", "Test_ROC_AUC"]])
    
    # 比较三种方法的结果
    print("\n" + "="*80)
    print("三种方法结果比较")
    print("="*80)
    
    # 计算三种方法共同的板块
    common_all_groups = set(df_results['Group']) & set(df_test_results['Group']) & set(df_full_results['Group'])
    print(f"\n三种方法共同的板块数量: {len(common_all_groups)}")
    
    if common_all_groups:
        method1_perf = df_results[df_results['Group'].isin(common_all_groups)][['Group', 'Accuracy', 'ROC_AUC']]
        method2_perf = df_test_results[df_test_results['Group'].isin(common_all_groups)][['Group', 'Accuracy', 'ROC_AUC']]
        method3_perf = df_full_results[df_full_results['Group'].isin(common_all_groups)][['Group', 'Test_Accuracy', 'Test_ROC_AUC']]
        
        # 重命名列以便合并
        method1_perf.columns = ['Group', 'Accuracy_方法1', 'ROC_AUC_方法1']
        method2_perf.columns = ['Group', 'Accuracy_方法2', 'ROC_AUC_方法2']
        method3_perf.columns = ['Group', 'Accuracy_方法3', 'ROC_AUC_方法3']
        
        # 合并三种方法的结果
        comparison_all = pd.merge(method1_perf, method2_perf, on='Group')
        comparison_all = pd.merge(comparison_all, method3_perf, on='Group')
        
        # 添加方法3与方法1、2的差异
        comparison_all['准确率差异_方法3vs1'] = comparison_all['Accuracy_方法3'] - comparison_all['Accuracy_方法1']
        comparison_all['准确率差异_方法3vs2'] = comparison_all['Accuracy_方法3'] - comparison_all['Accuracy_方法2']
        comparison_all['ROC_AUC差异_方法3vs1'] = comparison_all['ROC_AUC_方法3'] - comparison_all['ROC_AUC_方法1']
        comparison_all['ROC_AUC差异_方法3vs2'] = comparison_all['ROC_AUC_方法3'] - comparison_all['ROC_AUC_方法2']
        
        print("\n三种方法在共同板块上的性能比较:")
        print(comparison_all.sort_values('准确率差异_方法3vs1', ascending=False))
        
        # 计算平均差异
        mean_acc_diff_3vs1 = comparison_all['准确率差异_方法3vs1'].mean()
        mean_acc_diff_3vs2 = comparison_all['准确率差异_方法3vs2'].mean()
        mean_auc_diff_3vs1 = comparison_all['ROC_AUC差异_方法3vs1'].mean()
        mean_auc_diff_3vs2 = comparison_all['ROC_AUC差异_方法3vs2'].mean()
        
        print(f"\n方法3与方法1的平均准确率差异: {mean_acc_diff_3vs1:.4f}")
        print(f"方法3与方法2的平均准确率差异: {mean_acc_diff_3vs2:.4f}")
        print(f"方法3与方法1的平均ROC AUC差异: {mean_auc_diff_3vs1:.4f}")
        print(f"方法3与方法2的平均ROC AUC差异: {mean_auc_diff_3vs2:.4f}")
        
        # 总结三种方法的整体性能
        print("\n三种方法的整体性能比较:")
        print(f"方法1 (训练集内部分割): 准确率={overall_accuracy:.4f}, ROC AUC={overall_roc_auc:.4f}, PR AUC={overall_pr_auc:.4f}")
        print(f"方法2 (测试集内部分割): 准确率={test_overall_accuracy:.4f}, ROC AUC={test_overall_roc_auc:.4f}, PR AUC={test_overall_pr_auc:.4f}")
        print(f"方法3 (全训练集训练全测试集验证): 准确率={full_overall_accuracy:.4f}, ROC AUC={full_overall_roc_auc:.4f}, PR AUC={full_overall_pr_auc:.4f}")
    
    print("\n分析全部完成!") 

    # 第四部分：只使用训练集和测试集共有的股票
    print("\n" + "="*80)
    print("第四部分：只使用训练集和测试集共有的股票")
    print("="*80)
    
    # 加载完整数据集
    print("\n加载完整数据集并识别共有股票...")
    x_train_common = pd.read_csv("./datasets/x_train.csv")
    y_train_common = pd.read_csv("./datasets/y_train.csv")
    x_test_common = pd.read_csv("./datasets/x_test.csv")
    y_test_common = pd.read_csv("./datasets/test_rand.csv")
    
    # 加载处理好的特征
    try:
        loaded_train_common = np.load("./datasets/df_train_signature_data.npz")
        loaded_test_common = np.load("./datasets/df_test_signature_data.npz")
        
        df_train_common = pd.DataFrame(data=loaded_train_common['data'], columns=loaded_train_common['features'])
        df_test_common = pd.DataFrame(data=loaded_test_common['data'], columns=loaded_test_common['features'])
        
        df_train_new_features_common = pd.read_parquet("./datasets/df_train_new_features.parquet")
        df_test_new_features_common = pd.read_parquet("./datasets/df_test_new_features.parquet")
        
        print("成功加载特征数据")
    except FileNotFoundError:
        print("错误：找不到特征文件")
        print("请确保先运行ridge_regression_by_sector.py生成特征文件")
        exit(1)
    
    # 找出共同的股票（使用STOCK列）
    train_stocks = set(df_train_common['STOCK'].unique())
    test_stocks = set(df_test_common['STOCK'].unique())
    common_stocks = train_stocks & test_stocks
    
    # 统计股票数量
    print(f"训练集中的股票数量: {len(train_stocks)}")
    print(f"测试集中的股票数量: {len(test_stocks)}")
    print(f"共同股票数量: {len(common_stocks)}")
    print(f"只在训练集中的股票数量: {len(train_stocks - common_stocks)}")
    print(f"只在测试集中的股票数量: {len(test_stocks - common_stocks)}")
    
    # 只保留共同股票的数据
    df_train_common_filtered = df_train_common[df_train_common['STOCK'].isin(common_stocks)]
    df_test_common_filtered = df_test_common[df_test_common['STOCK'].isin(common_stocks)]
    
    print(f"过滤后训练集大小: {len(df_train_common_filtered)}, 测试集大小: {len(df_test_common_filtered)}")
    
    # 检查是否有足够的数据继续
    if len(df_train_common_filtered) == 0 or len(df_test_common_filtered) == 0:
        print("错误：过滤后没有足够的数据。请检查STOCK列是否正确。")
        print("跳过第四部分分析")
        print("\n分析全部完成!")
        exit(0)
    
    # 添加RET标签
    y_dict_train_RET = y_train_common.set_index('ID')['RET'].to_dict()
    y_dict_test_RET = y_test_common.set_index('ID')['RET'].to_dict()
    
    df_train_common_filtered['RET'] = df_train_common_filtered['ID'].map(y_dict_train_RET)
    df_test_common_filtered['RET'] = df_test_common_filtered['ID'].map(y_dict_test_RET)
    
    # 添加新特征
    with open("./datasets/df_train_new_features.txt", "r") as f:
        new_features_common = eval(f.read())
        
    for feature in new_features_common:
        try:
            y_dict_train_new_feature = df_train_new_features_common.set_index('ID')[feature].to_dict()
            y_dict_test_new_feature = df_test_new_features_common.set_index('ID')[feature].to_dict()
            
            df_train_common_filtered[feature] = df_train_common_filtered['ID'].map(y_dict_train_new_feature)
            df_test_common_filtered[feature] = df_test_common_filtered['ID'].map(y_dict_test_new_feature)
        except KeyError:
            print(f"警告: 特征 {feature} 不在数据中")
    
    # 确保没有NaN值
    df_train_common_filtered = df_train_common_filtered.fillna(0)
    df_test_common_filtered = df_test_common_filtered.fillna(0)
    
    # 添加行业分类
    if 'SECTOR' not in df_train_common_filtered.columns or 'INDUSTRY' not in df_train_common_filtered.columns:
        # 加载行业分类数据
        sector_data = pd.read_csv("./datasets/stock_info.csv")
        sector_dict = sector_data.set_index('Symbol')['Sector'].to_dict()
        industry_dict = sector_data.set_index('Symbol')['Industry'].to_dict()
        
        # 股票代码已经在STOCK列中，检查确保STOCK列存在
        if 'STOCK' not in df_train_common_filtered.columns:
            # 如果不存在，需要创建
            def extract_symbol(id_str):
                # 确保id_str是字符串类型
                id_str = str(id_str)
                return id_str.split('_')[0] if '_' in id_str else id_str
            
            df_train_common_filtered['STOCK'] = df_train_common_filtered['ID'].apply(extract_symbol)
            df_test_common_filtered['STOCK'] = df_test_common_filtered['ID'].apply(extract_symbol)
        
        # 映射行业分类 - 使用STOCK列直接映射
        df_train_common_filtered['SECTOR'] = df_train_common_filtered['STOCK'].map(sector_dict)
        df_train_common_filtered['INDUSTRY'] = df_train_common_filtered['STOCK'].map(industry_dict)
        
        df_test_common_filtered['SECTOR'] = df_test_common_filtered['STOCK'].map(sector_dict)
        df_test_common_filtered['INDUSTRY'] = df_test_common_filtered['STOCK'].map(industry_dict)
    
    # 进行训练集的内部分割 (80% 训练, 20% 测试)
    train_ids, val_ids = train_test_split(
        df_train_common_filtered['ID'].unique(), 
        test_size=0.2, 
        random_state=42
    )
    
    df_train_common_train = df_train_common_filtered[df_train_common_filtered['ID'].isin(train_ids)]
    df_train_common_val = df_train_common_filtered[df_train_common_filtered['ID'].isin(val_ids)]
    
    print(f"共同股票 - 训练集内部分割: 训练={len(df_train_common_train)}, 验证={len(df_train_common_val)}")
    
    # 获取所有板块
    common_sectors = df_train_common_filtered.SECTOR.dropna().unique()
    
    # 使用SECTOR进行分组
    common_group_by = 'SECTOR'
    common_group_list = common_sectors
    
    # 设置Grid Search参数
    param_grid = {"alpha": np.logspace(-3, 3, num=10)}
    
    # 存储各实验结果
    # 实验1：训练集内部分割
    common_results_internal = []
    common_y_true_internal = []
    common_y_pred_internal = []
    common_y_prob_internal = []
    
    # 实验2：训练集训练，测试集验证
    common_results_external = []
    common_y_true_external = []
    common_y_pred_external = []
    common_y_prob_external = []
    
    print("\n实验1：使用共同股票的训练集内部分割")
    
    # 按板块分组训练 - 内部分割
    for group_idx in tqdm(common_group_list, total=len(common_group_list), desc=f"按{common_group_by}分组训练和验证 - 内部分割"):
        try:
            # 获取该组的训练和验证数据
            df_group_train = df_train_common_train[df_train_common_train[common_group_by] == group_idx]
            df_group_val = df_train_common_val[df_train_common_val[common_group_by] == group_idx]
            
            # 如果该组数据太少，跳过
            if len(df_group_train) < 10 or len(df_group_val) < 5:
                print(f"跳过 {group_idx} 因为样本数量不足 (训练: {len(df_group_train)}, 验证: {len(df_group_val)})")
                continue
            
            # 特征选择
            feature_cols = [c for c in df_group_train.columns if c.startswith("SIG_")]
            
            # 准备训练和验证数据
            X_train = df_group_train[feature_cols].values
            y_train = (df_group_train['RET'] > 0).astype(int).values
            
            X_val = df_group_val[feature_cols].values
            y_val = (df_group_val['RET'] > 0).astype(int).values
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # GridSearch
            gs = GridSearchCV(
                estimator=RidgeClassifier(),
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            gs.fit(X_train_scaled, y_train)
            
            # 获取最佳模型
            best_alpha = gs.best_params_["alpha"]
            best_model = gs.best_estimator_
            
            # 预测
            y_val_pred = best_model.predict(X_val_scaled)
            
            # 计算预测概率
            try:
                y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]
            except AttributeError:
                decision_values = best_model.decision_function(X_val_scaled)
                y_val_prob = 1 / (1 + np.exp(-decision_values))
            
            # 计算指标
            acc = accuracy_score(y_val, y_val_pred)
            try:
                roc_auc = roc_auc_score(y_val, y_val_prob)
                precision, recall, _ = precision_recall_curve(y_val, y_val_prob)
                pr_auc = auc(recall, precision)
            except:
                roc_auc = 0.5
                pr_auc = 0.5
                print(f"无法为 {group_idx} 计算ROC/PR AUC")
            
            # 保存预测结果
            common_y_true_internal.extend(y_val)
            common_y_pred_internal.extend(y_val_pred)
            common_y_prob_internal.extend(y_val_prob)
            
            # 保存该组结果
            common_results_internal.append({
                "Group": group_idx,
                "Train_Samples": len(df_group_train),
                "Val_Samples": len(df_group_val),
                "Best_Alpha": best_alpha,
                "Accuracy": acc,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc
            })
            
        except Exception as e:
            print(f"处理 {group_idx} 时出错: {str(e)}")
            continue
    
    print("\n实验2：使用共同股票的训练集训练，测试集验证")
    
    # 按板块分组训练 - 训练集训练，测试集验证
    for group_idx in tqdm(common_group_list, total=len(common_group_list), desc=f"按{common_group_by}分组训练和验证 - 外部验证"):
        try:
            # 获取该组的训练和测试数据
            df_group_train = df_train_common_filtered[df_train_common_filtered[common_group_by] == group_idx]
            df_group_test = df_test_common_filtered[df_test_common_filtered[common_group_by] == group_idx]
            
            # 如果该组数据太少，跳过
            if len(df_group_train) < 10 or len(df_group_test) < 5:
                print(f"跳过 {group_idx} 因为样本数量不足 (训练: {len(df_group_train)}, 测试: {len(df_group_test)})")
                continue
            
            # 特征选择
            feature_cols = [c for c in df_group_train.columns if c.startswith("SIG_")]
            
            # 准备训练和测试数据
            X_train = df_group_train[feature_cols].values
            y_train = (df_group_train['RET'] > 0).astype(int).values
            
            X_test = df_group_test[feature_cols].values
            y_test = (df_group_test['RET'] > 0).astype(int).values
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # GridSearch
            gs = GridSearchCV(
                estimator=RidgeClassifier(),
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            gs.fit(X_train_scaled, y_train)
            
            # 获取最佳模型
            best_alpha = gs.best_params_["alpha"]
            best_model = gs.best_estimator_
            
            # 预测
            y_test_pred = best_model.predict(X_test_scaled)
            
            # 计算预测概率
            try:
                y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            except AttributeError:
                decision_values = best_model.decision_function(X_test_scaled)
                y_test_prob = 1 / (1 + np.exp(-decision_values))
            
            # 计算指标
            acc = accuracy_score(y_test, y_test_pred)
            try:
                roc_auc = roc_auc_score(y_test, y_test_prob)
                precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
                pr_auc = auc(recall, precision)
            except:
                roc_auc = 0.5
                pr_auc = 0.5
                print(f"无法为 {group_idx} 计算ROC/PR AUC")
            
            # 生成分类报告
            report = classification_report(y_test, y_test_pred, zero_division=0)
            
            # 保存预测结果
            common_y_true_external.extend(y_test)
            common_y_pred_external.extend(y_test_pred)
            common_y_prob_external.extend(y_test_prob)
            
            # 保存该组结果
            common_results_external.append({
                "Group": group_idx,
                "Train_Samples": len(df_group_train),
                "Test_Samples": len(df_group_test),
                "Best_Alpha": best_alpha,
                "Accuracy": acc,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc
            })
            
        except Exception as e:
            print(f"处理 {group_idx} 时出错: {str(e)}")
            continue
    
    # 汇总结果
    df_common_results_internal = pd.DataFrame(common_results_internal)
    df_common_results_external = pd.DataFrame(common_results_external)
    
    # 计算整体性能
    common_internal_accuracy = accuracy_score(common_y_true_internal, common_y_pred_internal)
    common_internal_roc_auc = roc_auc_score(common_y_true_internal, common_y_prob_internal)
    
    common_internal_precision, common_internal_recall, _ = precision_recall_curve(common_y_true_internal, common_y_prob_internal)
    common_internal_pr_auc = auc(common_internal_recall, common_internal_precision)
    
    common_external_accuracy = accuracy_score(common_y_true_external, common_y_pred_external)
    common_external_roc_auc = roc_auc_score(common_y_true_external, common_y_prob_external)
    
    common_external_precision, common_external_recall, _ = precision_recall_curve(common_y_true_external, common_y_prob_external)
    common_external_pr_auc = auc(common_external_recall, common_external_precision)
    
    # 打印结果
    print("\n共同股票 - 按组别性能汇总 (训练集内部分割):")
    print(df_common_results_internal[["Group", "Train_Samples", "Val_Samples", "Best_Alpha", "Accuracy", "ROC_AUC", "PR_AUC"]])
    
    print("\n共同股票 - 整体性能 (训练集内部分割):")
    print(f"准确率: {common_internal_accuracy:.4f}")
    print(f"ROC AUC: {common_internal_roc_auc:.4f}")
    print(f"PR AUC: {common_internal_pr_auc:.4f}")
    
    print("\n共同股票 - 按组别性能汇总 (训练集训练，测试集验证):")
    print(df_common_results_external[["Group", "Train_Samples", "Test_Samples", "Best_Alpha", "Accuracy", "ROC_AUC", "PR_AUC"]])
    
    print("\n共同股票 - 整体性能 (训练集训练，测试集验证):")
    print(f"准确率: {common_external_accuracy:.4f}")
    print(f"ROC AUC: {common_external_roc_auc:.4f}")
    print(f"PR AUC: {common_external_pr_auc:.4f}")
    
    # 保存结果
    df_common_results_internal.to_csv("./results/industry_ridge_regression_common_stocks_internal.csv", index=False)
    df_common_results_external.to_csv("./results/industry_ridge_regression_common_stocks_external.csv", index=False)
    
    # 比较结果
    print("\n" + "="*80)
    print("共同股票实验与完整数据集实验的比较")
    print("="*80)
    
    # 比较内部分割性能
    internal_diff = common_internal_accuracy - overall_accuracy
    print(f"\n内部分割性能差异 (共同股票 vs 全部股票): {internal_diff:.4f}")
    print(f"共同股票 (内部分割): 准确率={common_internal_accuracy:.4f}, ROC AUC={common_internal_roc_auc:.4f}, PR AUC={common_internal_pr_auc:.4f}")
    print(f"全部股票 (内部分割): 准确率={overall_accuracy:.4f}, ROC AUC={overall_roc_auc:.4f}, PR AUC={overall_pr_auc:.4f}")
    
    # 比较外部验证性能
    external_diff = common_external_accuracy - full_overall_accuracy
    print(f"\n外部验证性能差异 (共同股票 vs 全部股票): {external_diff:.4f}")
    print(f"共同股票 (外部验证): 准确率={common_external_accuracy:.4f}, ROC AUC={common_external_roc_auc:.4f}, PR AUC={common_external_pr_auc:.4f}")
    print(f"全部股票 (外部验证): 准确率={full_overall_accuracy:.4f}, ROC AUC={full_overall_roc_auc:.4f}, PR AUC={full_overall_pr_auc:.4f}")
    
    # 计算内外性能差异
    common_gap = common_internal_accuracy - common_external_accuracy
    full_gap = overall_accuracy - full_overall_accuracy
    gap_diff = common_gap - full_gap
    
    print(f"\n内外性能差异:")
    print(f"共同股票内外性能差距: {common_gap:.4f}")
    print(f"全部股票内外性能差距: {full_gap:.4f}")
    print(f"差距的差异: {gap_diff:.4f}")
    
    # 共同板块详细比较
    common_groups = set(df_common_results_internal['Group']) & set(df_common_results_external['Group'])
    if common_groups:
        internal_perf = df_common_results_internal[df_common_results_internal['Group'].isin(common_groups)][['Group', 'Accuracy', 'ROC_AUC']]
        external_perf = df_common_results_external[df_common_results_external['Group'].isin(common_groups)][['Group', 'Accuracy', 'ROC_AUC']]
        
        comparison = pd.merge(internal_perf, external_perf, on='Group', suffixes=('_内部', '_外部'))
        comparison['准确率差异'] = comparison['Accuracy_内部'] - comparison['Accuracy_外部']
        comparison['ROC_AUC差异'] = comparison['ROC_AUC_内部'] - comparison['ROC_AUC_外部']
        
        print("\n共同股票 - 各板块内外性能比较:")
        print(comparison.sort_values('准确率差异', ascending=False))
        
        # 计算平均差异
        mean_acc_diff = comparison['准确率差异'].mean()
        mean_auc_diff = comparison['ROC_AUC差异'].mean()
        
        print(f"\n共同股票 - 平均准确率差异: {mean_acc_diff:.4f}")
        print(f"共同股票 - 平均ROC AUC差异: {mean_auc_diff:.4f}")
    
    # 总体结论
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    print("\n所有方法的性能比较:")
    print(f"1. 训练集内部分割 (全部股票): 准确率={overall_accuracy:.4f}, ROC AUC={overall_roc_auc:.4f}")
    print(f"2. 测试集内部分割 (全部股票): 准确率={test_overall_accuracy:.4f}, ROC AUC={test_overall_roc_auc:.4f}")
    print(f"3. 训练集训练测试集验证 (全部股票): 准确率={full_overall_accuracy:.4f}, ROC AUC={full_overall_roc_auc:.4f}")
    print(f"4a. 训练集内部分割 (共同股票): 准确率={common_internal_accuracy:.4f}, ROC AUC={common_internal_roc_auc:.4f}")
    print(f"4b. 训练集训练测试集验证 (共同股票): 准确率={common_external_accuracy:.4f}, ROC AUC={common_external_roc_auc:.4f}")
    
    if common_internal_accuracy - common_external_accuracy < overall_accuracy - full_overall_accuracy:
        print("\n结论: 使用共同股票后，内外性能差距减小，说明股票不匹配是导致性能差异的重要原因之一。")
    else:
        print("\n结论: 使用共同股票后，内外性能差距并未明显减小，说明除股票不匹配外，还有其他因素导致性能差异。")
    
    print("\n所有分析全部完成!") 

    # ================================================================================
    # 第五部分：系统性差异分析
    # ================================================================================

    print("\n================================================================================")
    print("第五部分：系统性差异分析")
    print("================================================================================\n")

    # 确保必要的库已导入
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ks_2samp
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy

    # 创建图表保存目录
    try:
        plots_dir = './plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir, exist_ok=True)
            print(f"创建了保存目录: {plots_dir}")
        else:
            print(f"使用已存在的目录: {plots_dir}")
    except Exception as e:
        print(f"创建目录时出错: {str(e)}")
        plots_dir = '.'  # 如果创建目录失败，就使用当前目录
        print(f"将使用当前目录保存文件")

    # 初始化分析状态
    analysis_failed = False

    # 检查必要的数据是否存在
    if 'df_train_common' not in locals() or 'df_test_common' not in locals():
        print("错误: 未找到共同股票的数据，请先运行第四部分代码")
        analysis_failed = True
    elif df_train_common.empty or df_test_common.empty:
        print("错误: 共同股票的数据集为空")
        analysis_failed = True
    else:
        # 使用过滤后的数据集 (只包含共同股票)
        df_train_common = df_train_common_filtered if 'df_train_common_filtered' in locals() and not df_train_common_filtered.empty else df_train_common
        df_test_common = df_test_common_filtered if 'df_test_common_filtered' in locals() and not df_test_common_filtered.empty else df_test_common
        print(f"共同股票数据集大小 - 训练: {len(df_train_common)}, 测试: {len(df_test_common)}")

    if not analysis_failed:
        print("\n1. 特征分布差异分析")
        print("-"*60)

        # 1.1 KS检验 - 比较所有特征的分布差异
        # 检查是否有'SIG_'前缀的特征列
        feature_cols = [col for col in df_train_common.columns if col.startswith('SIG_')]
        
        if not feature_cols:
            print("警告: 未找到以'SIG_'开头的特征列，尝试查找其他数值型特征...")
            # 尝试查找所有数值型列作为特征
            feature_cols = df_train_common.select_dtypes(include=[np.number]).columns.tolist()
            # 排除ID、RET等非特征列
            exclude_cols = ['ID', 'RET', 'STOCK', 'SECTOR', 'INDUSTRY', 'SUB_INDUSTRY']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            print(f"找到 {len(feature_cols)} 个数值型特征列")
        
        if not feature_cols:
            print("错误: 无法找到可用的特征列进行分析")
            analysis_failed = True
        
        if not analysis_failed:
            ks_results = {}
            for feature in feature_cols:
                try:
                    # 排除可能的缺失值或无限值
                    train_vals = df_train_common[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    test_vals = df_test_common[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(train_vals) > 0 and len(test_vals) > 0:
                        stat, p_value = ks_2samp(train_vals, test_vals)
                        ks_results[feature] = {'statistic': stat, 'p_value': p_value}
                    else:
                        print(f"警告: 特征 {feature} 在清理后没有足够的有效值进行分析")
                except Exception as e:
                    print(f"分析特征 {feature} 时出错: {str(e)}")
            
            if not ks_results:
                print("警告: 所有特征的KS检验都失败了，无法进行分布差异分析")
                analysis_failed = True
            
            if not analysis_failed:
                # 排序并显示差异最大的特征
                sorted_features = sorted(ks_results.items(), key=lambda x: x[1]['statistic'], reverse=True)
                max_display = min(10, len(sorted_features))
                print(f"特征分布差异最大的{max_display}个特征 (总共分析了 {len(ks_results)} 个特征):")
                for feat, stats in sorted_features[:max_display]:
                    print(f"{feat}: KS统计量={stats['statistic']:.4f}, p值={stats['p_value']:.8f}")

    # 2. 标签(RET)分布分析
    print("\n2. 标签(RET)分布分析")
    print("-"*60)

    # 2.1 基本统计比较
    train_ret = df_train_common['RET'].dropna()
    test_ret = df_test_common['RET'].dropna()

    train_stats = train_ret.describe()
    test_stats = test_ret.describe()

    print("训练集RET统计量:")
    print(train_stats)
    print("\n测试集RET统计量:")
    print(test_stats)

    # 2.2 可视化标签分布
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_ret, kde=True, color='blue', alpha=0.5, label='训练集')
        sns.histplot(test_ret, kde=True, color='red', alpha=0.5, label='测试集')
        plt.title('RET标签分布比较')
        plt.xlabel('RET值')
        plt.ylabel('频数')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'ret_distribution.png'))
        print(f"RET分布图已保存到: {os.path.join(plots_dir, 'ret_distribution.png')}")
        plt.close()
    except Exception as e:
        print(f"绘制RET分布图时出错: {str(e)}")

    # 2.3 上涨/下跌比例比较
    train_pos_ratio = (train_ret > 0).mean()
    test_pos_ratio = (test_ret > 0).mean()
    train_neg_ratio = (train_ret <= 0).mean()
    test_neg_ratio = (test_ret <= 0).mean()

    print(f"训练集上涨比例: {train_pos_ratio:.4f}, 下跌比例: {train_neg_ratio:.4f}")
    print(f"测试集上涨比例: {test_pos_ratio:.4f}, 下跌比例: {test_neg_ratio:.4f}")
    print(f"上涨比例差异: {abs(train_pos_ratio - test_pos_ratio):.4f}")

    # 3. 行业/板块表现分析
    print("\n3. 行业/板块表现分析")
    print("-"*60)

    # 3.1 各行业在不同数据集中的表现比较
    if 'SECTOR' in df_train_common.columns and 'SECTOR' in df_test_common.columns:
        sector_results = []
        for sector in df_train_common['SECTOR'].dropna().unique():
            train_sector = df_train_common[df_train_common['SECTOR'] == sector]
            test_sector = df_test_common[df_test_common['SECTOR'] == sector]
            
            if len(train_sector) > 0 and len(test_sector) > 0:
                train_sector_ret = train_sector['RET'].mean()
                test_sector_ret = test_sector['RET'].mean()
                train_sector_count = len(train_sector)
                test_sector_count = len(test_sector)
                train_sector_ratio = len(train_sector) / len(df_train_common)
                test_sector_ratio = len(test_sector) / len(df_test_common)
                
                sector_results.append({
                    'SECTOR': sector,
                    'train_ret': train_sector_ret,
                    'test_ret': test_sector_ret,
                    'ret_diff': abs(train_sector_ret - test_sector_ret),
                    'train_count': train_sector_count,
                    'test_count': test_sector_count,
                    'train_ratio': train_sector_ratio,
                    'test_ratio': test_sector_ratio,
                    'ratio_diff': abs(train_sector_ratio - test_sector_ratio)
                })
        
        sector_df = pd.DataFrame(sector_results)
        if not sector_df.empty:
            # 按收益率差异排序
            sector_df_sorted = sector_df.sort_values('ret_diff', ascending=False)
            print("各行业收益率差异 (按差异大小排序):")
            print(sector_df_sorted[['SECTOR', 'train_ret', 'test_ret', 'ret_diff']].to_string(index=False))
            
            # 按比例差异排序
            sector_df_ratio_sorted = sector_df.sort_values('ratio_diff', ascending=False)
            print("\n各行业在数据集中的比例差异 (按差异大小排序):")
            print(sector_df_ratio_sorted[['SECTOR', 'train_ratio', 'test_ratio', 'ratio_diff']].to_string(index=False))
            
            # 可视化行业收益率差异
            try:
                plt.figure(figsize=(12, 8))
                sector_df_sorted = sector_df_sorted.head(10)  # 取差异最大的10个行业
                plt.bar(sector_df_sorted['SECTOR'].astype(str), sector_df_sorted['ret_diff'])
                plt.title('行业收益率差异 (Top 10)')
                plt.xlabel('行业')
                plt.ylabel('收益率差异')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'sector_ret_differences.png'))
                print(f"行业收益率差异图已保存到: {os.path.join(plots_dir, 'sector_ret_differences.png')}")
                plt.close()
            except Exception as e:
                print(f"绘制行业收益率差异图时出错: {str(e)}")
            
            # 可视化行业比例差异
            try:
                plt.figure(figsize=(12, 8))
                sector_df_ratio_sorted = sector_df_ratio_sorted.head(10)  # 取差异最大的10个行业
                plt.bar(sector_df_ratio_sorted['SECTOR'].astype(str), sector_df_ratio_sorted['ratio_diff'])
                plt.title('行业比例差异 (Top 10)')
                plt.xlabel('行业')
                plt.ylabel('比例差异')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'sector_ratio_differences.png'))
                print(f"行业比例差异图已保存到: {os.path.join(plots_dir, 'sector_ratio_differences.png')}")
                plt.close()
            except Exception as e:
                print(f"绘制行业比例差异图时出错: {str(e)}")
    else:
        print("数据集中没有SECTOR列，无法进行行业分析")

    # 4. 模型稳定性分析
    if not analysis_failed:
        print("\n4. 模型稳定性分析")
        print("-"*60)

        # 4.1 特征重要性分析
        # 确认feature_cols是否被定义
        if 'feature_cols' not in locals() or not feature_cols:
            feature_cols = [col for col in df_train_common.columns if col.startswith('SIG_')]
            if not feature_cols:
                print("警告: 未找到以'SIG_'开头的特征列，尝试查找其他数值型特征...")
                # 尝试查找所有数值型列作为特征
                feature_cols = df_train_common.select_dtypes(include=[np.number]).columns.tolist()
                # 排除ID、RET等非特征列
                exclude_cols = ['ID', 'RET', 'STOCK', 'SECTOR', 'INDUSTRY', 'SUB_INDUSTRY']
                feature_cols = [col for col in feature_cols if col not in exclude_cols]
                print(f"找到 {len(feature_cols)} 个数值型特征列")

        if len(feature_cols) > 0:
            # 在训练集上训练随机森林模型
            X_train = df_train_common[feature_cols].fillna(0)
            y_train = df_train_common['RET']
            
            # 检查是否有足够的数据进行训练
            if len(X_train) < 100 or len(np.unique(y_train)) < 2:
                print("警告: 数据不足或标签类别不足，无法训练随机森林模型")
            else:
                try:
                    # 使用较小的n_estimators以加快速度
                    rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                    rf_model.fit(X_train, y_train)
                    
                    # 获取特征重要性
                    feature_importances = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # 保存前20个最重要特征
                    top_features_count = min(20, len(feature_importances))
                    top_features = feature_importances.head(top_features_count)
                    print(f"模型中最重要的{top_features_count}个特征:")
                    print(top_features.to_string(index=False))
                    try:
                        top_features.to_csv(os.path.join(plots_dir, 'top_feature_importances.csv'), index=False)
                        print(f"特征重要性已保存到: {os.path.join(plots_dir, 'top_feature_importances.csv')}")
                    except Exception as e:
                        print(f"保存特征重要性时出错: {str(e)}")
                    
                    # 可视化前10个特征的重要性
                    try:
                        plt.figure(figsize=(12, 8))
                        top_n_features = min(10, len(feature_importances))
                        sns.barplot(x='importance', y='feature', data=top_features.head(top_n_features))
                        plt.title('特征重要性 (Top 10)')
                        plt.tight_layout()
                        plt.savefig(os.path.join(plots_dir, 'feature_importances.png'))
                        print(f"特征重要性图已保存到: {os.path.join(plots_dir, 'feature_importances.png')}")
                        plt.close()
                    except Exception as e:
                        print(f"绘制特征重要性图时出错: {str(e)}")
                    
                    # 4.2 检查这些重要特征在训练集和测试集中的分布差异
                    important_features = top_features['feature'].head(10).tolist()
                    for feature in important_features:
                        try:
                            train_vals = df_train_common[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                            test_vals = df_test_common[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if len(train_vals) > 0 and len(test_vals) > 0:
                                stat, p_value = ks_2samp(train_vals, test_vals)
                                print(f"重要特征 {feature} 的分布差异: KS统计量={stat:.4f}, p值={p_value:.8f}")
                        except Exception as e:
                            print(f"分析重要特征 {feature} 时出错: {str(e)}")
                except Exception as e:
                    print(f"训练随机森林模型时出错: {str(e)}")
        else:
            print("数据集中没有可用的特征列，无法进行特征重要性分析")

    # 5. 维度缩减分析 (PCA)
    if not analysis_failed:
        print("\n5. 维度缩减分析")
        print("-"*60)

        # 确认feature_cols是否被定义
        if 'feature_cols' not in locals() or not feature_cols:
            feature_cols = [col for col in df_train_common.columns if col.startswith('SIG_')]
            if not feature_cols:
                print("警告: 未找到以'SIG_'开头的特征列，无法进行PCA分析")
                analysis_failed = True
        
        if not analysis_failed and len(feature_cols) > 1:  # 需要至少2个特征才能进行PCA
            # 准备数据
            X_train = df_train_common[feature_cols].fillna(0)
            X_test = df_test_common[feature_cols].fillna(0)
            
            # 检查数据大小
            if X_train.shape[0] < 10 or X_test.shape[0] < 10:
                print("警告: 数据样本太少，无法进行有意义的PCA分析")
            else:
                try:
                    # 应用PCA
                    pca = PCA(n_components=min(2, len(feature_cols)))
                    train_pca = pca.fit_transform(X_train)
                    test_pca = pca.transform(X_test)
                    
                    # 查看解释方差
                    explained_variance = pca.explained_variance_ratio_
                    if len(explained_variance) >= 2:
                        print(f"前两个主成分解释的方差比例: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
                        print(f"总解释方差: {sum(explained_variance):.4f}")
                    else:
                        print(f"主成分解释的方差比例: {explained_variance[0]:.4f}")
                        print(f"总解释方差: {sum(explained_variance):.4f}")
                    
                    # 可视化PCA结果
                    try:
                        plt.figure(figsize=(10, 8))
                        
                        # 随机抽样以避免过度绘图
                        train_sample_size = min(5000, len(train_pca))
                        test_sample_size = min(5000, len(test_pca))
                        
                        if train_sample_size > 0 and test_sample_size > 0:
                            train_indices = np.random.choice(len(train_pca), train_sample_size, replace=False)
                            test_indices = np.random.choice(len(test_pca), test_sample_size, replace=False)
                            
                            if train_pca.shape[1] >= 2 and test_pca.shape[1] >= 2:
                                plt.scatter(train_pca[train_indices, 0], train_pca[train_indices, 1], alpha=0.5, color='blue', label='训练集')
                                plt.scatter(test_pca[test_indices, 0], test_pca[test_indices, 1], alpha=0.5, color='red', label='测试集')
                                plt.title('PCA降维结果 - 训练集与测试集比较')
                                plt.xlabel('主成分1')
                                plt.ylabel('主成分2')
                                plt.legend()
                                plt.savefig(os.path.join(plots_dir, 'pca_comparison.png'))
                                print(f"PCA比较图已保存到: {os.path.join(plots_dir, 'pca_comparison.png')}")
                            else:
                                print("警告: PCA结果维度不足，无法绘制二维散点图")
                        else:
                            print("警告: 数据样本太少，无法绘制有意义的PCA散点图")
                        plt.close()
                    except Exception as e:
                        print(f"绘制PCA比较图时出错: {str(e)}")
                except Exception as e:
                    print(f"PCA分析时出错: {str(e)}")
        else:
            if not analysis_failed:  # 避免重复打印错误
                print("数据集中特征数量不足，无法进行PCA分析")

    # 6. 统计结论
    print("\n6. 统计结论")
    print("-"*60)

    # 汇总不同类型的差异，并尝试量化它们的重要性
    # 6.1 特征分布差异
    significant_features = sum(1 for _, stats in ks_results.items() if stats['p_value'] < 0.05)
    total_features = len(ks_results)
    print(f"特征分布显著不同的比例: {significant_features / total_features:.4f} ({significant_features}/{total_features})")

    # 6.2 标签分布差异
    ret_ks_stat, ret_p_value = ks_2samp(train_ret, test_ret)
    print(f"标签(RET)分布差异: KS统计量={ret_ks_stat:.4f}, p值={ret_p_value:.8f}")

    # 6.3 总结可能的性能影响因素
    print("\n根据分析，以下因素可能影响模型性能差异:")
    print("1. 特征分布差异")
    print(f"   - {significant_features / total_features:.1%}的特征在训练集和测试集中分布显著不同")
    print(f"   - 特征分布差异最大的是: {sorted_features[0][0]} (KS统计量={sorted_features[0][1]['statistic']:.4f})")

    if abs(train_pos_ratio - test_pos_ratio) > 0.01:
        print("2. 标签(RET)分布差异")
        print(f"   - 训练集上涨比例: {train_pos_ratio:.4f}, 测试集上涨比例: {test_pos_ratio:.4f}")
        print(f"   - 上涨比例差异: {abs(train_pos_ratio - test_pos_ratio):.4f}")

    if 'SECTOR' in df_train_common.columns and not sector_df.empty:
        max_sector_ret_diff = sector_df['ret_diff'].max()
        max_sector_ratio_diff = sector_df['ratio_diff'].max()
        
        if max_sector_ret_diff > 0.05:
            print("3. 行业收益差异")
            max_diff_sector = sector_df.loc[sector_df['ret_diff'].idxmax(), 'SECTOR']
            print(f"   - 最大收益率差异出现在行业 {max_diff_sector} (差异={max_sector_ret_diff:.4f})")
        
        if max_sector_ratio_diff > 0.05:
            print("4. 行业分布差异")
            max_ratio_diff_sector = sector_df.loc[sector_df['ratio_diff'].idxmax(), 'SECTOR']
            print(f"   - 最大行业比例差异出现在行业 {max_ratio_diff_sector} (比例差异={max_sector_ratio_diff:.4f})")

    # 7. 输出建议
    if not analysis_failed:
        print("\n7. 改进建议")
        print("-"*60)
        print("根据分析结果，提出以下改进建议:")
        
        # 定义一些默认变量，以防前面的代码没有生成这些变量
        if 'significant_features' not in locals() or 'total_features' not in locals():
            significant_features = 0
            total_features = 1
        
        if 'train_pos_ratio' not in locals() or 'test_pos_ratio' not in locals():
            train_pos_ratio = 0
            test_pos_ratio = 0
        
        if 'sector_df' not in locals():
            sector_df = pd.DataFrame()
        
        # 7.1 针对特征分布差异的建议
        if significant_features / total_features > 0.3:
            print("1. 针对特征分布差异:")
            print("   - 考虑使用更鲁棒的特征变换方法，如分位数变换")
            print("   - 对分布差异最大的特征进行单独规范化处理")
            print("   - 尝试移除分布差异最大的特征，评估模型性能变化")

        # 7.2 针对行业/板块差异的建议
        if 'SECTOR' in df_train_common.columns and not sector_df.empty and sector_df['ret_diff'].max() > 0.05:
            print("2. 针对行业差异:")
            print("   - 考虑为不同行业训练单独的模型")
            print("   - 对行业特征加入交互项，增强模型的行业适应性")
            print("   - 探索行业中性化策略，减少行业偏差影响")

        # 7.3 针对标签分布差异的建议
        if abs(train_pos_ratio - test_pos_ratio) > 0.02:
            print("3. 针对标签分布差异:")
            print("   - 考虑分层采样技术，确保训练和测试集具有相似的标签分布")
            print("   - 探索阈值调整技术，根据测试集的上涨/下跌比例调整预测阈值")
            print("   - 添加上涨/下跌比例作为模型特征，帮助模型适应分布变化")

        # 7.4 其他一般性建议
        print("4. 一般性建议:")
        print("   - 尝试更简单的模型，减少过拟合可能性")
        print("   - 探索集成方法，如模型加权平均或投票")
        print("   - 考虑时间序列交叉验证，而非随机分割")
        print("   - 实现自适应特征选择机制，根据不同数据集调整特征权重")

        # 记录分析完成
        try:
            with open(os.path.join(plots_dir, 'analysis_completed.txt'), 'w') as f:
                f.write("系统性差异分析完成\n")
                f.write(f"分析时间: {pd.Timestamp.now()}\n")
                f.write(f"特征数量: {total_features}\n")
                f.write(f"显著不同的特征比例: {significant_features/total_features:.2%}\n")
                f.write(f"训练集上涨比例: {train_pos_ratio:.4f}\n")
                f.write(f"测试集上涨比例: {test_pos_ratio:.4f}\n")
            print("\n分析完成！所有图表和详细结果已保存至 %s 目录" % plots_dir)
        except Exception as e:
            print(f"保存分析总结时出错: {str(e)}")
            print("\n分析完成！但保存总结文件失败。")
    else:
        print("\n分析由于错误而中止。请检查以上错误信息并修复问题后重试。")

    print("\n所有分析全部完成!") 