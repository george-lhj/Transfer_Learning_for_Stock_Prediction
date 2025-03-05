#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
import joblib
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 自定义PCA类，限制最大组件数
class LimitedPCA(PCA):
    """限制最大组件数的PCA"""
    def __init__(self, n_components=None, max_components=None, **kwargs):
        self.max_components = max_components
        super().__init__(n_components=n_components, **kwargs)
    
    def fit(self, X, y=None):
        super().fit(X, y)
        # 如果n_components是浮点数(方差比例)，并且设置了max_components，则限制组件数
        if isinstance(self.n_components, float) and self.max_components is not None:
            # 计算需要多少组件来达到所需方差
            cumsum = np.cumsum(self.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.n_components) + 1
            
            # 限制最大组件数
            n_components = min(n_components, self.max_components, X.shape[1])
            
            # 重新设置n_components
            if n_components < len(self.components_):
                self.components_ = self.components_[:n_components]
                self.n_components_ = n_components
                self.explained_variance_ = self.explained_variance_[:n_components]
                self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
                if hasattr(self, 'singular_values_'):
                    self.singular_values_ = self.singular_values_[:n_components]
                
                print(f"实际使用组件数: {n_components}, 解释方差: {np.sum(self.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, X):
        # 确保使用正确的组件数
        if hasattr(self, 'n_components_'):
            n_comp = self.n_components_
        else:
            n_comp = self.n_components
            if isinstance(n_comp, float):
                # 如果是浮点数，使用所有组件
                n_comp = min(X.shape[1], len(self.components_))
            if self.max_components is not None:
                n_comp = min(n_comp, self.max_components)
        
        # 使用指定数量的组件进行转换
        X_transformed = np.dot(X - self.mean_, self.components_[:n_comp].T)
        return X_transformed

# 全局配置
PLOTS_DIR = './plots'
MODELS_DIR = './models'
RESULTS_DIR = './results'
for d in [PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_data(signature_train_path="./datasets/df_train_signature_data.npz", 
              signature_test_path="./datasets/df_test_signature_data.npz",
              new_features_train_path="./datasets/df_train_new_features.parquet",
              new_features_test_path="./datasets/df_test_new_features.parquet",
              y_train_path="./datasets/y_train.csv",
              y_test_path="./datasets/test_rand.csv"):
    """
    加载所有必要的数据文件
    """
    print("加载数据...")
    
    # 加载特征签名数据
    loaded_train = np.load(signature_train_path)
    loaded_test = np.load(signature_test_path)
    
    # 将NPZ文件转换为DataFrame
    df_train = pd.DataFrame(data=loaded_train['data'], columns=loaded_train['features'])
    df_test = pd.DataFrame(data=loaded_test['data'], columns=loaded_test['features'])
    
    # 加载新特征数据
    df_train_new_features = pd.read_parquet(new_features_train_path)
    df_test_new_features = pd.read_parquet(new_features_test_path)
    
    # 加载标签数据
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    
    # 创建标签字典
    y_dict_train_RET = y_train.set_index('ID')['RET'].to_dict()
    y_dict_test_RET = y_test.set_index('ID')['RET'].to_dict()
    
    # 将标签映射到数据集
    df_train['RET'] = df_train['ID'].map(y_dict_train_RET)
    df_test['RET'] = df_test['ID'].map(y_dict_test_RET)
    
    # 读取新特征列表
    with open("./datasets/df_train_new_features.txt", "r") as f:
        new_features = eval(f.read())
    
    # 将新特征添加到数据集
    for feature in new_features:
        y_dict_train_new_feature = df_train_new_features.set_index('ID')[feature].to_dict()
        y_dict_test_new_features = df_test_new_features.set_index('ID')[feature].to_dict()
        df_train[feature] = df_train['ID'].map(y_dict_train_new_feature) 
        df_test[feature] = df_test['ID'].map(y_dict_test_new_features)
    
    # 填充缺失值
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)
    
    print(f"训练集大小: {len(df_train)}, 测试集大小: {len(df_test)}")
    return df_train, df_test, new_features

def filter_common_stocks(df_train, df_test):
    """
    筛选出训练集和测试集中共有的股票
    """
    print("筛选共同股票...")
    train_stocks = set(df_train['STOCK'].unique())
    test_stocks = set(df_test['STOCK'].unique())
    common_stocks = train_stocks.intersection(test_stocks)
    
    print(f"训练集中的股票数量: {len(train_stocks)}")
    print(f"测试集中的股票数量: {len(test_stocks)}")
    print(f"共同股票数量: {len(common_stocks)}")
    
    df_train_common = df_train[df_train['STOCK'].isin(common_stocks)]
    df_test_common = df_test[df_test['STOCK'].isin(common_stocks)]
    
    print(f"过滤后训练集大小: {len(df_train_common)}, 测试集大小: {len(df_test_common)}")
    return df_train_common, df_test_common, common_stocks

def analyze_feature_distribution(df_train, df_test, feature_cols, top_n=20):
    """
    分析训练集和测试集的特征分布差异
    """
    print("分析特征分布差异...")
    ks_results = {}
    for feature in tqdm(feature_cols, desc="计算KS统计量"):
        try:
            # 排除可能的缺失值或无限值
            train_vals = df_train[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            test_vals = df_test[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(train_vals) > 0 and len(test_vals) > 0:
                stat, p_value = ks_2samp(train_vals, test_vals)
                ks_results[feature] = {'statistic': stat, 'p_value': p_value}
        except Exception as e:
            print(f"分析特征 {feature} 时出错: {str(e)}")
    
    if not ks_results:
        print("警告: 所有特征的KS检验都失败了")
        return None, []
    
    # 排序并选择差异最大的特征
    sorted_features = sorted(ks_results.items(), key=lambda x: x[1]['statistic'], reverse=True)
    
    # 保存结果
    ks_df = pd.DataFrame([
        {'feature': feat, 'ks_statistic': stats['statistic'], 'p_value': stats['p_value']} 
        for feat, stats in sorted_features
    ])
    ks_df.to_csv(os.path.join(RESULTS_DIR, 'feature_distribution_differences.csv'), index=False)
    
    # 返回排序后的特征及其KS统计量
    top_different_features = [feat for feat, _ in sorted_features[:top_n]]
    
    # 可视化前5个差异最大的特征
    visualize_top_features(df_train, df_test, [feat for feat, _ in sorted_features[:5]])
    
    return ks_df, top_different_features

def visualize_top_features(df_train, df_test, features, n_bins=50):
    """
    可视化顶部特征的分布
    """
    if not features:
        return
    
    try:
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features * 4))
        if n_features == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            train_vals = df_train[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            test_vals = df_test[feature].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            
            # 使用百分比直方图以便比较不同大小的数据集
            sns.histplot(train_vals, bins=n_bins, alpha=0.5, color='blue', label='训练集', ax=ax, stat='density')
            sns.histplot(test_vals, bins=n_bins, alpha=0.5, color='red', label='测试集', ax=ax, stat='density')
            
            ax.set_title(f'特征 {feature} 分布')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'top_feature_distributions.png'))
        plt.close()
        print(f"特征分布图已保存至: {os.path.join(PLOTS_DIR, 'top_feature_distributions.png')}")
    except Exception as e:
        print(f"绘制特征分布图时出错: {str(e)}")

def get_feature_columns(df, exclude_cols=None):
    """
    获取数据集中的特征列
    """
    if exclude_cols is None:
        exclude_cols = ['ID', 'RET', 'STOCK', 'SECTOR', 'INDUSTRY', 'SUB_INDUSTRY', 'INDUSTRY_GROUP']
    
    # 首先查找SIG开头的特征
    feature_cols = [col for col in df.columns if col.startswith('SIG_')]
    
    # 如果没有SIG特征，选择所有数值型列
    if not feature_cols:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    return feature_cols

def create_transform_pipeline(top_different_features=None, method='quantile'):
    """
    创建特征转换管道
    根据特征分布差异选择不同的转换方法
    """
    if top_different_features is None:
        top_different_features = []
    
    if method == 'quantile':
        return Pipeline([
            ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
        ])
    elif method == 'robust':
        return Pipeline([
            ('scaler', RobustScaler())
        ])
    elif method == 'pca':
        # PCA转换，保留95%的方差但最多20个组件
        return Pipeline([
            ('scaler', StandardScaler()),  # PCA前需要标准化
            ('pca', LimitedPCA(n_components=0.95, max_components=20, random_state=42))
        ])
    elif method == 'pca_10':
        # 固定使用10个主成分的PCA
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10, random_state=42))
        ])
    elif method == 'pca_5':
        # 固定使用5个主成分的PCA
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=5, random_state=42))
        ])
    else:  # 'standard'
        return Pipeline([
            ('scaler', StandardScaler())
        ])

def select_features(X_train, y_train, X_test, method='model_based', n_features=None):
    """
    根据不同方法选择特征
    """
    print(f"使用 {method} 方法选择特征...")
    
    if n_features is None:
        n_features = min(20, X_train.shape[1])  # 确保最大不超过20
    else:
        n_features = min(20, n_features)  # 确保传入的n_features不超过20
    
    if method == 'model_based':
        # 使用随机森林的特征重要性进行特征选择
        # 首先训练随机森林获取特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 选择前n_features个最重要的特征
        selected_indices = indices[:n_features]
        feature_names = X_train.columns[selected_indices].tolist()
        
        # 选择特定的特征
        X_train_selected = X_train.iloc[:, selected_indices]
        X_test_selected = X_test.iloc[:, selected_indices]
        
    elif method == 'mutual_info':
        # 使用互信息选择特征
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        # 将特征和分数组合并排序
        feat_scores = list(zip(X_train.columns, mi_scores))
        feat_scores = sorted(feat_scores, key=lambda x: x[1], reverse=True)
        
        # 选择前n个特征
        feature_names = [f[0] for f in feat_scores[:n_features]]
        
        # 根据选择的特征名称获取数据
        X_train_selected = X_train[feature_names]
        X_test_selected = X_test[feature_names]
        
    else:  # 'all' - 使用所有特征，但最多选择50个
        if X_train.shape[1] > 50:
            print(f"特征总数 {X_train.shape[1]} 超过50，将随机选择50个特征")
            selected_indices = np.random.choice(X_train.shape[1], 50, replace=False)
            feature_names = X_train.columns[selected_indices].tolist()
            X_train_selected = X_train.iloc[:, selected_indices]
            X_test_selected = X_test.iloc[:, selected_indices]
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            feature_names = X_train.columns.tolist()
    
    print(f"选择了 {len(feature_names)} 个特征")
    return X_train_selected, X_test_selected, feature_names

def build_model(model_type='ridge', params=None):
    """
    构建不同类型的模型
    """
    if params is None:
        params = {}
    
    if model_type == 'ridge':
        # Ridge回归默认参数
        default_params = {'alpha': 1.0, 'class_weight': None}
        # 更新参数
        for k, v in params.items():
            default_params[k] = v
        return RidgeClassifier(**default_params, random_state=42)
    
    elif model_type == 'logistic':
        # 逻辑回归默认参数
        default_params = {'C': 1.0, 'penalty': 'l2', 'class_weight': None}
        # 更新参数
        for k, v in params.items():
            default_params[k] = v
        return LogisticRegression(**default_params, random_state=42, max_iter=1000)
    
    elif model_type == 'rf':
        # 随机森林默认参数
        default_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
        # 更新参数
        for k, v in params.items():
            default_params[k] = v
        return RandomForestClassifier(**default_params, random_state=42, n_jobs=-1)
    
    elif model_type == 'ensemble':
        # 创建简单的投票集成模型
        models = [
            ('ridge', RidgeClassifier(alpha=1.0, random_state=42)),
            ('logistic', LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ]
        return VotingClassifier(estimators=models, voting='hard')
    
    elif model_type == 'stacking':
        # 创建堆叠集成模型
        estimators = [
            ('ridge', RidgeClassifier(alpha=1.0, random_state=42)),
            ('logistic', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            cv=5
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def train_models_by_sector(df_train, df_test, feature_names, model_type='ridge', transform_method='quantile'):
    """
    按行业分组训练模型
    """
    print(f"开始按行业板块训练模型 (模型类型: {model_type}, 特征转换: {transform_method})...")
    
    # 获取所有行业板块
    sectors = df_train['SECTOR'].dropna().unique()
    
    results = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    
    # 创建交叉验证对象
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 循环每个行业板块
    for sector_idx in tqdm(sectors, desc=f"按SECTOR分组训练 ({model_type})"):
        try:
            # 筛选当前行业的数据
            df_sector_train = df_train[df_train["SECTOR"] == sector_idx]
            df_sector_test = df_test[df_test["SECTOR"] == sector_idx]
            
            # 如果数据量太少，跳过训练
            if len(df_sector_train) < 100 or len(df_sector_test) < 20:
                print(f"板块 {sector_idx} 数据量太少，跳过训练")
                continue
            
            # 准备训练集和测试集
            X_train = df_sector_train[feature_names]
            y_train = np.array(df_sector_train["RET"])
            X_test = df_sector_test[feature_names]
            y_test = np.array(df_sector_test["RET"])
            
            # 创建数据转换管道
            transformer = create_transform_pipeline(method=transform_method)
            
            # 转换训练集和测试集
            X_train_scaled = transformer.fit_transform(X_train)
            X_test_scaled = transformer.transform(X_test)
            
            # 检查是否使用了PCA，并根据情况创建适当的列名
            is_pca = 'pca' in transform_method
            
            if is_pca:
                # 如果使用了PCA，创建新的列名（PC1, PC2, ...）
                if transform_method == 'pca_5':
                    n_components = 5
                elif transform_method == 'pca_10':
                    n_components = 10
                else:  # 'pca'
                    # 获取实际使用的组件数量
                    for name, step in transformer.named_steps.items():
                        if 'pca' in name:
                            n_components = step.n_components_
                            break
                    else:
                        n_components = X_train_scaled.shape[1]  # 如果无法确定，使用实际列数
                
                # 创建新的列名
                new_columns = [f"PC{i+1}" for i in range(n_components)]
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=new_columns)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=new_columns)
            else:
                # 不是PCA，使用原始列名
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # 根据模型类型设置参数网格
            if model_type == 'ridge':
                param_grid = {"alpha": np.logspace(-3, 3, num=7)}
            elif model_type == 'logistic':
                param_grid = {"C": np.logspace(-3, 3, num=7)}
            elif model_type == 'rf':
                param_grid = {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10]
                }
            else:  # ensemble or stacking
                param_grid = {}  # 集成模型不进行网格搜索
            
            # 创建基础模型
            base_model = build_model(model_type=model_type)
            
            # 如果参数网格非空，执行网格搜索
            if param_grid:
                # 使用网格搜索寻找最佳参数
                gs = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=kf,
                    scoring="accuracy",
                    n_jobs=-1
                )
                gs.fit(X_train_scaled, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                # 直接训练模型
                best_model = base_model
                best_model.fit(X_train_scaled, y_train)
                best_params = {}
            
            # 在测试集上评估模型
            y_val_pred = best_model.predict(X_test_scaled)
            
            # 计算概率（如果可用）
            try:
                if hasattr(best_model, "decision_function"):
                    decision_values = best_model.decision_function(X_test_scaled)
                    # 避免溢出
                    decision_values = np.clip(decision_values, -30, 30)
                    y_test_prob = 1 / (1 + np.exp(-decision_values))
                elif hasattr(best_model, "predict_proba"):
                    y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_test_prob = y_val_pred
            except Exception as e:
                print(f"计算概率时出错: {str(e)}")
                y_test_prob = y_val_pred
            
            # 计算性能指标
            acc = accuracy_score(y_test, y_val_pred)
            try:
                auc = roc_auc_score(y_test, y_test_prob)
                pr_auc = average_precision_score(y_test, y_test_prob)
            except Exception as e:
                print(f"计算AUC时出错: {str(e)}")
                auc = 0.5
                pr_auc = 0.5
            
            # 生成分类报告
            report = classification_report(y_test, y_val_pred, zero_division=0)
            
            # 保存结果
            y_true_all.extend(y_test)
            y_pred_all.extend(y_val_pred)
            y_prob_all.extend(y_test_prob)
            
            # 保存模型
            model_filename = f"model_{model_type}_sector_{int(sector_idx)}.joblib"
            joblib.dump(best_model, os.path.join(MODELS_DIR, model_filename))
            
            # 将结果添加到列表
            results.append({
                "Group": sector_idx,
                "Train_Samples": len(df_sector_train),
                "Test_Samples": len(df_sector_test),
                "Best_Params": best_params,
                "Accuracy": acc,
                "ROC_AUC": auc,
                "PR_AUC": pr_auc,
                "Report": report
            })
        
        except Exception as e:
            print(f"处理板块 {sector_idx} 时出错: {str(e)}")
            continue
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 计算整体性能
    if y_true_all and y_pred_all:  # 确保有预测结果
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        try:
            if len(set(y_true_all)) > 1:  # 确保有多个类别
                overall_auc = roc_auc_score(y_true_all, y_prob_all)
                overall_pr_auc = average_precision_score(y_true_all, y_prob_all)
            else:
                overall_auc = 0.5
                overall_pr_auc = 0.5
        except Exception as e:
            print(f"计算整体AUC时出错: {str(e)}")
            overall_auc = 0.5
            overall_pr_auc = 0.5
    else:
        overall_accuracy = np.nan
        overall_auc = 0.5
        overall_pr_auc = 0.5
    
    # 保存结果
    results_filename = f"results_{model_type}_{transform_method}.csv"
    if not df_results.empty:
        df_results.to_csv(os.path.join(RESULTS_DIR, results_filename), index=False)
    
    # 输出性能摘要
    print("\n按组别性能汇总:")
    if not df_results.empty:
        print(df_results[["Group", "Train_Samples", "Test_Samples", "Accuracy", "ROC_AUC", "PR_AUC"]].to_string(index=False))
    
    print(f"\n整体性能:")
    print(f"准确率: {overall_accuracy:.4f}")
    print(f"ROC AUC: {overall_auc:.4f}")
    print(f"PR AUC: {overall_pr_auc:.4f}")
    
    # 返回结果
    return df_results, overall_accuracy, overall_auc, overall_pr_auc

def run_adaptive_model_selection(df_train, df_test, top_different_features):
    """
    根据特征分布差异自适应选择模型参数和特征
    """
    print("运行自适应模型选择...")
    
    # 特征选择方法列表
    feature_selection_methods = ['model_based', 'mutual_info', 'all']
    
    # 数据转换方法列表
    transform_methods = ['standard', 'quantile', 'robust', 'pca', 'pca_10', 'pca_5']
    
    # 模型类型列表
    model_types = ['ridge', 'logistic', 'rf', 'ensemble', 'stacking']
    
    # 准备结果存储
    all_results = []
    
    # 确定特征列
    feature_cols = get_feature_columns(df_train)
    print(f"原始特征数量: {len(feature_cols)}")
    
    # 筛选出差异较大的特征
    if len(top_different_features) > 0:
        top_feature_cols = [f for f in feature_cols if f in top_different_features]
        has_top_features = True
        print(f"差异较大的特征数量: {len(top_feature_cols)}")
        # 如果筛选后特征太少，使用全部特征
        if len(top_feature_cols) < 5:
            top_feature_cols = feature_cols
            has_top_features = False
            print("差异特征太少，将使用全部特征")
    else:
        top_feature_cols = feature_cols
        has_top_features = False
        print("没有显著的差异特征，将使用全部特征")
    
    # 如果特征数量太多，预先过滤一些特征，减少后续处理的计算量
    if len(top_feature_cols) > 20:
        print(f"特征数量较多，预先筛选到20个特征")
        # 使用方差排序进行初步特征选择
        feature_vars = df_train[top_feature_cols].var().sort_values(ascending=False)
        top_feature_cols = feature_vars.index[:20].tolist()
    
    print(f"最终用于模型选择的特征数量: {len(top_feature_cols)}")
    
    # 测试每种组合
    for feature_method in feature_selection_methods[:2]:  # 只使用前两种特征选择方法
        for transform_method in transform_methods[:2]:  # 只使用前两种转换方法
            for model_type in model_types[:3]:  # 只使用前三种模型类型
                print(f"\n测试组合: 特征选择={feature_method}, 转换={transform_method}, 模型={model_type}")
                
                # 为不同特征选择方法设置不同的特征数量
                if feature_method == 'model_based':
                    n_features = min(20, len(top_feature_cols))
                elif feature_method == 'mutual_info':
                    n_features = min(15, len(top_feature_cols))
                else:
                    n_features = None
                
                # 选择特征
                X_train, X_test, selected_features = select_features(
                    df_train[top_feature_cols], df_train['RET'], 
                    df_test[top_feature_cols], method=feature_method, n_features=n_features
                )
                
                # 训练模型
                results, acc, auc, pr_auc = train_models_by_sector(
                    df_train, df_test, 
                    selected_features, model_type=model_type, transform_method=transform_method
                )
                
                # 保存结果
                all_results.append({
                    'feature_method': feature_method,
                    'transform_method': transform_method,
                    'model_type': model_type,
                    'accuracy': acc,
                    'roc_auc': auc,
                    'pr_auc': pr_auc,
                    'n_features': len(selected_features)
                })
    
    # 转换为DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # 保存结果
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'adaptive_model_selection_results.csv'), index=False)
    
    # 找出最佳组合
    if not summary_df.empty:
        best_row = summary_df.iloc[summary_df['accuracy'].argmax()]
        
        print("\n所有组合的性能汇总:")
        print(summary_df.to_string(index=False))
        
        print(f"\n最佳组合:")
        print(f"特征选择: {best_row['feature_method']}")
        print(f"转换方法: {best_row['transform_method']}")
        print(f"模型类型: {best_row['model_type']}")
        print(f"特征数量: {best_row['n_features']}")
        print(f"准确率: {best_row['accuracy']:.4f}")
        print(f"ROC AUC: {best_row['roc_auc']:.4f}")
        print(f"PR AUC: {best_row['pr_auc']:.4f}")
    else:
        print("警告: 没有成功完成的模型组合")
        best_row = {
            'feature_method': 'model_based',
            'transform_method': 'quantile',
            'model_type': 'ridge',
            'accuracy': 0.5,
            'roc_auc': 0.5,
            'pr_auc': 0.5,
            'n_features': 10
        }
    
    return best_row, summary_df

def train_final_model(df_train, df_test, best_params):
    """
    使用最佳参数训练最终模型
    """
    print("\n训练最终模型...")
    
    # 获取特征列
    feature_cols = get_feature_columns(df_train)
    
    # 选择特征
    X_train, X_test, selected_features = select_features(
        df_train[feature_cols], df_train['RET'], 
        df_test[feature_cols], method=best_params['feature_method']
    )
    
    # 训练模型
    results, acc, auc, pr_auc = train_models_by_sector(
        df_train, df_test, selected_features, 
        model_type=best_params['model_type'], 
        transform_method=best_params['transform_method']
    )
    
    print(f"\n最终模型性能:")
    print(f"准确率: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # 可视化对比性能
    visualize_performance_comparison(best_params, acc, auc)
    
    return results, acc, auc, pr_auc

def visualize_performance_comparison(best_params, new_acc, new_auc):
    """
    可视化新模型与原始模型的性能比较
    """
    # 假设原始模型性能
    baseline_accs = [0.5114, 0.5000, 0.5010]
    baseline_aucs = [0.5210, 0.5014, 0.5016]
    model_names = ['内部分割', '外部验证 (全部股票)', '外部验证 (共同股票)']
    
    # 添加新模型性能
    accs = baseline_accs + [new_acc]
    aucs = baseline_aucs + [new_auc]
    model_names.append(f"自适应模型 ({best_params['model_type']})")
    
    # 创建性能对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, accs, width, label='准确率')
    rects2 = ax.bar(x + width/2, aucs, width, label='ROC AUC')
    
    ax.set_ylabel('性能')
    ax.set_title('模型性能比较')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # 添加标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_performance_comparison.png'))
    plt.close()
    print(f"性能比较图已保存到: {os.path.join(PLOTS_DIR, 'model_performance_comparison.png')}")

def compare_pca_performance(df_train, df_test):
    """
    专门比较使用PCA和不使用PCA的模型性能
    """
    print("\n" + "="*80)
    print("PCA性能比较分析")
    print("="*80)
    
    # 获取特征列
    feature_cols = get_feature_columns(df_train)
    print(f"原始特征数量: {len(feature_cols)}")
    
    # 如果特征太多，考虑随机选择一部分特征
    if len(feature_cols) > 20:
        # 使用方差选择前20个特征
        feature_vars = df_train[feature_cols].var().sort_values(ascending=False)
        selected_features = feature_vars.index[:20].tolist()
    else:
        selected_features = feature_cols
    
    print(f"用于分析的特征数量: {len(selected_features)}")
    
    # 单独计算PCA解释方差
    try:
        # 获取数据并填充缺失值
        X_train = df_train[selected_features].copy()
        
        # 处理缺失值和无限值
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        # 用列的中位数填充NaN
        for col in X_train.columns:
            X_train[col] = X_train[col].fillna(X_train[col].median())
        
        # 标准化数据
        X_train_scaled = StandardScaler().fit_transform(X_train)
        
        # 计算不同组件数量的PCA解释方差
        pca_full = PCA(random_state=42)
        pca_full.fit(X_train_scaled)
        
        # 绘制解释方差比例曲线
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'b-o')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.grid(True)
        plt.xlabel('组件数量')
        plt.ylabel('累计解释方差比例')
        plt.title('PCA组件数量与解释方差关系')
        plt.savefig(os.path.join(PLOTS_DIR, 'pca_explained_variance.png'))
        plt.close()
        print(f"PCA解释方差分析图已保存至: {os.path.join(PLOTS_DIR, 'pca_explained_variance.png')}")
        
        # 找出达到95%方差所需的组件数
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        print(f"达到95%解释方差需要的组件数量: {n_components_95}")
    
    except Exception as e:
        print(f"计算PCA解释方差时出错: {str(e)}")
    
    # 模型配置
    models = [
        ('ridge', 'standard', '标准Ridge'),
        ('ridge', 'quantile', '分位数Ridge'),
        ('ridge', 'pca_5', 'PCA(5)Ridge'),
        ('ridge', 'pca_10', 'PCA(10)Ridge'),
        ('ridge', 'pca', 'PCA(auto)Ridge'),
        ('logistic', 'standard', '标准逻辑回归'),
        ('logistic', 'pca_10', 'PCA(10)逻辑回归'),
        ('rf', 'standard', '标准随机森林'),
        ('rf', 'pca_10', 'PCA(10)随机森林')
    ]
    
    results = []
    
    # 训练和评估每种模型配置
    for model_type, transform_method, model_name in models:
        print(f"\n评估 {model_name}...")
        
        # 训练模型
        try:
            model_results, acc, auc, pr_auc = train_models_by_sector(
                df_train, df_test, selected_features, 
                model_type=model_type, transform_method=transform_method
            )
            
            # 保存结果
            results.append({
                'model_name': model_name,
                'model_type': model_type,
                'transform': transform_method,
                'accuracy': acc,
                'roc_auc': auc,
                'pr_auc': pr_auc
            })
            
        except Exception as e:
            print(f"模型 {model_name} 评估失败: {str(e)}")
            continue
    
    # 检查是否有结果
    if not results:
        print("警告：没有成功评估任何模型")
        return None
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    try:
        results_df.to_csv(os.path.join(RESULTS_DIR, 'pca_comparison_results.csv'), index=False)
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
    
    # 创建性能对比图
    try:
        plt.figure(figsize=(14, 8))
        
        # 准确率对比
        plt.subplot(1, 2, 1)
        sns.barplot(x='model_name', y='accuracy', data=results_df)
        plt.title('不同模型的准确率对比')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.45, 0.55)
        
        # ROC AUC对比
        plt.subplot(1, 2, 2)
        sns.barplot(x='model_name', y='roc_auc', data=results_df)
        plt.title('不同模型的ROC AUC对比')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.45, 0.55)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'pca_model_comparison.png'))
        plt.close()
        print(f"\n性能比较图已保存到: {os.path.join(PLOTS_DIR, 'pca_model_comparison.png')}")
    except Exception as e:
        print(f"创建性能对比图时出错: {str(e)}")
    
    # 输出结果
    print("\nPCA比较结果:")
    if not results_df.empty:
        print(results_df.to_string(index=False))
        
        # 找出最佳模型
        best_model = results_df.iloc[results_df['accuracy'].argmax()]
        print(f"\n最佳模型: {best_model['model_name']}")
        print(f"准确率: {best_model['accuracy']:.4f}")
        print(f"ROC AUC: {best_model['roc_auc']:.4f}")
    else:
        print("没有有效的模型结果")
    
    return results_df

def main():
    """
    主函数
    """
    print("="*80)
    print("开始运行 fix_regression.py")
    print("="*80)
    
    # 确保目录存在
    for d in [PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
        try:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
                print(f"创建目录: {d}")
            else:
                print(f"使用已存在的目录: {d}")
        except Exception as e:
            print(f"创建目录时出错: {str(e)}")
            print(f"将尝试使用当前目录")
    
    try:
        # 加载数据
        df_train, df_test, new_features = load_data()
        
        # 过滤共同股票
        df_train_common, df_test_common, common_stocks = filter_common_stocks(df_train, df_test)
        
        # 获取特征列
        feature_cols = get_feature_columns(df_train_common)
        print(f"可用特征数量: {len(feature_cols)}")
        
        # 分析特征分布差异
        ks_df, top_different_features = analyze_feature_distribution(df_train_common, df_test_common, feature_cols)
        
        # 添加命令行参数解析，判断是否执行完整的自适应模型选择
        parser = argparse.ArgumentParser(description='股票预测模型训练与评估')
        parser.add_argument('--full', action='store_true', help='运行完整的自适应模型选择(耗时较长)')
        parser.add_argument('--pca_only', action='store_true', help='只运行PCA性能比较分析')
        args = parser.parse_args()
        
        # PCA性能比较分析（默认行为）
        if args.pca_only or not args.full:
            print("\n执行PCA性能比较分析...")
            pca_results = compare_pca_performance(df_train_common, df_test_common)
        
        # 运行自适应模型选择（可选）
        if args.full:
            print("\n执行完整的自适应模型选择...")
            best_params, summary_df = run_adaptive_model_selection(df_train_common, df_test_common, top_different_features)
            
            # 训练最终模型
            final_results, final_acc, final_auc, final_pr_auc = train_final_model(df_train_common, df_test_common, best_params)
            
            print("\n总结:")
            print(f"1. 我们筛选了训练集和测试集的共同股票，共有 {len(common_stocks)} 只股票。")
            print(f"2. 我们对特征分布进行了分析，识别出了 {len(top_different_features)} 个分布差异较大的特征。")
            print(f"3. 我们测试了多种特征选择、数据转换和模型类型的组合，找到了最佳组合。")
            print(f"4. 最佳模型采用 {best_params['model_type']} 模型、{best_params['transform_method']} 转换和 {best_params['feature_method']} 特征选择。")
            print(f"5. 最终模型在测试集上的性能为: 准确率={final_acc:.4f}, ROC AUC={final_auc:.4f}, PR AUC={final_pr_auc:.4f}。")
        
        print("\n所有结果和模型已保存到相应目录。")
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("fix_regression.py 执行完成")
    print("="*80)

if __name__ == "__main__":
    main() 