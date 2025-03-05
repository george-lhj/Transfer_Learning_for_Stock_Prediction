# 添加各种神经网络模型类和传统机器学习模型集成

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import seaborn as sns
import gc
import os
from collections import defaultdict
import matplotlib.font_manager as fm

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像负号'-'显示为方块的问题

# 设置随机种子以获得可重复的结果
torch.manual_seed(42)
np.random.seed(42)

class StockLSTM(nn.Module):
    """使用LSTM进行股票预测的神经网络模型"""
    def __init__(self, input_size, sector_num, industry_num, embedding_dim=8, hidden_size=128, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()
        
        # 类别嵌入层
        self.sector_embedding = nn.Embedding(sector_num+1, embedding_dim)
        self.industry_embedding = nn.Embedding(industry_num+1, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 1),  # 双向LSTM，维度翻倍
            nn.Tanh()
        )
        
        # 特征融合层
        combined_size = hidden_size*2 + embedding_dim*2
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sector_ids, industry_ids):
        batch_size = x.size(0)
        
        # 检查输入维度，确保是3D格式 [batch, sequence, features]
        if x.dim() == 3:
            # 输入已经是3D，不需要额外处理
            pass
        elif x.dim() == 2:
            # 如果是2D输入 [batch, features]，转换为3D
            x = x.unsqueeze(2)  # [batch, features, 1]
            x = x.transpose(1, 2)  # [batch, 1, features]
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # [batch, sequence, hidden_size*2]
        
        # 应用注意力机制
        attn_weights = self.attention(lstm_out).squeeze(2)  # [batch, sequence]
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)  # [batch, sequence, 1]
        
        # 加权求和
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_size*2]
        
        # 类别嵌入
        sector_emb = self.sector_embedding(sector_ids)  # [batch, embedding_dim]
        industry_emb = self.industry_embedding(industry_ids)  # [batch, embedding_dim]
        
        # 特征融合
        combined = torch.cat([context_vector, sector_emb, industry_emb], dim=1)
        
        # 最终预测
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

class StockTransformer(nn.Module):
    """使用Transformer架构进行股票预测的神经网络模型"""
    def __init__(self, input_size, sector_num, industry_num, embedding_dim=8, hidden_size=64, 
                 num_heads=4, num_layers=2, dropout=0.2):
        super(StockTransformer, self).__init__()
        
        # 特征维度必须能被注意力头数整除
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        
        # 类别嵌入层
        self.sector_embedding = nn.Embedding(sector_num+1, embedding_dim)
        self.industry_embedding = nn.Embedding(industry_num+1, embedding_dim)
        
        # 输入投影层 - 修改为使用正确的input_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 特征融合层
        combined_size = hidden_size + embedding_dim*2
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sector_ids, industry_ids):
        # 检查输入维度，确保是3D格式 [batch, sequence, features]
        if x.dim() == 4:
            # 如果是4D输入 [batch, sequence, time, features]
            batch_size, seq_len, time_steps, features = x.size()
            # 重塑为3D [batch, sequence*time, features]
            x = x.view(batch_size, seq_len*time_steps, features)
        elif x.dim() == 3:
            # 输入已经是3D [batch, sequence, features]，无需调整
            pass
        
        # 输入投影
        x = self.input_proj(x)  # [batch, sequence, hidden_size]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        # 注意：不需要mask，因为我们希望每个位置都能看到所有其他位置
        transformer_out = self.transformer_encoder(x)  # [batch, sequence, hidden_size]
        
        # 使用平均池化汇总所有位置的信息
        context_vector = torch.mean(transformer_out, dim=1)  # [batch, hidden_size]
        
        # 类别嵌入
        sector_emb = self.sector_embedding(sector_ids)
        industry_emb = self.industry_embedding(industry_ids)
        
        # 特征融合
        combined = torch.cat([context_vector, sector_emb, industry_emb], dim=1)
        
        # 最终预测
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class StockCNN(nn.Module):
    """使用CNN进行股票预测的神经网络模型"""
    def __init__(self, input_size, sector_num, industry_num, embedding_dim=8, filters=[64, 128, 64], 
                 kernel_sizes=[3, 5, 7], dropout=0.3):
        super(StockCNN, self).__init__()
        
        # 类别嵌入层
        self.sector_embedding = nn.Embedding(sector_num+1, embedding_dim)
        self.industry_embedding = nn.Embedding(industry_num+1, embedding_dim)
        
        # 1D卷积层
        self.conv_layers = nn.ModuleList()
        for i, (filter_size, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            # 为第一层设置正确的输入通道
            in_channels = 1 if i == 0 else filters[i-1]
            
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filter_size, kernel_size, padding='same'),
                    nn.BatchNorm1d(filter_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征融合层
        combined_size = filters[-1] + embedding_dim*2
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sector_ids, industry_ids):
        # 调整x的形状为[batch, channel, feature]用于1D卷积
        x = x.unsqueeze(1)  # [batch, 1, feature_num]
        
        # 应用卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 全局池化
        x = self.global_avg_pool(x).squeeze(2)  # [batch, filters[-1]]
        
        # 类别嵌入
        sector_emb = self.sector_embedding(sector_ids)
        industry_emb = self.industry_embedding(industry_ids)
        
        # 特征融合
        combined = torch.cat([x, sector_emb, industry_emb], dim=1)
        
        # 最终预测
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

class CategoryEmbeddingMLP(nn.Module):
    """具有类别嵌入功能的MLP模型，用于股票预测"""
    def __init__(self, input_size, sector_num, industry_num, embedding_dim=8, 
                 hidden_sizes=[128, 64, 32], dropout=0.3):
        super(CategoryEmbeddingMLP, self).__init__()
        
        # 类别嵌入层
        self.sector_embedding = nn.Embedding(sector_num+1, embedding_dim)
        self.industry_embedding = nn.Embedding(industry_num+1, embedding_dim)
        
        # Signature特征处理分支
        self.signature_branch = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 主要网络层
        layers = []
        input_dim = hidden_sizes[0] + embedding_dim*2  # Signature特征 + 行业嵌入
        
        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.main_network = nn.Sequential(*layers)
        
    def forward(self, x, sector_ids, industry_ids):
        # 处理Signature特征
        signature_feats = self.signature_branch(x)
        
        # 行业类别嵌入
        sector_emb = self.sector_embedding(sector_ids)
        industry_emb = self.industry_embedding(industry_ids)
        
        # 特征融合
        combined = torch.cat([signature_feats, sector_emb, industry_emb], dim=1)
        
        # 输出预测
        return self.main_network(combined)

class BaggingStockModel:
    """集成多个基础模型进行股票预测的Bagging模型"""
    def __init__(self, base_model_class, n_estimators=10, sample_ratio=0.8, feature_ratio=0.8, **model_kwargs):
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.model_kwargs = model_kwargs
        self.models = []
        self.feature_indices_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, train_loader, val_loader=None, epochs=20, lr=0.001, save_dir='./results/bagging'):
        """训练Bagging集成模型"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取原始特征数量
        for batch in train_loader:
            inputs, _, _, _ = batch
            input_size = inputs.shape[1]
            break
            
        # 获取类别数量
        sector_num = self.model_kwargs.get('sector_num', 0)
        industry_num = self.model_kwargs.get('industry_num', 0)
        
        for i in range(self.n_estimators):
            print(f"\n训练Bagging模型 {i+1}/{self.n_estimators}")
            
            # 随机选择特征子集
            num_features = max(1, int(input_size * self.feature_ratio))
            feature_indices = torch.randperm(input_size)[:num_features]
            self.feature_indices_list.append(feature_indices)
            
            # 创建bootstrap采样的数据加载器
            bootstrap_train_loader = self._create_bootstrap_loader(train_loader, feature_indices)
            
            # 创建模型实例
            model = self.base_model_class(
                input_size=num_features,
                sector_num=sector_num,
                industry_num=industry_num,
                **{k: v for k, v in self.model_kwargs.items() if k not in ['sector_num', 'industry_num']}
            ).to(self.device)
            
            # 训练模型
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            
            # 早停机制
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 5
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0
                train_batches = 0
                
                for inputs, targets, sector_ids, industry_ids in bootstrap_train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    sector_ids, industry_ids = sector_ids.to(self.device), industry_ids.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs, sector_ids, industry_ids)
                    loss = focal_loss(outputs, targets, gamma=2.0, alpha=0.25)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = train_loss / train_batches
                
                # 验证阶段（如果有验证集）
                if val_loader:
                    val_loss, val_acc = self._evaluate_model(model, val_loader, feature_indices)
                    
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # 早停检查
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        torch.save(model.state_dict(), f"{save_dir}/model_{i+1}_best.pth")
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            print(f"早停：{patience_counter}个epoch没有改善")
                            break
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            
            # 如果没有验证集或者没有触发早停，保存最后的模型
            if not val_loader or patience_counter < max_patience:
                torch.save(model.state_dict(), f"{save_dir}/model_{i+1}_final.pth")
            
            # 加载最佳模型（如果有验证集）
            if val_loader:
                model.load_state_dict(torch.load(f"{save_dir}/model_{i+1}_best.pth"))
            
            self.models.append(model)
        
        return self
    
    def _create_bootstrap_loader(self, data_loader, feature_indices):
        """创建bootstrap采样的数据加载器"""
        all_data = []
        for inputs, targets, sector_ids, industry_ids in data_loader:
            # 只保留所选特征
            inputs_subset = inputs[:, feature_indices]
            all_data.append((inputs_subset, targets, sector_ids, industry_ids))
        
        # 创建bootstrap样本
        bootstrap_indices = torch.randint(
            low=0, high=len(all_data), 
            size=(int(len(all_data) * self.sample_ratio),)
        )
        bootstrap_data = [all_data[i] for i in bootstrap_indices]
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
        
        # 创建数据加载器
        return torch.utils.data.DataLoader(
            SimpleDataset(bootstrap_data),
            batch_size=data_loader.batch_size,
            shuffle=True
        )
    
    def _evaluate_model(self, model, data_loader, feature_indices):
        """评估单个模型"""
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        batches = 0
        
        with torch.no_grad():
            for inputs, targets, sector_ids, industry_ids in data_loader:
                # 只使用选择的特征
                inputs_subset = inputs[:, feature_indices]
                
                inputs_subset, targets = inputs_subset.to(self.device), targets.to(self.device)
                sector_ids, industry_ids = sector_ids.to(self.device), industry_ids.to(self.device)
                
                outputs = model(inputs_subset, sector_ids, industry_ids)
                loss = nn.BCELoss()(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                batches += 1
        
        val_loss /= batches
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        
        binary_preds = (val_preds > 0.5).astype(int)
        accuracy = accuracy_score(val_targets, binary_preds)
        
        return val_loss, accuracy
    
    def predict_proba(self, test_loader):
        """获取所有模型的概率预测平均值"""
        all_model_probs = []
        
        for i, model in enumerate(self.models):
            model.eval()
            feature_indices = self.feature_indices_list[i]
            model_probs = []
            
            with torch.no_grad():
                for inputs, _, sector_ids, industry_ids in test_loader:
                    # 只使用选择的特征
                    inputs_subset = inputs[:, feature_indices].to(self.device)
                    sector_ids = sector_ids.to(self.device)
                    industry_ids = industry_ids.to(self.device)
                    
                    outputs = model(inputs_subset, sector_ids, industry_ids)
                    model_probs.extend(outputs.cpu().numpy().flatten())
            
            all_model_probs.append(model_probs)
        
        # 平均所有模型的预测
        avg_probs = np.mean(all_model_probs, axis=0)
        return avg_probs
    
    def predict(self, test_loader, threshold=0.5):
        """获取二分类预测"""
        probs = self.predict_proba(test_loader)
        return (probs > threshold).astype(int)
    
    def evaluate(self, test_loader):
        """评估模型性能"""
        # 获取预测概率
        probs = self.predict_proba(test_loader)
        preds = (probs > 0.5).astype(int)
        
        # 获取真实标签
        all_targets = []
        for _, targets, _, _ in test_loader:
            all_targets.extend(targets.numpy().flatten())
        
        all_targets = np.array(all_targets)
        
        # 计算指标
        accuracy = accuracy_score(all_targets, preds)
        roc_auc = roc_auc_score(all_targets, probs)
        
        # 计算PR AUC
        precision, recall, _ = precision_recall_curve(all_targets, probs)
        pr_auc = auc(recall, precision)
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, preds)
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(all_targets, probs)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征 (ROC) 曲线')
        plt.legend()
        plt.savefig('./results/bagging/roc_curve.png')
        plt.close()
        
        # 绘制PR曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR曲线 (AUC = {pr_auc:.3f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.savefig('./results/bagging/pr_curve.png')
        plt.close()
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        plt.title('混淆矩阵')
        plt.savefig('./results/bagging/confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm
        }

def prepare_sequence_data(df_train, df_test, feature_cols, seq_length=10, stride=1):
    """准备序列数据，适用于LSTM和Transformer模型"""
    print("准备序列数据...")
    
    # 创建序列样本
    def create_sequences(df, feature_cols, seq_length, stride):
        data = df[feature_cols].values
        
        sequences = []
        targets = []
        sector_ids = []
        industry_ids = []
        
        for i in range(0, len(data) - seq_length + 1, stride):
            sequences.append(data[i:i+seq_length])
            # 使用序列末尾的目标
            targets.append(1 if df['RET'].iloc[i+seq_length-1] > 0 else 0)
            # 使用序列末尾的类别信息
            sector_ids.append(df['SECTOR'].iloc[i+seq_length-1])
            industry_ids.append(df['INDUSTRY'].iloc[i+seq_length-1])
        
        return np.array(sequences), np.array(targets), sector_ids, industry_ids
    
    # 创建训练和测试序列
    X_train_seq, y_train_seq, train_sector_ids, train_industry_ids = create_sequences(
        df_train, feature_cols, seq_length, stride)
    
    X_test_seq, y_test_seq, test_sector_ids, test_industry_ids = create_sequences(
        df_test, feature_cols, seq_length, stride)
    
    # 标准化特征
    # 对每个特征维度分别进行标准化
    mean = np.mean(X_train_seq.reshape(-1, X_train_seq.shape[-1]), axis=0)
    std = np.std(X_train_seq.reshape(-1, X_train_seq.shape[-1]), axis=0)
    X_train_seq = (X_train_seq - mean) / (std + 1e-8)
    X_test_seq = (X_test_seq - mean) / (std + 1e-8)
    
    # 对类别特征进行编码
    sector_encoder = LabelEncoder()
    industry_encoder = LabelEncoder()
    
    # 合并训练和测试集的类别，以确保编码一致
    all_sectors = train_sector_ids + test_sector_ids
    all_industries = train_industry_ids + test_industry_ids
    
    sector_encoder.fit([str(s) if s is not None else "unknown" for s in all_sectors])
    industry_encoder.fit([str(i) if i is not None else "unknown" for i in all_industries])
    
    train_sector_encoded = sector_encoder.transform(
        [str(s) if s is not None else "unknown" for s in train_sector_ids])
    train_industry_encoded = industry_encoder.transform(
        [str(i) if i is not None else "unknown" for i in train_industry_ids])
    
    test_sector_encoded = sector_encoder.transform(
        [str(s) if s is not None else "unknown" for s in test_sector_ids])
    test_industry_encoded = industry_encoder.transform(
        [str(i) if i is not None else "unknown" for i in test_industry_ids])
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq.reshape(-1, 1))
    train_sector_tensor = torch.LongTensor(train_sector_encoded)
    train_industry_tensor = torch.LongTensor(train_industry_encoded)
    
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq.reshape(-1, 1))
    test_sector_tensor = torch.LongTensor(test_sector_encoded)
    test_industry_tensor = torch.LongTensor(test_industry_encoded)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_sector_tensor, train_industry_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_sector_tensor, test_industry_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    category_info = {
        'sector_size': len(sector_encoder.classes_),
        'industry_size': len(industry_encoder.classes_),
        'sector_names': dict(enumerate(sector_encoder.classes_)),
        'industry_names': dict(enumerate(industry_encoder.classes_))
    }
    
    return train_loader, test_loader, category_info, (mean, std)

def prepare_data_with_categories(df_train, df_test, feature_cols, batch_size=64):
    """准备带有类别特征的数据，适用于MLP和CNN模型"""
    print("准备带有类别特征的数据...")
    
    # 提取特征和目标
    X_train = df_train[feature_cols].values
    y_train = (df_train['RET'] > 0).astype(int).values
    
    X_test = df_test[feature_cols].values
    y_test = (df_test['RET'] > 0).astype(int).values
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 处理SECTOR和INDUSTRY分类特征
    sector_encoder = LabelEncoder()
    industry_encoder = LabelEncoder()
    
    # 获取分类特征
    train_sectors = df_train['SECTOR'].fillna('unknown').astype(str).values
    train_industries = df_train['INDUSTRY'].fillna('unknown').astype(str).values
    
    test_sectors = df_test['SECTOR'].fillna('unknown').astype(str).values
    test_industries = df_test['INDUSTRY'].fillna('unknown').astype(str).values
    
    # 合并训练和测试数据进行编码，以确保一致性
    all_sectors = np.concatenate([train_sectors, test_sectors])
    all_industries = np.concatenate([train_industries, test_industries])
    
    sector_encoder.fit(all_sectors)
    industry_encoder.fit(all_industries)
    
    train_sector_ids = sector_encoder.transform(train_sectors)
    train_industry_ids = industry_encoder.transform(train_industries)
    
    test_sector_ids = sector_encoder.transform(test_sectors)
    test_industry_ids = industry_encoder.transform(test_industries)
    
    # 创建PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    train_sector_tensor = torch.LongTensor(train_sector_ids)
    train_industry_tensor = torch.LongTensor(train_industry_ids)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    test_sector_tensor = torch.LongTensor(test_sector_ids)
    test_industry_tensor = torch.LongTensor(test_industry_ids)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_sector_tensor, train_industry_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_sector_tensor, test_industry_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
    
    # 返回编码器信息，用于模型创建
    encoders = {
        'sector_size': len(sector_encoder.classes_),
        'industry_size': len(industry_encoder.classes_),
        'sector_names': dict(enumerate(sector_encoder.classes_)),
        'industry_names': dict(enumerate(industry_encoder.classes_))
    }
    
    return train_loader, test_loader, scaler, encoders

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """计算focal loss，用于处理不平衡分类问题"""
    bce_loss = nn.BCELoss(reduction='none')(pred, target)
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    f_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()

def train_model_with_categories(model, train_loader, val_loader=None, epochs=20, lr=0.001, patience=10, device=None):
    """训练模型（包括类别特征）并使用早停"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_batches = 0
        
        for inputs, targets, sector_ids, industry_ids in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            sector_ids, industry_ids = sector_ids.to(device), industry_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, sector_ids, industry_ids)
            loss = focal_loss(outputs, targets, gamma=2.0, alpha=0.25)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / train_batches
        
        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets, sector_ids, industry_ids in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    sector_ids, industry_ids = sector_ids.to(device), industry_ids.to(device)
                    
                    outputs = model(inputs, sector_ids, industry_ids)
                    loss = focal_loss(outputs, targets, gamma=2.0, alpha=0.25)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            
            print(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
            
            # 早停检查
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience}个epoch验证损失没有改善")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}")
    
    # 如果有最佳模型状态，则加载
    if best_model_state and val_loader:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model_with_categories(model, test_loader, device=None):
    """评估模型性能（包括计算ROC AUC和混淆矩阵）"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, sector_ids, industry_ids in test_loader:
            inputs = inputs.to(device)
            sector_ids, industry_ids = sector_ids.to(device), industry_ids.to(device)
            
            outputs = model(inputs, sector_ids, industry_ids)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # 计算预测类别
    binary_preds = (all_preds > 0.5).astype(int)
    
    # 计算性能指标
    accuracy = accuracy_score(all_targets, binary_preds)
    roc_auc = roc_auc_score(all_targets, all_preds)
    
    # 计算精确率-召回率曲线下的面积
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, binary_preds)
    
    print(f"准确率: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print("混淆矩阵:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'predictions': all_preds,
        'targets': all_targets,
        'confusion_matrix': cm
    }

def select_optimal_features(df, all_features, importance_threshold=0.005, n_estimators=200):
    """使用RandomForest选择最重要的特征"""
    print("选择最优特征...")
    
    # 提取特征和目标
    X = df[all_features].values
    y = (df['RET'] > 0).astype(int).values
    
    # 创建和训练RandomForest模型
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    
    rf.fit(X, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # 选择重要性大于阈值的特征
    selected_features = importance_df[importance_df['Importance'] > importance_threshold]['Feature'].tolist()
    
    # 确保至少选择10个特征
    if len(selected_features) < 10:
        selected_features = importance_df.nlargest(10, 'Importance')['Feature'].tolist()
    
    print(f"选择了 {len(selected_features)}/{len(all_features)} 个特征")
    
    return selected_features, importance_df

def train_traditional_models(df_train, df_test, feature_cols):
    """训练传统机器学习模型(RandomForest, LightGBM)并比较性能"""
    from sklearn.ensemble import RandomForestClassifier
    try:
        import lightgbm as lgb
        lightgbm_available = True
    except ImportError:
        print("LightGBM未安装，跳过LightGBM模型训练")
        lightgbm_available = False
    
    print("训练传统机器学习模型...")
    
    # 创建结果目录
    os.makedirs('./results/traditional_models', exist_ok=True)
    
    # 提取特征和目标
    X_train = df_train[feature_cols].values
    y_train = (df_train['RET'] > 0).astype(int).values
    
    X_test = df_test[feature_cols].values
    y_test = (df_test['RET'] > 0).astype(int).values
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练RandomForest模型
    print("训练RandomForest模型...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # 获取RandomForest预测
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_preds = rf_model.predict(X_test_scaled)
    
    # 计算RandomForest指标
    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_roc_auc = roc_auc_score(y_test, rf_probs)
    
    # 计算PR AUC
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)
    rf_pr_auc = auc(recall_rf, precision_rf)
    
    print(f"RandomForest - 准确率: {rf_accuracy:.4f}, ROC AUC: {rf_roc_auc:.4f}, PR AUC: {rf_pr_auc:.4f}")
    
    # 训练LightGBM模型（如果可用）
    lgb_accuracy = lgb_roc_auc = lgb_pr_auc = 0
    if lightgbm_available:
        print("训练LightGBM模型...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42
        )
        
        # 修改这部分，适应不同版本的LightGBM API
        try:
            # 尝试使用新版本callbacks API
            eval_set = [(X_test_scaled, y_test)]
            lgb_model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                eval_metric='auc',
                callbacks=[lgb.early_stopping(30, verbose=True)]
            )
        except (TypeError, AttributeError):
            try:
                # 尝试使用callback列表方式
                eval_set = [(X_test_scaled, y_test)]
                callbacks = [lgb.callback.early_stopping(stopping_rounds=30, verbose=True)]
                lgb_model.fit(
                    X_train_scaled, y_train,
                    eval_set=eval_set,
                    eval_metric='auc',
                    callbacks=callbacks
                )
            except (TypeError, AttributeError):
                try:
                    # 尝试使用旧版参数方式
                    lgb_model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_test_scaled, y_test)],
                        eval_metric='auc',
                        early_stopping_rounds=30,
                        verbose=100
                    )
                except TypeError:
                    try:
                        # 最后尝试不使用早停
                        print("LightGBM版本不支持早停，使用基本训练...")
                        lgb_model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_test_scaled, y_test)],
                            eval_metric='auc',
                            verbose=100
                        )
                    except Exception as e:
                        print(f"LightGBM训练失败: {str(e)}")
                        print("跳过LightGBM模型")
                        lightgbm_available = False
        
        if lightgbm_available:
            # 获取LightGBM预测
            lgb_probs = lgb_model.predict_proba(X_test_scaled)[:, 1]
            lgb_preds = lgb_model.predict(X_test_scaled)
            
            # 计算LightGBM指标
            lgb_accuracy = accuracy_score(y_test, lgb_preds)
            lgb_roc_auc = roc_auc_score(y_test, lgb_probs)
            
            # 计算PR AUC
            precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, lgb_probs)
            lgb_pr_auc = auc(recall_lgb, precision_lgb)
            
            # 可视化LightGBM特征重要性
            lgb_importance = lgb_model.feature_importances_
            lgb_importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': lgb_importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.bar(lgb_importance_df['Feature'].head(20), lgb_importance_df['Importance'].head(20))
            plt.xticks(rotation=90)
            plt.xlabel('特征')
            plt.ylabel('重要性')
            plt.title('LightGBM特征重要性（Top 20）')
            plt.tight_layout()
            plt.savefig('./results/traditional_models/lgb_feature_importance.png')
            plt.close()
    
    # 可视化RandomForest特征重要性
    rf_importance = rf_model.feature_importances_
    rf_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(rf_importance_df['Feature'].head(20), rf_importance_df['Importance'].head(20))
    plt.xticks(rotation=90)
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.title('RandomForest特征重要性（Top 20）')
    plt.tight_layout()
    plt.savefig('./results/traditional_models/rf_feature_importance.png')
    plt.close()
    
    # 对比ROC曲线
    plt.figure(figsize=(10, 6))
    
    # RandomForest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {rf_roc_auc:.3f})')
    
    # LightGBM ROC（如果可用）
    if lightgbm_available:
        fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_probs)
        plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {lgb_roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('传统模型ROC曲线对比')
    plt.legend()
    plt.savefig('./results/traditional_models/roc_comparison.png')
    plt.close()
    
    # 对比PR曲线
    plt.figure(figsize=(10, 6))
    
    # RandomForest PR
    plt.plot(recall_rf, precision_rf, label=f'RandomForest (PR AUC = {rf_pr_auc:.3f})')
    
    # LightGBM PR（如果可用）
    if lightgbm_available:
        plt.plot(recall_lgb, precision_lgb, label=f'LightGBM (PR AUC = {lgb_pr_auc:.3f})')
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('传统模型PR曲线对比')
    plt.legend()
    plt.savefig('./results/traditional_models/pr_comparison.png')
    plt.close()
    
    # 返回模型和结果
    models = {'RandomForest': rf_model}
    results = {
        'RandomForest': {
            'accuracy': rf_accuracy,
            'roc_auc': rf_roc_auc,
            'pr_auc': rf_pr_auc
        }
    }
    
    if lightgbm_available:
        models['LightGBM'] = lgb_model
        results['LightGBM'] = {
            'accuracy': lgb_accuracy,
            'roc_auc': lgb_roc_auc,
            'pr_auc': lgb_pr_auc
        }
    
    # 保存特征重要性
    rf_importance_df.to_csv('./results/traditional_models/rf_feature_importance.csv', index=False)
    if lightgbm_available:
        lgb_importance_df.to_csv('./results/traditional_models/lgb_feature_importance.csv', index=False)
    
    return models, results, scaler

def compare_models(df_train, df_test, feature_cols):
    """比较不同神经网络模型和传统模型的性能"""
    import torch.nn.functional as F
    import math
    
    os.makedirs('./results/model_comparison', exist_ok=True)
    
    # 使用特征选择获取最佳特征
    selected_features, _ = select_optimal_features(df_train, feature_cols)
    
    # 准备标准数据（用于MLP和CNN）
    train_loader, test_loader, _, category_info = prepare_data_with_categories(
        df_train, df_test, selected_features)
    
    # 准备序列数据（用于LSTM和Transformer）
    seq_train_loader, seq_test_loader, seq_category_info, _ = prepare_sequence_data(
        df_train, df_test, selected_features, seq_length=10, stride=5)
    
    # 训练传统模型
    _, traditional_results, _ = train_traditional_models(df_train, df_test, selected_features)
    
    # 定义要比较的模型
    models_to_compare = {
        'MLP': {
            'class': CategoryEmbeddingMLP,
            'args': {
                'input_size': len(selected_features),
                'sector_num': category_info['sector_size'],
                'industry_num': category_info['industry_size'],
                'embedding_dim': 8
            },
            'train_loader': train_loader,
            'test_loader': test_loader
        },
        'LSTM': {
            'class': StockLSTM,
            'args': {
                'input_size': len(selected_features),
                'sector_num': seq_category_info['sector_size'],
                'industry_num': seq_category_info['industry_size'],
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3
            },
            'train_loader': seq_train_loader,
            'test_loader': seq_test_loader
        },
        'Transformer': {
            'class': StockTransformer,
            'args': {
                'input_size': len(selected_features),  # 修改: 使用特征数量
                'sector_num': seq_category_info['sector_size'],
                'industry_num': seq_category_info['industry_size'],
                'hidden_size': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.2
            },
            'train_loader': seq_train_loader,
            'test_loader': seq_test_loader
        },
        'CNN': {
            'class': StockCNN,
            'args': {
                'input_size': len(selected_features),
                'sector_num': category_info['sector_size'],
                'industry_num': category_info['industry_size'],
                'filters': [32, 64, 32],
                'kernel_sizes': [3, 5, 7],
                'dropout': 0.3
            },
            'train_loader': train_loader,
            'test_loader': test_loader
        }
    }
    
    # 存储每个模型的性能指标
    results = {}
    
    # 训练和评估每个模型
    for model_name, model_config in models_to_compare.items():
        print(f"\n训练 {model_name} 模型...")
        
        # 创建模型
        model = model_config['class'](**model_config['args'])
        
        # 训练模型
        model = train_model_with_categories(
            model, 
            model_config['train_loader'], 
            model_config['test_loader'],
            epochs=25,
            patience=5
        )
        
        # 评估模型
        eval_results = evaluate_model_with_categories(model, model_config['test_loader'])
        
        # 保存模型
        torch.save(model.state_dict(), f'./results/model_comparison/{model_name}_model.pth')
        
        # 记录结果
        results[model_name] = {
            'accuracy': eval_results['accuracy'],
            'roc_auc': eval_results['roc_auc'],
            'pr_auc': eval_results['pr_auc']
        }
    
    # 合并传统模型和神经网络模型的结果
    all_results = {**results, **traditional_results}
    
    # 创建比较表
    results_df = pd.DataFrame([
        {'Model': model_name, 'Accuracy': metrics['accuracy'], 'ROC_AUC': metrics['roc_auc'], 'PR_AUC': metrics.get('pr_auc', 0)}
        for model_name, metrics in all_results.items()
    ]).sort_values('ROC_AUC', ascending=False)
    
    # 保存比较结果
    results_df.to_csv('./results/model_comparison/model_comparison.csv', index=False)
    print("\n模型性能比较:")
    print(results_df)
    
    # 可视化比较结果
    plt.figure(figsize=(12, 8))
    
    # 准确率比较
    plt.subplot(1, 3, 1)
    plt.bar(results_df['Model'], results_df['Accuracy'])
    plt.ylim(0.4, 0.7)  # 适当设置y轴范围，使差异更明显
    plt.xticks(rotation=45)
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('模型准确率比较')
    
    # ROC AUC比较
    plt.subplot(1, 3, 2)
    plt.bar(results_df['Model'], results_df['ROC_AUC'])
    plt.ylim(0.5, 0.8)  # 适当设置y轴范围，使差异更明显
    plt.xticks(rotation=45)
    plt.xlabel('模型')
    plt.ylabel('ROC AUC')
    plt.title('模型ROC AUC比较')
    
    # PR AUC比较
    plt.subplot(1, 3, 3)
    plt.bar(results_df['Model'], results_df['PR_AUC'])
    plt.ylim(0.4, 0.7)  # 适当设置y轴范围，使差异更明显
    plt.xticks(rotation=45)
    plt.xlabel('模型')
    plt.ylabel('PR AUC')
    plt.title('模型PR AUC比较')
    
    plt.tight_layout()
    plt.savefig('./results/model_comparison/model_comparison.png')
    plt.close()
    
    return results_df

def train_comparison_pipeline(df_train, df_test, feature_cols):
    """完整的模型比较和训练流程"""
    import torch.nn.functional as F
    import math
    
    # 确保结果目录存在
    os.makedirs('./results', exist_ok=True)
    
    # 1. 首先训练传统机器学习模型
    print("第1步: 训练传统机器学习模型 (RandomForest, LightGBM)")
    traditional_models, traditional_results, _ = train_traditional_models(df_train, df_test, feature_cols)
    
    # 2. 比较不同的神经网络架构
    print("\n第2步: 比较不同的神经网络架构")
    model_comparison = compare_models(df_train, df_test, feature_cols)
    
    # 3. 训练最佳模型的Bagging集成版本
    print("\n第3步: 创建最佳模型的Bagging集成")
    
    # 根据比较结果找出最佳神经网络模型
    best_nn_model = model_comparison[model_comparison['Model'].isin(['MLP', 'LSTM', 'Transformer', 'CNN'])].iloc[0]['Model']
    
    print(f"最佳神经网络模型是: {best_nn_model}")
    
    # 使用特征选择获取最佳特征
    selected_features, _ = select_optimal_features(df_train, feature_cols)
    
    # 为最佳模型准备数据
    if best_nn_model in ['LSTM', 'Transformer']:
        train_loader, test_loader, category_info, _ = prepare_sequence_data(
            df_train, df_test, selected_features, seq_length=10, stride=5)
    else:
        train_loader, test_loader, _, category_info = prepare_data_with_categories(
            df_train, df_test, selected_features)
    
    # 创建对应的模型类
    model_classes = {
        'MLP': CategoryEmbeddingMLP,
        'LSTM': StockLSTM,
        'Transformer': StockTransformer,
        'CNN': StockCNN
    }
    
    best_model_class = model_classes[best_nn_model]
    
    # 创建模型参数
    if best_nn_model == 'LSTM':
        model_args = {
            'input_size': len(selected_features),
            'sector_num': category_info['sector_size'],
            'industry_num': category_info['industry_size']
        }
    elif best_nn_model == 'Transformer':
        model_args = {
            'input_size': len(selected_features),  # 修改: 使用特征数量
            'sector_num': category_info['sector_size'],
            'industry_num': category_info['industry_size']
        }
    else:
        model_args = {
            'input_size': len(selected_features),
            'sector_num': category_info['sector_size'],
            'industry_num': category_info['industry_size']
        }
    
    # 训练Bagging集成模型
    bagging_model = BaggingStockModel(
        base_model_class=best_model_class,
        n_estimators=5,  # 可以根据计算资源调整
        sample_ratio=0.8,
        feature_ratio=0.8,
        **model_args
    )
    
    bagging_model.fit(train_loader, test_loader, epochs=20, save_dir=f'./results/bagging_{best_nn_model}')
    
    # 评估Bagging模型
    bagging_results = bagging_model.evaluate(test_loader)
    
    # 4. 比较单个模型和集成模型的性能
    print("\n第4步: 比较单个模型和集成模型的性能")
    
    # 获取最佳单个模型的性能
    best_single_model_acc = model_comparison[model_comparison['Model'] == best_nn_model]['Accuracy'].values[0]
    best_single_model_roc = model_comparison[model_comparison['Model'] == best_nn_model]['ROC_AUC'].values[0]
    
    # 可视化对比
    plt.figure(figsize=(10, 6))
    models = ['Best Single Model', 'Bagging Ensemble']
    accuracies = [best_single_model_acc, bagging_results['accuracy']]
    
    plt.subplot(1, 2, 1)
    plt.bar(models, accuracies)
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('准确率对比: 单个模型 vs Bagging集成')
    
    roc_aucs = [best_single_model_roc, bagging_results['roc_auc']]
    
    plt.subplot(1, 2, 2)
    plt.bar(models, roc_aucs)
    plt.xlabel('模型')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC对比: 单个模型 vs Bagging集成')
    
    plt.tight_layout()
    plt.savefig('./results/single_vs_ensemble.png')
    plt.close()
    
    # 总结性能
    print("\n性能总结:")
    print(f"最佳单个神经网络模型 ({best_nn_model}) - 准确率: {best_single_model_acc:.4f}, ROC AUC: {best_single_model_roc:.4f}")
    print(f"Bagging集成模型 - 准确率: {bagging_results['accuracy']:.4f}, ROC AUC: {bagging_results['roc_auc']:.4f}")
    
    best_traditional = model_comparison[model_comparison['Model'].isin(['RandomForest', 'LightGBM'])].iloc[0]['Model']
    best_traditional_acc = model_comparison[model_comparison['Model'] == best_traditional]['Accuracy'].values[0]
    best_traditional_roc = model_comparison[model_comparison['Model'] == best_traditional]['ROC_AUC'].values[0]
    
    print(f"最佳传统模型 ({best_traditional}) - 准确率: {best_traditional_acc:.4f}, ROC AUC: {best_traditional_roc:.4f}")
    
    return {
        'traditional_models': traditional_models,
        'traditional_results': traditional_results,
        'model_comparison': model_comparison,
        'bagging_model': bagging_model,
        'bagging_results': bagging_results
    }

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('./results', exist_ok=True)
    
    # 加载数据
    loaded_train = np.load("./datasets/df_train_signature_data.npz")
    loaded_test = np.load("./datasets/df_test_signature_data.npz")

    df_train_final = pd.DataFrame(data=loaded_train['data'], columns=loaded_train['features'])
    df_test_final = pd.DataFrame(data=loaded_test['data'], columns=loaded_test['features'])
    with open("./datasets/df_train_new_features.txt", "r") as f:
        new_features = eval(f.read())

    # 添加RET标签
    y_dict_train_RET = pd.read_csv("./datasets/y_train.csv").set_index('ID')['RET'].to_dict()
    y_dict_test_RET = pd.read_csv("./datasets/test_rand.csv").set_index('ID')['RET'].to_dict()

    df_train_final['RET'] = df_train_final['ID'].map(y_dict_train_RET)
    df_test_final['RET'] = df_test_final['ID'].map(y_dict_test_RET)
    
    # 确保SECTOR和INDUSTRY列存在（如果没有，需要加载）
    if 'SECTOR' not in df_train_final.columns:
        # 加载SECTOR数据
        sector_data = pd.read_csv("./datasets/stock_info.csv")
        sector_dict = sector_data.set_index('Symbol')['Sector'].to_dict()
        industry_dict = sector_data.set_index('Symbol')['Industry'].to_dict()
        
        # 获取股票代码（假设ID的前部分是股票代码）
        def extract_symbol(id_str):
            return id_str.split('_')[0] if '_' in id_str else id_str
        
        df_train_final['Symbol'] = df_train_final['ID'].apply(extract_symbol)
        df_test_final['Symbol'] = df_test_final['ID'].apply(extract_symbol)
        
        # 映射SECTOR和INDUSTRY
        df_train_final['SECTOR'] = df_train_final['Symbol'].map(sector_dict)
        df_train_final['INDUSTRY'] = df_train_final['Symbol'].map(industry_dict)
        
        df_test_final['SECTOR'] = df_test_final['Symbol'].map(sector_dict)
        df_test_final['INDUSTRY'] = df_test_final['Symbol'].map(industry_dict)
    
    # 获取特征列 - 不修改signature
    feature_cols = [col for col in df_train_final.columns if col.startswith('SIG_')]
    
    # 过滤new_features中实际存在于数据框中的特征
    valid_new_features = [feature for feature in new_features if feature in df_train_final.columns]
    if len(valid_new_features) < len(new_features):
        missing_features = set(new_features) - set(valid_new_features)
        print(f"警告: 以下特征不在数据框中: {missing_features}")
    
    feature_cols.extend(valid_new_features)
    print(f"特征数量: {len(feature_cols)}")
    
    # 执行完整的模型比较流程
    results = train_comparison_pipeline(df_train_final, df_test_final, feature_cols)
    
    print("\n模型训练和评估完成!")