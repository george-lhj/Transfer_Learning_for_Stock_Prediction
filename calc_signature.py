import pandas as pd
import numpy as np
import pandas as pd
import iisignature
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def calc_signature_new(data_path, time_window=10, order=3, save=True, 
                     sig_columns=['Return', 'Transaction_Rate', 'date'], 
                     output_filename='new_sig_data'):
    """
    计算股票数据的特征签名，使用前time_window天的数据预测当天的收益率
    
    参数:
    data_path (str): 数据文件路径
    time_window (int): 用于计算签名的历史时间窗口大小
    order (int): 签名阶数
    save (bool): 是否保存结果
    sig_columns (list): 用于计算签名的列
    output_filename (str): 输出文件名
    
    返回:
    pandas.DataFrame: 包含签名特征的数据框
    """
    print(f"加载数据: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("不支持的文件格式，请使用.csv或.parquet")

    df = df[['symbol', 'date', 'volume', 'Transaction_Rate', 'Return', 'Sector', 'Industry']]
    print(f"数据已加载，形状: {df.shape}")
    print(f"使用时间窗口: 前{time_window}天")
    print(f"计算签名的列: {sig_columns}")
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 对缺失值进行处理
    df = df.dropna()
    
    # 对收益率进行对数转换 (1+x)，避免负值问题
    if 'Return' in df.columns:
        # 保存原始Return用于最终结果
        df['RET_original'] = df['Return']
        df['Return'] = np.log1p(df['Return'])
    
    # 处理交易量的尺度问题（使用对数变换）
    if 'volume' in df.columns:
        df['volume'] = np.log1p(df['volume'])
        
    # 将日期转换为数值格式(时间戳)，以便用于签名计算
    if 'date' in sig_columns:
        # 创建一个新列用于签名计算
        df['date_numeric'] = (df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
        # 调整sig_columns列表，用数值日期替换原始日期
        sig_columns = [col if col != 'date' else 'date_numeric' for col in sig_columns]
    
    # 按股票和日期排序
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 为每个股票创建滚动窗口
    def process_stock_group(stock_data):
        stock_symbol = stock_data['symbol'].iloc[0]
        all_signatures = []
        
        # 确保有足够的历史数据
        if len(stock_data) <= time_window:
            return []
        
        # 创建滚动窗口
        # 从索引time_window开始，这样每个窗口都有足够的历史数据
        for i in range(time_window, len(stock_data)):
            # 使用前time_window天的数据计算特征
            feature_window = stock_data.iloc[i-time_window:i]
            # 当前行（包含要预测的return）
            current_row = stock_data.iloc[i]
            
            # 提取计算签名所需数据（只包含数值型数据）
            path_data = feature_window[sig_columns].values.astype(np.float64)
            
            try:
                # 计算签名
                signature = iisignature.sig(path_data, order)
                augmented_sig = np.insert(signature, 0, 1.0)  # 增广签名
                
                # 当前行的日期（预测日）
                current_date = current_row['date']
                
                # 获取其他需要的信息
                signature_info = {
                    'symbol': stock_symbol,
                    'date': current_date,
                    'Sector': current_row['Sector'],
                    'Industry': current_row['Industry']
                }
                
                # 添加签名特征
                for j, sig_val in enumerate(augmented_sig):
                    signature_info[f'SIG_{j}'] = sig_val
                
                # 使用当前行的return作为标签
                if 'RET_original' in current_row:
                    signature_info['RET'] = current_row['RET_original']
                else:
                    signature_info['RET'] = current_row['Return']
                
                all_signatures.append(signature_info)
            except Exception as e:
                print(f"处理股票 {stock_symbol} 窗口时出错: {e}")
                continue
        
        return all_signatures
    
    # 并行处理每个股票
    print("开始计算签名...")
    all_stock_groups = list(df.groupby('symbol'))
    
    results = Parallel(n_jobs=-1)(
        delayed(process_stock_group)(group_data) 
        for _, group_data in tqdm(all_stock_groups, desc="处理股票")
    )
    
    # 合并所有结果
    all_signatures = []
    for stock_signatures in results:
        all_signatures.extend(stock_signatures)
    
    # 创建结果数据框
    signature_df = pd.DataFrame(all_signatures)
    
    # 统计结果
    print(f"计算完成！生成了 {len(signature_df)} 个签名样本")
    print(f"每个样本包含 {sum(1 for col in signature_df.columns if col.startswith('SIG_'))} 个签名特征")
    
    # 保存结果
    if save:
        output_path = f"./datasets/{output_filename}_w{time_window}_o{order}.parquet"
        os.makedirs("./datasets", exist_ok=True)
        signature_df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"结果已保存至: {output_path}")
    
    return signature_df

# 示例使用
# 函数调用示例，您可以根据需要修改参数
if __name__ == "__main__":
    window_span = 10
    order_num = 3
    df_signatures = calc_signature_new(
        data_path='datasets/sp500_final_data_5y.csv',
        time_window=window_span,  # 可以改为3,5,7,10等
        order=order_num,
        sig_columns=['date', 'Return', 'Transaction_Rate'],  # 计算签名的列
        output_filename=f'sig_data_SP500'
    )