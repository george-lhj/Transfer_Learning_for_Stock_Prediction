import numpy as np
import pandas as pd
import iisignature
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kurtosis

def noise_reduction_variables_by_sector(df: pd.DataFrame):
    # Feature engineering
    df = df.copy()
    new_features = []
    # Conditional aggregated features on 1 stock in 1 date and it's previous 20 days
    # 计算 20 天收益率 (RET_) 相关特征
    ret_cols = [col for col in df.columns if col.startswith("RET_")]
    df["std_RET_20days"] = df[ret_cols].std(axis=1)
    df["skew_RET_20days"] = df[ret_cols].skew(axis=1)
    df["kurtosis_RET_20days"] = df[ret_cols].kurtosis(axis=1)
    df["median_RET_20days"] = df[ret_cols].median(axis=1)

    # 计算 20 天交易量 (VOLUME_) 相关特征
    vol_cols = [col for col in df.columns if col.startswith("VOLUME_")]
    df["std_VOL_20days"] = df[vol_cols].std(axis=1)
    df["skew_VOL_20days"] = df[vol_cols].skew(axis=1)
    df["kurtosis_VOL_20days"] = df[vol_cols].kurtosis(axis=1)
    df["median_VOL_20days"] = df[vol_cols].median(axis=1)

    # 更新 new_features 列表
    new_features = [
        "std_RET_20days", "skew_RET_20days", "kurtosis_RET_20days", "median_RET_20days",
        "std_VOL_20days", "skew_VOL_20days", "kurtosis_VOL_20days", "median_VOL_20days"
    ]

    return new_features, df

def process_group(key_group, sig_col, order):
    (stock, id_, industry, industry_group, sector, sub_industry), group = key_group
    group = group.sort_values("DAY", ascending=True)
    
    # Compute additional indicators: cumulative statistics
    # group['std_RET'] = group['RET'].expanding(min_periods=1).std() # 计算每个股票从第一天到当前的累积收益率标准差
    # group['mean_RET'] = group['RET'].expanding(min_periods=1).mean() # 计算每个股票从第一天到当前的累积收益率均值
    # group['std_VOL'] = group['VOLUME'].expanding(min_periods=1).std() # 计算每个股票从第一天到当前的累积交易量标准差
    # group['mean_VOL'] = group['VOLUME'].expanding(min_periods=1).mean() # 计算每个股票从第一天到当前的累积交易量均值

    # sig_col = sig_col + ['std_RET', 'mean_RET', 'std_VOL', 'mean_VOL']
    path = group[sig_col].fillna(0).values.astype(np.float64)
    base_sig = iisignature.sig(path, order)
    sig = np.insert(base_sig, 0, 1.0)  # augmented signature

    # Optionally, trigger garbage collection if needed:
    # gc.collect()
    return sig, (stock, id_, industry, industry_group, sector, sub_industry)

def calculation_signature_using_multiprocessing(df, order=3, sig_col=[]):
    grouped = list(df.groupby(["STOCK", "ID", "INDUSTRY", "INDUSTRY_GROUP", "SECTOR", "SUB_INDUSTRY"]))
    print(len(grouped))
    total_groups = len(grouped)

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_group)(key_group, sig_col, order)
        for key_group in tqdm(grouped, total=total_groups, desc="Processing groups")
    )
    signatures, keys = zip(*results)
    # sig_col = sig_col + ['std_RET', 'mean_RET', 'std_VOL', 'mean_VOL']
    sig_length = iisignature.siglength(len(sig_col), order)
    sig_columns = [f"SIG_{i}" for i in range(sig_length+1)]
    # Pre-allocate NumPy arrays for keys and signatures:
    keys_np = np.array(keys)  # shape: (total_groups, 6)
    signatures_np = np.array(signatures, dtype=np.float64)  # shape: (total_groups, len(sig_columns))
    # Concatenate keys and signatures along the column axis
    combined_np = np.hstack([keys_np, signatures_np])
    columns = ["STOCK", "ID", "INDUSTRY", "INDUSTRY_GROUP", "SECTOR", "SUB_INDUSTRY"] + sig_columns
    # df_signature = pd.DataFrame(combined_np, columns=columns)
    return columns, combined_np

def prepare_df_for_signature_computation(x: pd.DataFrame, y: pd.DataFrame, save=False, filename="Data"):
    df = x.sort_values(by=['STOCK', "ID"]).reset_index(drop=True)
    df = df.fillna(0)
    ret_cols = [col for col in df.columns if col.startswith('RET')]
    df[ret_cols] = df[ret_cols].apply(lambda x: np.log1p(x))

    # noise reduction
    # new_features, new_features_df = noise_reduction_variables_by_sector(df)

    ret_cols = sorted(
        [col for col in df.columns if col.startswith("RET_")],
        key=lambda x: int(x.split("_")[1])
    )
    volume_cols = sorted(
        [col for col in df.columns if col.startswith("VOLUME_")],
        key=lambda x: int(x.split("_")[1])
    )

    # expand RET and VOLUME
    rets = df[ret_cols].values.ravel()
    volumes = df[volume_cols].values.ravel()

    # feature_values = {}
    # for feature in new_features:
    #     feature_values[feature] = np.repeat(df[feature].values.ravel(), 20)

    # repeat INDUSTRY, INDUSTRY_GROUP, SECTOR, SUB_INDUSTRY for each RET and VOLUME
    industry = np.repeat(df["INDUSTRY"].values, 20)
    industry_group = np.repeat(df["INDUSTRY_GROUP"].values, 20)
    sector = np.repeat(df["SECTOR"].values, 20)
    sub_industry = np.repeat(df["SUB_INDUSTRY"].values, 20)

    # generate day for RET{i} and VOLUME{i}
    days = np.tile(np.arange(20,0,-1), len(df))

    # generate ID and stock (repeat 20 times)
    ids = np.repeat(df["ID"].values, 20)
    stocks = np.repeat(df["STOCK"].values, 20)
    dates = np.repeat(df["DATE"].values, 20)

    df_signature = pd.DataFrame({
        "ID": ids,
        "STOCK": stocks,
        "DAY": days,
        "DATE": dates,
        "RET": rets,
        "VOLUME": volumes,
        "INDUSTRY": industry,
        "INDUSTRY_GROUP": industry_group,
        "SECTOR": sector,
        "SUB_INDUSTRY": sub_industry
    })

    # for feature in new_features:
    #     df_signature[feature] = feature_values[feature]

    df_signature = df_signature.sort_values(["STOCK", "ID", "DAY"])

    df_signature = df_signature[
        (df_signature['RET'] >= df_signature['RET'].quantile(0.005)) & 
        (df_signature['RET'] <= df_signature['RET'].quantile(0.995)) & 
        (df_signature['VOLUME'] >= df_signature['VOLUME'].quantile(0.005)) & 
        (df_signature['VOLUME'] <= df_signature['VOLUME'].quantile(0.995))
    ]
    
    if save:
        df_signature.to_parquet(f"./datasets/{filename}.parquet", engine='pyarrow', compression='snappy')  # 保存为 Parquet
        # new_features_df.to_parquet(f"./datasets/{filename}_new_features.parquet", engine='pyarrow', compression='snappy')  # 保存为 Parquet
        # with open(f"./datasets/{filename}_new_features.txt", "w") as f:
        #     f.write(str(new_features))


def calc_signature(df, order, new_features, filename, save=False):
    df = df.copy().fillna(0)
    # Define the list of columns for signature computation:
    sig_lst = [
        "RET",          # Original log-transformed return
        "VOLUME",       # Trading volume
        "DAY",          # Time indicator
    ]
    
    # Assume df_signature is your input DataFrame
    column, combined_np_array = calculation_signature_using_multiprocessing(df, order=order, sig_col=sig_lst)
    column_array = np.array(column, dtype='U')
    if save == True:
        np.savez_compressed(
            f"./datasets/{filename}_signature_data.npz",
            features=column_array,          # 列名单独存储
            data=combined_np_array.astype(np.float64)
        )
        print(f"Signature data saved to ./datasets/{filename}_signature_data.npz")