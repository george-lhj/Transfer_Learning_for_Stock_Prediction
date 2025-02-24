import numpy as np
import pandas as pd
import iisignature

def calc_signature(x: pd.DataFrame, y: pd.DataFrame, order=3):
    df = x.sort_values(by=['STOCK', "ID"]).reset_index(drop=True)
    df = df.drop(columns=['DATE'])
    df = df.fillna(0)

    ret_cols = [col for col in df.columns if col.startswith('RET')]
    df[ret_cols] = df[ret_cols].apply(lambda x: np.log1p(x))

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

    # repeat INDUSTRY, INDUSTRY_GROUP, SECTOR, SUB_INDUSTRY for each RET and VOLUME
    industry = np.repeat(df["INDUSTRY"].values, len(ret_cols))
    industry_group = np.repeat(df["INDUSTRY_GROUP"].values, len(ret_cols))
    sector = np.repeat(df["SECTOR"].values, len(ret_cols))
    sub_industry = np.repeat(df["SUB_INDUSTRY"].values, len(ret_cols))

    # generate day for RET{i} and VOLUME{i}
    days = np.tile(np.arange(20,0,-1), len(df))


    # generate ID and stock (repeat 20 times)
    ids = np.repeat(df["ID"].values, len(ret_cols))
    stocks = np.repeat(df["STOCK"].values, len(ret_cols))

    df_signature = pd.DataFrame({
        "ID": ids,
        "STOCK": stocks,
        "DAY": days,
        "RET": rets,
        "VOLUME": volumes,
        "INDUSTRY": industry,
        "INDUSTRY_GROUP": industry_group,
        "SECTOR": sector,
        "SUB_INDUSTRY": sub_industry
    })

    df_signature = df_signature.sort_values(["STOCK", "ID", "DAY"])

    df_signature = df_signature[
        (df_signature['RET'] >= df_signature['RET'].quantile(0.005)) & 
        (df_signature['RET'] <= df_signature['RET'].quantile(0.995)) & 
        (df_signature['VOLUME'] >= df_signature['VOLUME'].quantile(0.005)) & 
        (df_signature['VOLUME'] <= df_signature['VOLUME'].quantile(0.995))
    ]

    sig_col = ["RET", "VOLUME", "DAY"]
    grouped = df_signature.groupby(["STOCK", "ID", 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY'])
    signatures = []
    keys = []

    for (stock, id_, industry, industry_group, sector, sub_industry), group in grouped:
        group = group.sort_values("DAY", ascending=True)
        path = group[sig_col].values.astype(np.float64)

        base_sig = iisignature.sig(path, order)
        sig = np.insert(base_sig, 0, 1.0) # augmented signature
        signatures.append(sig)
        keys.append((stock, id_, industry, industry_group, sector, sub_industry))
    
    sig_length = iisignature.siglength(len(sig_col), order)
    sig_columns = [f"SIG_{i}" for i in range(sig_length+1)]
    
    df_signature = pd.DataFrame(signatures, columns=sig_columns)
    df_signature[["STOCK", "ID", 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY']] = keys
    
    # Reorganize columns
    cols_order = ["STOCK", "ID", 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY'] + sig_columns
    df_final = df_signature[cols_order]

    df_final = pd.merge(df_final, y, on=["ID"], how="inner")
    df_final['RET'] = df_final["RET"].astype(int)

    return df_final