import numpy as np
import pandas as pd

def compute_kernel(sig):
    sig_columns = [col for col in sig.columns if col.startswith("SIG_")]

    sig_data = sig[sig_columns].values

    kernel_matrix = np.dot(sig_data, sig_data.T)

    stock_ids = sig["ID"]
    kernel_df = pd.DataFrame(kernel_matrix, index=stock_ids, columns=stock_ids)
    
    return kernel_df

sig = pd.read_csv(r"C:\Users\shirl\OneDrive\桌面\ucb capstone qrt\df_train_wdate.csv")
sig_date1 = sig[sig['DATE'] == 1]

kernel_df = compute_kernel(sig_date1)

print(kernel_df)