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
    Calculate feature signatures for stock data, using data from the previous time_window days to predict the current day's return
    
    Parameters:
    data_path (str): Path to the data file
    time_window (int): Historical time window size for signature calculation
    order (int): Signature order
    save (bool): Whether to save results
    sig_columns (list): Columns used for signature calculation
    output_filename (str): Output filename
    
    Returns:
    pandas.DataFrame: DataFrame containing signature features
    """
    print(f"Loading data: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format, please use .csv or .parquet")

    df = df[['symbol', 'date', 'volume', 'Transaction_Rate', 'Return', 'Sector', 'Industry']]
    print(f"Data loaded, shape: {df.shape}")
    print(f"Using time window: previous {time_window} days")
    print(f"Columns for signature calculation: {sig_columns}")
    
    # Ensure correct date format
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values
    df = df.dropna()
    
    # Apply logarithmic transformation to returns (1+x) to avoid negative values
    if 'Return' in df.columns:
        # Save original Return for final results
        df['RET_original'] = df['Return']
        df['Return'] = np.log1p(df['Return'])
    
    # Handle volume scale issues (using logarithmic transformation)
    if 'volume' in df.columns:
        df['volume'] = np.log1p(df['volume'])
        
    # Convert date to numeric format (timestamp) for signature calculation
    if 'date' in sig_columns:
        # Create a new column for signature calculation
        df['date_numeric'] = (df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
        # Adjust sig_columns list, replace original date with numeric date
        sig_columns = [col if col != 'date' else 'date_numeric' for col in sig_columns]
    
    # Sort by stock and date
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Create rolling windows for each stock
    def process_stock_group(stock_data):
        stock_symbol = stock_data['symbol'].iloc[0]
        all_signatures = []
        
        # Ensure sufficient historical data
        if len(stock_data) <= time_window:
            return []
        
        # Create rolling windows
        # Start from index time_window to ensure each window has sufficient historical data
        for i in range(time_window, len(stock_data)):
            # Use data from previous time_window days to calculate features
            feature_window = stock_data.iloc[i-time_window:i]
            # Current row (contains the return to predict)
            current_row = stock_data.iloc[i]
            
            # Extract data needed for signature calculation (only numeric data)
            path_data = feature_window[sig_columns].values.astype(np.float64)
            
            try:
                # Calculate signature
                signature = iisignature.sig(path_data, order)
                augmented_sig = np.insert(signature, 0, 1.0)  # Augmented signature
                
                # Current row's date (prediction date)
                current_date = current_row['date']
                
                # Get other required information
                signature_info = {
                    'symbol': stock_symbol,
                    'date': current_date,
                    'Sector': current_row['Sector'],
                    'Industry': current_row['Industry']
                }
                
                # Add signature features
                for j, sig_val in enumerate(augmented_sig):
                    signature_info[f'SIG_{j}'] = sig_val
                
                # Use current row's return as label
                if 'RET_original' in current_row:
                    signature_info['RET'] = current_row['RET_original']
                else:
                    signature_info['RET'] = current_row['Return']
                
                all_signatures.append(signature_info)
            except Exception as e:
                print(f"Error processing window for stock {stock_symbol}: {e}")
                continue
        
        return all_signatures
    
    # Process each stock in parallel
    print("Starting signature calculation...")
    all_stock_groups = list(df.groupby('symbol'))
    
    results = Parallel(n_jobs=-1)(
        delayed(process_stock_group)(group_data) 
        for _, group_data in tqdm(all_stock_groups, desc="Processing stocks")
    )
    
    # Merge all results
    all_signatures = []
    for stock_signatures in results:
        all_signatures.extend(stock_signatures)
    
    # Create result DataFrame
    signature_df = pd.DataFrame(all_signatures)
    
    # Statistics
    print(f"Calculation complete! Generated {len(signature_df)} signature samples")
    print(f"Each sample contains {sum(1 for col in signature_df.columns if col.startswith('SIG_'))} signature features")
    
    # Save results
    if save:
        output_path = f"./datasets/{output_filename}_w{time_window}_o{order}_d{len(sig_columns)}.parquet"
        os.makedirs("./datasets", exist_ok=True)
        signature_df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"Results saved to: {output_path}")
    
    return signature_df

# Example usage
# Function call example, you can modify parameters as needed
if __name__ == "__main__":
    window_span = 60
    order_num = 3
    df_signatures = calc_signature_new(
        data_path='datasets/sp500_final_data_5y.csv',
        time_window=window_span,  # Can be changed to 3,5,7,10 etc.
        order=order_num,
        sig_columns=['date', 'Return', 'Transaction_Rate'],  # Columns for signature calculation
        output_filename=f'sig_data_SP500'
    )