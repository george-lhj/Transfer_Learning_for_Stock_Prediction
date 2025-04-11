import numpy as np
import pandas as pd
import os

def signature_distance(sig_a, sig_b):
    """Calculate the squared Euclidean distance between two signature vectors"""
    return np.dot(sig_a, sig_a) - 2 * np.dot(sig_a, sig_b) + np.dot(sig_b, sig_b)

def compute_adaptive_weights(time_window=10, order=3, dimension=3, gamma=1.0, 
                              input_filename='sig_data_SP500', 
                              save=True):
    """
    Load signature feature data, calculate adaptive weights for each stock, and save results.
    
    Parameters:
    time_window (int): Window size for file path identification (must match signature calculation)
    order (int): Signature order (must match signature calculation)
    gamma (float): Weight decay factor
    input_filename (str): Base input filename (without extension)
    save (bool): Whether to save results with weights
    
    Returns:
    pd.DataFrame: Signature data with adaptive weights
    """
    # Construct input file path
    # input_path = f"./datasets/{input_filename}_w{time_window}_o{order}_d{dimension}.parquet"
    input_path = f"./datasets/{input_filename}_w{time_window}_o{order}_d{dimension}_clusters.parquet"
    print(f"Loading signature feature data: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_parquet(input_path)

    # Automatically identify signature columns
    sig_cols = [col for col in df.columns if col.startswith('SIG_')]
    print(f"Identified {len(sig_cols)} signature feature columns")

    # Initialize
    df = df.sort_values(['symbol', 'date']).copy()
    df['adaptive_weight'] = np.nan

    def compute_weights_for_symbol(group):
        group = group.copy()
        if len(group) < 2:
            return group
        
        # Use the last row as reference point
        current_sig = group[sig_cols].iloc[-1].values
        
        distances = group[sig_cols].iloc[:-1].apply(
            lambda row: signature_distance(row.values, current_sig),
            axis=1
        )
        exp_weights = np.exp(-gamma * distances)
        normalized_weights = exp_weights / exp_weights.sum()
        
        group.loc[group.index[:-1], 'adaptive_weight'] = normalized_weights.values
        return group

    # Calculate weights for each stock group
    df = df.groupby('symbol', group_keys=False).apply(compute_weights_for_symbol)

    print(f"Weight calculation completed, processed {df['symbol'].nunique()} stock samples")

    if save:
        # output_path = f"./datasets/{input_filename}_w{time_window}_o{order}_d{dimension}_weighted_gamma{gamma}.parquet"
        output_path = f"./datasets/{input_filename}_w{time_window}_o{order}_d{dimension}_weighted_gamma{gamma}_clusters.parquet"
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"Results saved to: {output_path}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Make sure these parameters match those used in `calc_signature_new()`
    for t in [10, 15, 30, 40, 50, 60]:
        weighted_df = compute_adaptive_weights(
            time_window=t,
            order=3,
            dimension=3,
            gamma=10,
            input_filename='sig_data_SP500',
            save=True
        )
