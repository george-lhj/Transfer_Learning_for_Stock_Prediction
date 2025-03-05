import pandas as pd
import math

def random_sample(df, group, random_state=42):
    print(f"Number of values in {group} is {df[group].nunique()}")
    counts = df[group].value_counts()
    min_count = counts.min()
    min_category = counts.idxmin()
    print(f"Minimum count is {min_count} for {group}: {min_category}")

    sampled_df_by_group = []
    for i in df[group].unique():
        group_df = df[df[group] == i]
        sampled_df = group_df.sample(n=min_count, random_state=random_state)
        sampled_df_by_group.append(sampled_df)
    
    result_df = pd.concat(sampled_df_by_group).reset_index(drop=True)
    
    return result_df

def random_sample_w_mean_reduction(df, group, reduction=0.9, random_state=42):
    print(f"Number of values in {group} is {df[group].nunique()}")
    counts = df[group].value_counts()
    min_count = counts.min()
    min_category = counts.idxmin()
    print(f"Minimum count is {min_count} for {group}: {min_category}")

    reduced_count = max(1, math.floor(min_count * reduction))
    print(f"Sampling {reduced_count} stocks for the minimum category ({min_category}).")

    sampled_df_by_group = []
    for i in df[group].unique():
        cat_df = df[df[group] == i]
        if i == min_category:
            n_samples = reduced_count
        else:
            n_samples = min_count

        if len(cat_df) < n_samples:
            raise ValueError(f"Not enough stocks in category {i} to sample {n_samples} rows")
        sampled_df = cat_df.sample(n=n_samples, random_state=random_state)
        sampled_df_by_group.append(sampled_df)

    result_df = pd.concat(sampled_df_by_group).reset_index(drop=True)
    
    return result_df

# df = pd.read_csv(r"C:\Users\shirl\OneDrive\桌面\signature.csv")