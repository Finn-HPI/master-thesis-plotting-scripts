import os
import pandas as pd
from scipy.stats import gmean

# Define the directories and algorithm versions
directories = ['avx2', 'avx512', 'arm', 'power']
algorithm_versions = ['swb', 'swb_nt', 'swb_nt_pf']
file_names = [f"radix_partition_{version}.csv" for version in algorithm_versions]

# Initialize a list to hold data for DataFrame
df_list = []

# Mapping of directory names to system names
name = {
    "arm": "ARM Neoverse-V2",
    "avx2": "AMD EPYC 7742",
    "avx512": "Intel Xeon Gold 5220S",
    "power": "IBM Power10"
}

# Read all CSV files and collect data
for dir in directories:
    for version, file_name in zip(algorithm_versions, file_names):
        file_path = os.path.join(dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['dir'] = dir  # Add directory as a column
            df['variant'] = version  # Add variant as a column
            df['time_total'] = df['time_histogram'] + df['time_init'] + df['time_partition']  # Compute total time
            df_list.append(df)

# Combine all data into a single DataFrame
combined_df = pd.concat(df_list)

# Reshape DataFrame: Create columns for partition, dir, variant, and times
reshaped_df = combined_df.melt(id_vars=['num_partitions', 'dir', 'variant'], 
                                value_vars=['time_histogram', 'time_init', 'time_partition', 'time_total'], 
                                var_name='metric', value_name='time')

# Pivot the DataFrame so that each (variant, metric) gets its own column
pivoted_df = reshaped_df.pivot_table(index=['dir', 'num_partitions'], 
                                      columns=['variant', 'metric'], 
                                      values='time', aggfunc='first')

# Reorganize column names to improve readability
pivoted_df.columns = [f"{variant} {metric}" for variant, metric in pivoted_df.columns]

# print(pivoted_df.columns)
print(pivoted_df.head())

# Step 1: Compute the ratio
pivoted_df['ratio'] = pivoted_df['swb time_total'] / pivoted_df['swb_nt time_total']

# Step 2: Group by 'dir' and compute geometric mean
geo_mean = pivoted_df.groupby('dir')['ratio'].apply(gmean)

print(geo_mean)