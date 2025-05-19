import os
import pandas as pd

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

# Reset index for easier row iteration
pivoted_df = pivoted_df.reset_index()

# Generate the LaTeX table
latex_table = "\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|c||}\n\\hline\n"
latex_table += "\\multirow{2}{*}{System} & \\multirow{2}{*}{Fan-out} & \\multicolumn{4}{|c}{SWB} & \\multicolumn{4}{|c}{SWB+NT} & \\multicolumn{4}{|c}{SWB+NT+PF} \\\\\n"
latex_table += "\\cline{3-14}\n"
latex_table += " &  & time histo & time init & time partition & total & time histo & time init & time partition & total & time histo & time init & time partition & total \\\\\n"

# Track the previous directory to apply multirow formatting
prev_dir = None
row_count = pivoted_df['dir'].value_counts()

metrics_one = {
    'arm': [],
    'avx2': [],
    'avx512': [],
    'power': []
}

metrics_two = {
    'arm': [],
    'avx2': [],
    'avx512': [],
    'power': []
}

for index, row in pivoted_df.iterrows():
    dir_name = row['dir']
    partition_size = row['num_partitions']

    # If this is the first row of a new directory, use \multirow
    if dir_name != prev_dir:
        latex_table += f"\\hline\n\\multirow{{{row_count[dir_name]}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{name[dir_name]}}}}} & {partition_size} "
    else:
        latex_table += f"& {partition_size} "

    # Fill in the values for each variant and metric, including the total time
    times = []
    for variant in algorithm_versions:
        for metric in ['time_histogram', 'time_init', 'time_partition']:
            col_name = f"{variant} {metric}"
            latex_table += f"& {int(row.get(col_name, ''))} "
            
        col_name = f"{variant} time_total"
        latex_table += f"& {int(row.get(col_name, ''))} "
        times.append(int(row.get(col_name, '')))
        
    latex_table += "\\\\\n"
    prev_dir = dir_name  # Update previous directory

latex_table += "\\hline\n\\end{tabular}}\n\\caption{Benchmark results clustered by directory}\n\\label{tab:radix_partition}\n\\end{table}"

# Output the LaTeX code
print(latex_table)
