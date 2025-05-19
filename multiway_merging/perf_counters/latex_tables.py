import pandas as pd
import math
import re;
import numpy as np

def buffer_size(total_bytes, fan_in):
    if (math.isnan(total_bytes)):
        return -1
    leaf_count = fan_in
    size_of_simd_element = 8
    size_of_circular_buffer = 32
    size_of_relation = 16
    inner_nodes = fan_in-2
    total_fifo_size = total_bytes / size_of_simd_element - leaf_count - (inner_nodes * size_of_circular_buffer + inner_nodes * 1 + leaf_count * size_of_relation + size_of_simd_element -1) / size_of_simd_element
    return int(total_fifo_size / inner_nodes)


def read_benchmark_to_dataframe(filepath):
    with open(filepath, "r") as f:
        # Skip until we find the header
        while True:
            pos = f.tell()
            line = f.readline()
            if line.startswith("name,"):
                f.seek(pos)
                break

        # Read the remaining lines into a DataFrame
        df = pd.read_csv(f)
        df["Throughput"] = (df["Tuples"] / df["cpu_time"]) * 1e-3
        
        # Extract the 0th and 3rd parts
        algo = df["name"].str.split("/").str[0].str.replace("^BM_", "", regex=True)
        algo = algo.replace({
            "MWAY_MERGE": "MWay",
            "KWAY_MERGE": "KWay"
        })
        size_str = df["name"].str.split("/").str[3]

        # Convert to numeric (bytes)
        size_bytes = pd.to_numeric(size_str, errors="coerce")

        # Convert to MiB with float division
        size_mib = size_bytes / (2**20)

        # Format with up to 2 decimal places (e.g., 32.00 MiB)
        formatted_size = size_mib.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        
        # Avoid adding cache label for NaN values in size_mib
        cache_label = np.where(pd.notna(size_mib) & (size_mib == size_mib.min()), " (L2)", "")
        cache_label = np.where(pd.notna(size_mib) & (size_mib != size_mib.min()), " (L3)", cache_label)
        
        # print(size_bytes)
        df['Total Bytes'] = size_bytes
        df['Buffer'] = df.apply(lambda row: buffer_size(row['Total Bytes'], row['Fan-in']), axis=1)
        
        # Combine the algo and cache labels
        df["Algo"] = algo + cache_label
        df["IPC"] = df["instructions"] / df["cpu-cycles"]
        df["Instr./Tup."] = df["instructions"] / df["Tuples"]
        df["Cyc./Tup."] = df["cpu-cycles"] / df["Tuples"]

    return df

def to_latex_tables(df):
    buffer_algos = ["MWay (L2)", "MWay (L3)"]
    for system_name, group in df.groupby("System"):
        group = group.copy()

        # Pivot each metric
        pivot_instr = group.pivot(index='Fan-in', columns='Algo', values='Instr./Tup.')
        pivot_ipc = group.pivot(index='Fan-in', columns='Algo', values='IPC')
        pivot_cyc = group.pivot(index='Fan-in', columns='Algo', values='Cyc./Tup.')

        # Filter buffer-specific algorithms only
        pivot_buf = group[group['Algo'].isin(buffer_algos)].pivot(
            index='Fan-in', columns='Algo', values='Buffer'
        )

        # Drop any buffer columns that are fully NaN
        pivot_buf.dropna(axis=1, how='all', inplace=True)

        # Build the list of sections to concat
        components = [
            ('Instr./Tupl.', pivot_instr),
            ('IPC', pivot_ipc),
            ('Cyc./Tup.', pivot_cyc),
        ]

        if not pivot_buf.empty:
            components.append(('Tupl. per Circ. Buffer', pivot_buf))

        # Combine into a MultiIndex column DataFrame
        combined = pd.concat(
            [df for _, df in components],
            axis=1,
            keys=[key for key, _ in components]
        )
        
        # print(combined)
        
        rate_of_change = combined.pct_change() * 100 
        

        # Rename columns to indicate they are rate of change
        rate_of_change.columns = pd.MultiIndex.from_tuples(
            [(lvl0, '$\\Delta$ [\\%]') for lvl0, lvl1 in combined.columns]
        )
        

        # # # Generate LaTeX table
        latex_table = combined.to_latex(
            multicolumn=True,
            multicolumn_format='c',
            index=True,
            na_rep='\\textemdash',
            float_format="%.2f",
            caption=f"{system_name}",
            label=f"tab:mway_kway_comparison_{re.search(r"\{([^}]*)\}", system_name).group(1).replace(" ", "_").lower()}"
        )

        # Replace top/mid/bottom rules with standard lines (if needed)
        latex_table = latex_table.replace("\\toprule", "\\hline")
        latex_table = latex_table.replace("\\midrule", "\\hline")
        latex_table = latex_table.replace("\\bottomrule", "\\hline")

        # Wrap in resizebox
        latex_table = latex_table.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}")
        latex_table = latex_table.replace("\\end{tabular}", "\\end{tabular}%\n}")

        # Extract subcolumn counts per group for cline
        cline_line = ""
        current_col = 2  # starts at 2 to skip the index column
        for label, df_part in components:
            num_subcols = df_part.shape[1]
            if num_subcols > 0:
                cline_line += f"\\cline{{{current_col}-{current_col + num_subcols - 1}}} "
                current_col += num_subcols

        # Insert \cline after the multicolumn header row
        lines = latex_table.splitlines()
        for i, line in enumerate(lines):
            if all(header in line for header, _ in components):
                lines.insert(i + 1, cline_line)
                break

        # Add vertical lines to tabular column format
        # Find the line starting with \begin{tabular}{...}
        for i, line in enumerate(lines):
            if line.strip().startswith("\\begin{tabular}"):
                # Get number of columns including index
                num_total_cols = sum(df.shape[1] for _, df in components) + 1
                lines[i] = f"\\begin{{tabular}}{{|{'|'.join(['c'] * num_total_cols)}|}}"
                break

        latex_table = "\n".join(lines)
        
        latex_table = latex_table.replace("\\multicolumn{3}{c}{Instr./Tupl.}", "\\multicolumn{3}{c|}{Instr./Tupl.}")
        latex_table = latex_table.replace("\\multicolumn{3}{c}{Cyc./Tupl.}", "\\multicolumn{3}{c|}{Cyc./Tupl.}")
        latex_table = latex_table.replace("\\multicolumn{3}{c}{IPC}", "\\multicolumn{3}{c|}{IPC}")
        latex_table = latex_table.replace("{|c|c|c|c|c|c|c|c|c|c|c|c|}", "{c|ccc|ccc|ccc|cc}")
        latex_table = latex_table.replace("Fan-in &  &  &  &  &  &  &  &  &  &  &  \\\\", "")
        latex_table = latex_table.replace("Algo", " ")
        latex_table = latex_table.replace("& \\multicolumn{3}{c|}{Instr./Tupl.}", "\\multirow{2}{*}{Fan-in} & \\multicolumn{3}{c|}{Instr./Tupl.}")
        
        lines = latex_table.splitlines()[:-3]

        # Join the remaining lines back into a string
        result = "\n".join(lines)
        
        # Print or save
        roc = rate_of_change.to_latex(
            multicolumn=True,
            multicolumn_format='c',
            index=True,
            na_rep='\\textemdash',
            float_format="%.2f",
            caption=f"{system_name}",
            label=f"tab:mway_kway_comparison_{re.search(r"\{([^}]*)\}", system_name).group(1).replace(" ", "_").lower()}"
        )
        lines = roc.splitlines()[8:]

        # Join the remaining lines back into a string
        result = result + '\n& \\multicolumn{11}{c}{$\\Delta$ ROC (Change in [\\%])}\\\\' + "\n".join(lines)
        result = result.replace('Fan-in &  &  &  &  &  &  &  &  \\\\', '')
        result = result.replace("\\toprule", "\\hline")
        result = result.replace("\\midrule", "\\hline")
        result = result.replace("\\bottomrule", "\\hline")
        result = result.replace('\\end{tabular}', '\\end{tabular}}')
        result = result.replace("\\begin{table}", "\\begin{table}[H]")
        print(result)
        print('\n\n')
        
        
name = {
    "cp03": "\\systemA{System A}",
    "cx32": "\\systemB{System B}",
    "ga02": "\\systemC{System C}",
    "nx05": "\\systemE{System E}"
}

def main():
    systems = ['cp03', 'cx32', 'nx05', 'ga02']
    df_list = []
    for system in systems:
        df = read_benchmark_to_dataframe(system + '_perf.csv')
        df["System"] = name[system]
        df_list.append(df)

    unified_df = pd.concat(df_list, ignore_index=True)
    to_latex_tables(unified_df)

if __name__ == "__main__":
    main()
