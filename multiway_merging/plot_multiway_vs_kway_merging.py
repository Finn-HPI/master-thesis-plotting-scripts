import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


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
            "MWAY_MERGE": "Multiway Merge",
            "KWAY_MERGE": "K-Way Merge"
        })
        size_str = df["name"].str.split("/").str[3]

        # Convert to numeric (bytes)
        size_bytes = pd.to_numeric(size_str, errors="coerce")

        # Convert to MiB with float division
        size_mib = size_bytes / (2**20)
        
        # Avoid adding cache label for NaN values in size_mib
        cache_label = np.where(pd.notna(size_mib) & (size_mib == size_mib.min()), " (L2-CACHE)", "")
        cache_label = np.where(pd.notna(size_mib) & (size_mib != size_mib.min()), " (L3-CACHE/THREADS)", cache_label)
        
        # Combine the algo and cache labels
        df["Algo"] = algo + cache_label

    return df

name = {
    "power_merge_bench.csv": r"$\bf{(a)}\ \bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "amd_merge_bench.csv": r"$\bf{(b)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "arm_merge_bench.csv": r"$\bf{(c)}\ \bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "nx05_merge_bench.csv": r"$\bf{(d)}\ \bf{System}\ \bf{E}$" + "\n(Intel Xeon Platinum 8352Y)"
}

file_paths = [
    "avx2_merge_bench.csv",
    "arm_merge_bench.csv",
    "power_merge_bench.csv",
    "avx512_merge_bench.csv"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png).")
    args = parser.parse_args()

    df_list = []
    for file_path in file_paths:
        df = read_benchmark_to_dataframe(file_path)
        df["System"] = name[os.path.basename(file_path)]
        df_list.append(df)

    # Concatenate into a single DataFrame
    unified_df = pd.concat(df_list, ignore_index=True)
        
    g = sns.FacetGrid(unified_df, col="System", col_wrap=2,height=5, aspect=1.25,col_order=list(name.values()))
    g.map_dataframe(sns.pointplot, x="Fan-in", y="Throughput", hue="Algo", palette="Set1", markers=["o", "s", "^"])
    g.set_titles(col_template="{col_name}", row_template="", size=16)#, weight="bold")
    
    def annotate(data, **kws):
        # Filter rows that start with 'BM_MWAY_MERGE'
        filtered_data = data[data['name'].str.startswith('BM_MWAY_MERGE')]
        
        # Split the 'name' column by '/' and extract group 3 (index 2)
        extracted_numbers = filtered_data['name'].str.split('/').str[3]
        
        # Convert the extracted group to unique numbers
        unique_numbers = extracted_numbers.unique()
        sorted_numbers = sorted(unique_numbers, key=int)
        cache_sizes_in_mib = [int(num) / (1024 * 1024) for num in sorted_numbers]

        ax = plt.gca()
        x = 0.55
        ax.text(x, .94, r"$\mathbf{L2\text{-}Cache}$ = " + f"{cache_sizes_in_mib[0]:.2f} MiB", transform=ax.transAxes,color='#E41A1C')#,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
        ax.text(
            x, .89,
            r"$\mathbf{L3\text{-}Cache\ per\ Thread}$ = " + f"{cache_sizes_in_mib[1]:.2f} MiB",
            transform=ax.transAxes,
            color='#377EB8'
        )


    g.map_dataframe(annotate)
    
    for ax in g.axes.flat:
        ax.set_ylabel('Merge throughput [M. tuples / s]')
        ax.set_xlabel('Merge fan-in')
        ax.grid(True, axis='y', linestyle='--', linewidth=1)
    
    # g.add_legend()
    # sns.move_legend(
    #     g, "lower center",
    #     bbox_to_anchor=(.425, 1), ncol=3, title=None, frameon=True
    # )
   
    g.savefig(args.output, dpi=500)
    # plt.show()

if __name__ == "__main__":
    main()
