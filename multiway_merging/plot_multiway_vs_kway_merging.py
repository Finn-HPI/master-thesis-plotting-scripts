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
        algo = algo.replace(
            {"MWAY_MERGE": "Multiway Merge", "KWAY_MERGE": "K-Way Merge"}
        )
        size_str = df["name"].str.split("/").str[3]

        # Convert to numeric (bytes)
        size_bytes = pd.to_numeric(size_str, errors="coerce")

        # Compute size in MiB
        size_mib = size_bytes / (2**20)

        # Compute sorted unique values (excluding NaNs)
        sorted_uniques = np.sort(np.unique(size_mib[~np.isnan(size_mib)]))

        # Initialize empty labels
        cache_label = np.full_like(size_mib, "", dtype=object)

        # Assign labels using sorted unique values
        cache_label[size_mib == sorted_uniques[0]] = " (L2 Cache)"
        cache_label[size_mib == sorted_uniques[1]] = " (L3 Cache)"

        df["Algo"] = algo + cache_label

    return df


name = {
    "power_merge_bench.csv": r"$\bf{(a)}\ \bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "avx2_merge_bench.csv": r"$\bf{(b)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "arm_merge_bench.csv": r"$\bf{(c)}\ \bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "avx512_merge_bench.csv": r"$\bf{(d)}\ \bf{System}\ \bf{E}$"
    + "\n(Intel Xeon Platinum 8352Y)",
}

file_paths = [
    "avx512_merge_bench.csv",
    "avx2_merge_bench.csv",
    "arm_merge_bench.csv",
    "power_merge_bench.csv",
]

def format_mib(value):
    return f"{value:.0f}" if value.is_integer() else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, help="Output file name (e.g., plot.png)."
    )
    args = parser.parse_args()

    df_list = []
    for file_path in file_paths:
        print(file_path)
        df = read_benchmark_to_dataframe(file_path)
        df["System"] = name[os.path.basename(file_path)]
        df_list.append(df)

    # Concatenate into a single DataFrame
    unified_df = pd.concat(df_list, ignore_index=True)
    # palette = ["#1f77b4", "#6baed6", "#ff7f0e"]
    palette = ["#298c8c", "#a00000", "#646464"]

    g = sns.FacetGrid(
        unified_df,
        col="System",
        col_wrap=2,
        height=4,
        aspect=2,
        col_order=list(name.values()),
        sharey=False,
    )
    g.map_dataframe(
        sns.pointplot,
        x="Fan-in",
        y="Throughput",
        hue="Algo",
        markersize=5,
        markers=["o", "s", "^", "v", "X", "P"],
        hue_order=[
            "Multiway Merge (L2 Cache)",
            "Multiway Merge (L3 Cache)",
            "K-Way Merge",
        ],
        palette=palette,
    )
    g.set_titles(
        col_template="{col_name}", row_template="", size=16
    )  # , weight="bold")

    def annotate(data, **kws):
        # Filter rows that start with 'BM_MWAY_MERGE'
        filtered_data = data[data["name"].str.startswith("BM_MWAY_MERGE")]

        # Split the 'name' column by '/' and extract group 3 (index 2)
        extracted_numbers = filtered_data["name"].str.split("/").str[3]

        # Convert the extracted group to unique numbers
        unique_numbers = extracted_numbers.unique()
        sorted_numbers = sorted(unique_numbers, key=int)
        cache_sizes_in_mib = [int(num) / (1024 * 1024) for num in sorted_numbers]

        ax = plt.gca()
        x = 0.1
        y = 0.825
        ax.text(
            x,
            y + 0.08,
            r"$\mathbf{L2\text{-}Cache}$ = " + f"{format_mib(cache_sizes_in_mib[0])} MiB",
            transform=ax.transAxes,
            color=palette[0],fontsize=14,
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
        )
        ax.text(
            x,
            y,
            r"$\mathbf{L3\text{-}Cache}$ = " + f"{format_mib(cache_sizes_in_mib[1])} MiB",
            transform=ax.transAxes,
            color=palette[1],fontsize=14,
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
        )

    g.map_dataframe(annotate)

    font_size = 16
    for ax in g.axes.flat:
        ax.set_xlabel("")  # Remove individual y-axis titles
        ax.set_ylabel("")  # Remove individual y-axis titles
        font_size = ax.title.get_fontsize()
        ax.grid(True, axis="y", linestyle="--", linewidth=1)
        # ax.tick_params(axis="y", which="both", left=True, labelsize=14)  # Ensure y-ticks are visible
        ax.tick_params(axis="x", which="both", bottom=True, labelsize=14)  # Ensure x-ticks are visible

    # Manually set global axis labels
    plt.figtext(0.5, 0, "Fan-in (fan-in sorted input lists of size $16 \\cdot 2^{20}~/~\\text{fan-in}$)", ha="center", fontsize=font_size + 4)
    plt.figtext(
        0,
        0.5,
        "Throughput [M. tuples / s]",
        va="center",
        rotation="vertical",
        fontsize=font_size + 4,
    )
    

    # g.add_legend()
    # sns.move_legend(
    #     g, "lower center", bbox_to_anchor=(0.425, 1), ncol=3, title=None, frameon=True
    # )

    g.savefig(args.output, dpi=500)
    # plt.show()


if __name__ == "__main__":
    main()
