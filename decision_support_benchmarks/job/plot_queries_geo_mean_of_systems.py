import seaborn as sns
import pandas as pd
import glob
import argparse
from matplotlib.patches import Patch
from scipy.stats import gmean
import matplotlib.ticker as ticker
import os
import re

time_units = {
    "ns": 1,  # Nanoseconds
    "µs": 1_000,  # Microseconds to nanoseconds
    "us": 1_000,  # ASCII microsecond (fallback)
    "ms": 1_000_000,  # Milliseconds to nanoseconds
    "s": 1_000_000_000,  # Seconds to nanoseconds
}

time_unit = "ns"
time_divide = time_units[time_unit]


def dummy_formatter(val, pos):
    return f"{int(val):>3}"  # Right-aligned 3-digit numbers


def extract_times_for_query(dir, query):
    pattern = os.path.join(dir, f"{query}*-PQP.txt")
    # Find all matching files
    matching_files = glob.glob(pattern)

    file_sums = {}

    for file in matching_files:
        with open(file, "r") as f:
            times = [int(line.split("|")[0].rstrip("ns")) for line in f if "|" in line]
            file_sums[file] = sum(times)

    sum_values = list(file_sums.values())
    average_sum = sum(sum_values) / len(sum_values) if sum_values else 0

    return average_sum


systems_ordered = ["System A", "System B", "System C", "System E"]

full_name = {
    "System A": r"$\bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "System B": r"$\bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "System C": r"$\bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "System E": r"$\bf{System}\ \bf{E}$" + "\n(Intel Xeon Platinum 8352Y)",
}


def plot(data, plot_name):
    df = pd.DataFrame(data)
    df = df.groupby(["Query", "Join Type"])["Time"].apply(gmean).reset_index()

    df.rename(columns={"Time": "GeoMeanTime"}, inplace=True)

    sns.set_theme(style="white")

    colwrap = 11
    g = sns.FacetGrid(
        df,
        col="Query",
        col_wrap=colwrap,
        margin_titles=True,
        sharey=False,
        sharex=True,
        height=2,
        aspect=1,
    )
    g.map_dataframe(
        sns.barplot,
        x="Join Type",
        y="GeoMeanTime",
        hue="Join Type",
        palette=["#e02b35", "#082a54", "#59a89c"],
        order=["HJ", "SSMJ", "SSMJ w/o Bloom Filter"],
    )
    g.set_titles(
        col_template="Q {col_name}", row_template="", fontsize=14, fontweight="bold"
    )
    for i, ax in enumerate(g.axes.flatten()):
        ax.tick_params(axis="y", which="both", left=True, top=False)
        ax.set_xticklabels([])
        ax.set_ylabel(f"Time [{time_unit}]")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(dummy_formatter))
        if i % colwrap == 0:
            ax.set_ylabel(f"Time [{time_unit}]")
        else:
            ax.set_ylabel("")

    custom_labels = ["HJ", "SSMJ", "SSMJ w/o Bloom Filter"]
    custom_colors = ["#e02b35", "#082a54", "#59a89c"]
    handles = [
        Patch(color=color, label=label)
        for color, label in zip(custom_colors, custom_labels)
    ]

    # Add legend to the FacetGrid
    g.add_legend(handles=handles)
    g.tight_layout()
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.274, 0.0725), frameon=False)

    g.savefig(plot_name, dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, help="Output file name (e.g., plot.png)."
    )
    args = parser.parse_args()

    data = {"Query": [], "Join Type": [], "Time": [], "System": []}

    time_unit = "ms"
    time_divide = time_units[time_unit]

    server = {
        "arm": "System C",
        "avx2": "System B",
        "nx05": "System E",
        "power": "System A",
    }

    files = os.listdir("avx2/hj")
    parts = [filename.split("-")[0] for filename in files if "-" in filename]

    def numeric_prefix(part):
        match = re.match(r"(\d+)", part)
        return int(match.group(1)) if match else float("inf")

    # Sort by numeric prefix, then lexicographically
    queries = sorted(set(parts), key=lambda p: (numeric_prefix(p), p))

    systems = ["arm", "avx2", "avx512", "power"]
    for system in systems:
        for query in queries:
            query_name = f"{query}-"
            time_hj = extract_times_for_query(system + "/hj", query_name) / time_divide
            time_ssmj = (
                extract_times_for_query(system + "/ssmj", query_name) / time_divide
            )
            time_ssmj_no_bf = (
                extract_times_for_query(system + "/ssmj_no_bf", query_name)
                / time_divide
            )

            for label, time in [
                ("HJ", time_hj),
                ("SSMJ", time_ssmj),
                ("SSMJ w/o Bloom Filter", time_ssmj_no_bf),
            ]:
                data["Query"].append(query)
                data["System"].append(system)
                data["Join Type"].append(label)
                data["Time"].append(time)

    plot(data, args.output)
