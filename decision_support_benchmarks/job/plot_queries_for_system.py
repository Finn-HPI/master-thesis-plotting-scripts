import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import statistics
from matplotlib.patches import Patch
import glob
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from statistics import geometric_mean
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
time_divide = time_units[time_unit]  # Nano to Milli


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

    sns.set_theme(style="white")

    g = sns.FacetGrid(
        df,
        col="Query",
        col_wrap=11,
        margin_titles=True,
        sharey=False,
        sharex=True,
        height=2,
        aspect=1,
    )
    g.map_dataframe(
        sns.barplot,
        x="Join Type",
        y="Time",
        hue="Join Type",
        palette=["#e02b35", "#082a54", "#59a89c"],
        order=["HJ", "SSMJ", "SSMJ w/o Bloom Filter"],
    )
    g.set_titles(col_template="Q {col_name}", row_template="", fontweight="bold")
    for ax in g.axes.flatten():
        ax.tick_params(axis="y", which="both", left=True, top=False)
        ax.set_xticklabels([])
        ax.set_ylabel(f"Time [{time_unit}]")

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

    g.savefig(plot_name)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True, help="system directory")
    parser.add_argument("--output", type=str, required=True, help="output")
    args = parser.parse_args()

    prefix = "TPC-H_"
    data = {"Query": [], "Join Type": [], "Time": []}

    time_unit = "ms"
    time_divide = time_units[time_unit]

    server = {
        "arm": "System C",
        "avx2": "System B",
        "nx05": "System E",
        "power": "System A",
    }

    arch = args.system
    print("Viusalization for system:", arch)

    files = os.listdir(arch + "/hj")
    parts = [filename.split("-")[0] for filename in files if "-" in filename]
    queries = sorted(set(parts))

    for query in queries:
        query_name = f"{query}-"
        time_hj = extract_times_for_query(arch + "/hj", query_name) / time_divide
        time_ssmj = extract_times_for_query(arch + "/ssmj", query_name) / time_divide
        time_ssmj_no_bf = (
            extract_times_for_query(arch + "/ssmj_no_bf", query_name) / time_divide
        )

        for label, time in [
            ("HJ", time_hj),
            ("SSMJ", time_ssmj),
            ("SSMJ w/o Bloom Filter", time_ssmj_no_bf),
        ]:
            data["Query"].append(query)
            # data["Architecture"].append(server[arch])
            data["Join Type"].append(label)
            data["Time"].append(time)

    plot(data, args.output)
