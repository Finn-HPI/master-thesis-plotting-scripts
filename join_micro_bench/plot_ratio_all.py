import os
import argparse
import pandas as pd
from scipy.stats import gmean
import seaborn as sns
import numpy as np
import re
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt

name = {"JoinHash": "HJ", "JoinSortMerge": "SMJ", "JoinSimdSortMerge": "SSMJ"}

system_name = {
    "cp03": r"$\bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "cx30": r"$\bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "ga02": r"$\bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "nx05": r"$\bf{System}\ \bf{E}$" + "\n(Intel Xeon Platinum 83252Y)",
}

algo_speedup = "SSMJ"
base_algo = "HJ"


def read_benchmark_file(filepath):
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
        df = df[df["name"].str.contains("_mean", na=False)]
        df["name"] = df["name"].str.replace("_mean", "", regex=False)
        df["algo"] = (
            df["name"].str.split("/").str[0].str.extract(r"<(.*?)>")[0].map(name)
        )

        # Define conditions for special cases
        is_small = df["name"].str.startswith("BM_Join_Small")
        is_medium = df["name"].str.startswith("BM_Join_Medium")

        # Compute 'R'
        df["R"] = np.select(
            [is_small, is_medium],
            [1 / 1024, 100 / 1024],
            default=df["name"].str.split("/").str[1].astype(float),
        )

        # Compute 'S'
        df["S"] = np.select(
            [is_small, is_medium],
            [
                df["R"] * df["name"].str.split("/").str[1].astype(float),
                df["R"] * df["name"].str.split("/").str[1].astype(float),
            ],
            default=df["R"] * df["name"].str.split("/").str[2].astype(float),
        )
        df["S_scale"] = np.select(
            [is_small, is_medium],
            [
                df["name"].str.split("/").str[1].astype(float),
                df["name"].str.split("/").str[1].astype(float),
            ],
            default=df["name"].str.split("/").str[2].astype(float),
        )
        df["ratio"] = df["S_scale"].apply(lambda scale: f"1:{int(scale)}")
    return df


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="")
args = parser.parse_args()

all_dfs = []
directories = ["cp03", "cx30", "ga02", "nx05"]
for split in [32, 64, 128]:
    for dir in directories:
        print(split, dir)
        df = read_benchmark_file(dir + "/join_" + str(split) + ".csv")
        df["system"] = system_name[dir]
        df["split"] = split
        df["sys"] = dir
        if (
            split != 32
        ):  # we only need it once for 0.000977 as we do not split the materialized data for this scale
            df = df[df["R"] > 0.001]
        all_dfs.append(df)


combined_df = pd.concat(all_dfs, ignore_index=True)
print(combined_df["R"])
print("draw plot")


def compute_speedup(data, **kwargs):
    data = data.copy()
    baseline = data[data["algo"] == base_algo][
        ["system", "S_scale", "cpu_time"]
    ].rename(columns={"cpu_time": "base_time"})

    # # Merge baseline back into the full DataFrame on the 'system' column
    data = data.merge(baseline, on=["system", "S_scale"], how="left")

    # # Compute speedup as percent
    data["speedup_pct"] = (
        (data["base_time"] - data["cpu_time"]) / data["base_time"] * 100
    )

    # Only keep the algorithms we want to show
    algos = [algo_speedup]
    plot_data = data[data["algo"].isin(algos)]
    # info = plot_data[plot_data['sys']=='ga02']
    # print(info[['sys','R', 'S_scale', 'speedup_pct']])

    # info2 = plot_data[plot_data['sys']=='ga02']
    # print(info2[['sys','R', 'S_scale', 'speedup_pct']])
    # print("mean:", np.mean(info2['speedup_pct']))

    markers = ["o", "s", "D", "^", "P"]
    sns.pointplot(
        data=plot_data,
        x="ratio",
        y="speedup_pct",
        hue="system",
        markersize=5,
        linewidth=2,
        markers=markers,
    )
    plt.axhline(0, color="black", linewidth=1, linestyle="--")  # Zero baseline


g = sns.FacetGrid(
    combined_df, row="R", col="split", sharex=True, sharey=False, height=1.75, aspect=3,
)
g.map_dataframe(compute_speedup)
g.set_titles("{row_name}|{col_name}")

for (i,j,k), data in g.facet_data():
    if data.empty:
        ax = g.facet_axis(i, j)
        ax.set_axis_off()
        ax.set_title('')

for i, ax in enumerate(g.axes.flatten()):

    title = ax.get_title()

    def format_unit(value):
        value = float(value.split("|")[0])
        if value < 1:
            value *= 1024
            unit = "K"
        else:
            unit = "M"
        return f"{value:.0f} {unit}" if value.is_integer() else f"{value:.2f} {unit}"

    if len(title) > 0:
        R = format_unit(title)
        splits = title.split("|")[1]
        new_title = f"|R| = {R}"
        if R != "1 K":
            new_title += f", {splits} parts"
        if R == "100 K":
            new_title += " (ratio >= 1:16)"
        ax.set_title(new_title)

    tick_texts = ax.get_xticklabels()

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    # ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", which="both", bottom=True, top=False)
    ax.tick_params(axis="y", which="both", left=True, top=False, width=1.5)

    major_ticks = ax.get_yticks()
    # Compute minor ticks (midpoints between major ticks)
    minor_ticks = (major_ticks[:-1] + major_ticks[1:]) / 2

    # # Add minor ticks
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks, minor=False)
    ymin, ymax = ax.get_ylim()
    # ax.set_ylim(bottom=ymin, top=ymax)

    # # Customize minor tick appearance (shorter lines)
    ax.tick_params(axis="y", which="minor", length=3, width=1)
    ax.grid(True, axis="y", linestyle="-", alpha=0.5, linewidth=1)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.5, linewidth=1)


# g.set_axis_labels("|R| : |S|", "")
g.set_axis_labels("", "")
plt.figtext(0.425, 0, "|R|:|S|", va="center", fontsize=14)
plt.figtext(0, 0.5, "Speedup [%]", va="center", rotation="vertical", fontsize=14)

g.add_legend()
sns.move_legend(g, "upper left", bbox_to_anchor=(.3, 1),ncol=2,fontsize='16')
# sns.move_legend(
#     g,
#     "lower center",
#     bbox_to_anchor=(0.425, 1),
#     ncol=4,
#     title=None,
#     frameon=False,
# )

g.tight_layout()
# plt.show()
g.savefig(args.output, dpi=500)
