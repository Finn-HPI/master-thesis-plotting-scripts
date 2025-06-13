import os
import argparse
import pandas as pd
from scipy.stats import gmean
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt

from pylab import rcParams
import matplotlib as mpl


name = {"JoinHash": "HJ", "JoinSortMerge": "SMJ", "JoinSimdSortMerge": "SSMJ"}

system_name = {
    "cp03": r"$\bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "cx30": r"$\bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "ga02": r"$\bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "nx05": r"$\bf{System}\ \bf{E}$" + "\n(Intel Xeon Platinum 83252Y)",
}


speedup_algo = "SSMJ"
base_algo = "HJ"
file = "join_32.csv"
# file = "join_64.csv"
# file = "join_128.csv"

palette = sns.color_palette()
sys_color = {
    "cp03": palette[0],
    "cx30": palette[1],
    "ga02": palette[2],
    "nx05": palette[3],
}


def format_unit(value):
    value = float(value)
    if value < 1:
        value *= 1024
        unit = "K"
    else:
        unit = "M"
    return f"{value:.0f} {unit}" if value.is_integer() else f"{value:.2f} {unit}"


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
    df["join"] = df.apply(
        lambda row: f"{format_unit(row['R'])}", axis=1
    )
    return df[df["R"] == df["S"]]


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="")
args = parser.parse_args()

all_dfs = []
directories = ["cp03", "cx30", "ga02", "nx05"]
for dir in directories:
    df = read_benchmark_file(dir + "/" + file)
    df["system"] = system_name[dir]
    df["sys"] = dir
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
baseline = df[df["algo"] == base_algo][["system", "R", "cpu_time"]]
baseline = baseline.rename(columns={"cpu_time": "baseline_cpu_time"})

# Step 2: Merge with original DataFrame
df = df.merge(baseline, on=["system", "R"], how="left")

# Step 3: Compute percent speedup
df["speedup_percent"] = (
    (df["baseline_cpu_time"] - df["cpu_time"]) / df["baseline_cpu_time"]
) * 100
df = df[df["algo"] == speedup_algo]

# mpl.rcParams["figure.figsize"] = (12, 2)

# Use constrained_layout
# fig, ax = plt.subplots(constrained_layout=True)

# sns.set(style="whitegrid")


def plot_speedup(data, **kwargs):
    system = data["sys"].iloc[0]
    print('system:', system)
    print(data[['join', 'speedup_percent']])
    print('\n')
    ax = sns.pointplot(
        data=data,
        x="join",
        y="speedup_percent",
        markersize=5,
        linewidth=2,
        color=sys_color[system],
        order=["1 K", "100 K", "1 M", "8 M", "16 M"],
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")  # Zero baseline
    ax.set_xlabel("")
    if system == "cp03":
        ax.set_ylabel("Speedup [%]", fontsize=18)
    else:
        ax.set_ylabel("", fontsize=14)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    # ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=14)
    ax.tick_params(
        axis="y", which="both", left=True, top=False, width=1.25, labelsize=14
    )

    major_ticks = ax.get_yticks()
    # major_ticks = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    minor_ticks = (major_ticks[:-1] + major_ticks[1:]) / 2

    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks, minor=False)

    ax.tick_params(axis="y", which="minor", length=3, width=1)
    ax.grid(True, axis="y", linestyle="-", linewidth=1.25)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.5, linewidth=1)


g = sns.FacetGrid(
    df, col="system", hue="system", sharex=True, sharey=False, height=2, aspect=2
)
g.map_dataframe(plot_speedup)
g.set_titles("{col_name}", size=16)

# plt.figtext(0.425, -0.05, f"Speedup of {speedup_algo} over {base_algo}", ha="center", fontsize=16)

g.savefig(args.output, dpi=500, bbox_inches="tight")
