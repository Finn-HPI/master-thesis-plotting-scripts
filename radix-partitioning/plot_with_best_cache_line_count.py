import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of directories containing CSV files
directories = ["avx2", "avx512", "arm", "power"]

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png). If not provided, the plot is shown.")
args = parser.parse_args()


name = {
    "power": r"$\bf{(a)}\ \bf{System}\ \bf{A}$" + "\n(IBM Power10, cache-lines = 1)",
    "avx2": r"$\bf{(b)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742, cache-lines = 1)",
    "arm": r"$\bf{(c)}\ \bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2, cache-lines = 4)",
    "avx512": r"$\bf{(d)}\ \bf{System}\ \bf{D}$"
    + "\n(Intel Xeon Gold 5220S, cache-lines = 2)",
}

# Read and combine data
csv_files = {
    "avx2": "radix_partition_1.csv",
    "avx512": "radix_partition_2.csv",
    "arm": "radix_partition_4.csv",
    "power": "radix_partition_1.csv",
}

# Used number of cache-lines per system
cache_lines = {
    "avx2": "1",
    "avx512": "2",
    "arm": "4",
    "power": "1",
}

# Define TLB limits of systems
tlb_l1 = {"avx2": 48, "avx512": 64, "arm": 64, "power": 64}
tlb_l2 = {"avx2": 2048, "avx512": 1536, "arm": 2048, "power": 4096}


hue_color = sns.color_palette("deep")

dfs = []

color_map = {directories[i]: hue_color[i] for i in range(len(directories))}
for df_part in dfs:
    df_part["bar_color"] = df_part["system"].map(color_map)

for directory in directories:
    csv_file = os.path.join(directory, csv_files[directory])
    df = pd.read_csv(csv_file)
    df["total_time"] = df["time_histogram"] + df["time_init"] + df["time_partition"]
    df["system"] = name[directory]
    df["tlb_l1"] = tlb_l1[directory]
    df["tlb_l2"] = tlb_l2[directory]
    df["cache_lines"] = cache_lines[directory]

    # Ensure num_partitions is treated as a numeric variable for correct positioning
    df["num_partitions"] = pd.to_numeric(df["num_partitions"], errors="coerce")

    dfs.append(df)

# Combine all data
data = pd.concat(dfs, ignore_index=True)


g = sns.FacetGrid(
    data,
    col="system",
    col_wrap=2,
    sharex=True,
    sharey=False,
    height=2.5,
    aspect=3,
    col_order=list(name.values()),
)
g.set_titles(col_template="{col_name}", size=14)

g.map_dataframe(sns.scatterplot, x="num_partitions", y="total_time", alpha=0)


plt.xscale("log", base=2)


def annotate(data, **kws):
    ax = plt.gca()

    # Get TLB L1 and L2 values
    tlb_l1_x = data["tlb_l1"].iloc[0]
    tlb_l2_x = data["tlb_l2"].iloc[0]
    
    y_pos = 0.85

    ax.axvline(
        x=tlb_l1_x,
        color=sns.color_palette("deep")[4],
        linestyle="--",
        label="L1 TLB entries",
        lw=2,
        ymax=y_pos,
    )
    ax.axvline(
        x=tlb_l2_x,
        color=sns.color_palette("deep")[3],
        linestyle="--",
        label="L2 TLB entries",
        lw=2,
        ymax=y_pos,
    )
    
    label_size = 12

    ax.annotate(
        "L1 TLB limit",
        xy=(tlb_l1_x, ax.get_ylim()[1] * 0.95),
        xytext=(0, 0),
        textcoords="offset points",
        rotation=0,
        va="center",
        ha="center",
        fontsize=label_size,
        weight='bold',
        color=sns.color_palette("deep")[4],
    )
    
    ax.annotate(
        "L2 TLB limit",
        xy=(tlb_l2_x, ax.get_ylim()[1] * 0.95),
        xytext=(0, 0),
        textcoords="offset points",
        rotation=0,
        va="center",
        ha="center",
        fontsize=label_size,
        weight='bold',
        color=sns.color_palette("deep")[3],
    )

    # Plot bars for each partition fan-out
    for i, row in data.iterrows():
        log_x = row["num_partitions"]
        width = log_x * 0.45
        ax.bar(
            row["num_partitions"],
            row["total_time"],
            width=width,
            bottom=0,
            color=sns.color_palette("deep")[0],
            edgecolor="none",
        )

    ax.set_xticks(
        sorted(data["num_partitions"].unique())
    )  # Ensure unique x-ticks from your data
    ax.set_xticklabels(
        [f"{int(x)}" for x in sorted(data["num_partitions"].unique())], ha="center"
    )


# Apply annotation to each FacetGrid subplot
g.map_dataframe(annotate)

g.set_axis_labels("Number of partitions", "Total time")
font_size = 8
for ax in g.axes.flat:
    ax.set_xlabel("")  # Remove individual y-axis titles
    ax.set_ylabel("")  # Remove individual y-axis titles
    font_size = ax.title.get_fontsize()

# plt.figtext(0.5, 0, "Number of cache-lines per buffer", ha="center", fontsize=font_size+4)
#     plt.figtext(0, 0.5, "Total Partitioning Time [ms]", va="center", rotation="vertical", fontsize=font_size+4)

plt.figtext(
    0.5,
    0,
    "Number of partitions (= required TLB entries)",
    ha="center",
    fontsize=20,
)
plt.figtext(0, 0.5, "Time [ms]", va="center", rotation="vertical", fontsize=20)

# plt.show()
g.savefig(args.output, dpi=800, bbox_inches="tight")
