import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects
import glob
from matplotlib.colors import to_rgba
from scipy.stats import gmean
import os
import argparse

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
    pattern = os.path.join(dir, f"TPC-H_{query}*-PQP.txt")
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

    # Group by Query and Algo, compute geometric mean of Time
    df = (
        df.groupby(["Query", "Scale Factor", "Algo"])["Time"].apply(gmean).reset_index()
    )

    # Rename the column to reflect the aggregated value
    df.rename(columns={"Time": "GeoMeanTime"}, inplace=True)

    hj_times = df[df["Algo"] == "HJ"][["Query", "Scale Factor", "GeoMeanTime"]]
    hj_times = hj_times.rename(columns={"GeoMeanTime": "HJ_GeoMeanTime"})

    # 2. Merge HJ baseline back into the full dataframe
    df = df.merge(hj_times, on=["Query", "Scale Factor"], how="left")

    # 3. Normalize
    df["NormalizedTime"] = df["GeoMeanTime"] / df["HJ_GeoMeanTime"]
    print(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    sns.set_theme(style="white")

    subset = df[(df["Algo"] == "HJ") & (df["Scale Factor"] == "100")]
    query_order = (
        subset.groupby("Query")["GeoMeanTime"].mean().sort_values().index.tolist()
    )

    colwrap = 5
    g = sns.FacetGrid(
        df,
        col="Query",
        sharey=False,
        sharex=True,
        height=2,
        aspect=1.5,
        col_wrap=5,
        col_order=query_order,
    )

    g.map_dataframe(
        sns.pointplot,
        x="Scale Factor",
        y="NormalizedTime",
        hue="Algo",
        hue_order=["SSMJ", "SSMJ w/o Bloom Filter", "HJ"],
        palette=["#082a54", "#59a89c", "#e02b35"],
        markers=["D", "o", "^"],
        order=["10", "50", "100"],
        markersize=colwrap,
        linewidth=2.5,
    )
    g.fig.subplots_adjust(hspace=-6, wspace=-1)

    g.set_titles(
        col_template="Q {col_name}", row_template="", size=14, fontweight="bold"
    )

    for i, ax in enumerate(g.axes.flatten()):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.tick_params(axis="y", which="both", left=True, top=False, width=1.5)

        if i % colwrap == 0:
            ax.set_ylabel(f"Time / Time_HJ", fontsize=10)
        else:
            ax.set_ylabel("")

        major_ticks = ax.get_yticks()
        # Compute minor ticks (midpoints between major ticks)
        minor_ticks = (major_ticks[:-1] + major_ticks[1:]) / 2

        # Add minor ticks
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, minor=False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=max(ymin, 0), top=ymax)

        # Customize minor tick appearance (shorter lines)
        ax.tick_params(axis="y", which="minor", length=3, width=1)
        ax.grid(True, axis="y", linestyle="-", alpha=0.5, linewidth=1.5)
        ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.5, linewidth=1.5)

        query_title = ax.get_title().replace("Q ", "")
        query = int(query_title)

        # Filter the relevant HJ data for this query
        hj_data = df[(df["Query"] == query) & (df["Algo"] == "HJ")]

        for _, row in hj_data.iterrows():
            x = row["Scale Factor"]
            y = row["NormalizedTime"]
            hj_time = row["HJ_GeoMeanTime"]

            text = ax.annotate(
                f"{hj_time:.2f} {time_unit}",
                xy=(x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#a65358",
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=2, foreground=to_rgba("white", alpha=0.5)
                    ),
                    path_effects.Normal(),
                ]
            )

    g.add_legend()
    g.tight_layout()
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.715, 0.18), frameon=False)

    g.savefig(plot_name, dpi=500, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, help="Output file name (e.g., plot.png)."
    )
    args = parser.parse_args()

    prefix = "TPC-H_"

    queries = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22]
    scale_factors = ["10", "50", "100"]
    architectures = ["arm", "avx2", "avx512", "power"]

    data = {"Query": [], "Scale Factor": [], "Architecture": [], "Algo": [], "Time": []}

    time_unit = "s"
    time_divide = time_units[time_unit]

    server = {
        "arm": "System C",
        "avx2": "System B",
        "avx512": "System E",
        "power": "System A",
    }

    for sf in scale_factors:
        for arch in architectures:
            for query in queries:
                query_name = f"{query:02d}"
                time_hj = (
                    extract_times_for_query(arch + "/hj_sf" + sf, query_name)
                    / time_divide
                )
                time_ssmj = (
                    extract_times_for_query(arch + "/ssmj_sf" + sf, query_name)
                    / time_divide
                )
                time_ssmj_no_bf = (
                    extract_times_for_query(arch + "/ssmj_sf" + sf + "_nb", query_name)
                    / time_divide
                )

                for label, time in [
                    ("HJ", time_hj),
                    ("SSMJ", time_ssmj),
                    ("SSMJ w/o Bloom Filter", time_ssmj_no_bf),
                ]:
                    data["Query"].append(query)
                    data["Scale Factor"].append(sf)
                    data["Architecture"].append(server[arch])
                    data["Algo"].append(label)
                    data["Time"].append(time)

    plot(data, args.output)
