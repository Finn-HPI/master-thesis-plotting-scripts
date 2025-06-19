import seaborn as sns
import pandas as pd
import argparse
import glob
import os
from matplotlib.ticker import MaxNLocator

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


def points_to_data_units(ax, points):
    # Convert points to pixels
    fig = ax.figure
    dpi = fig.dpi
    pixels = points * dpi / 72  # 1 point = 1/72 inch

    # Get axis height in pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in_inches = bbox.height
    height_in_pixels = height_in_inches * dpi

    # Get y-axis limits and range
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Calculate data units per pixel
    data_per_pixel = y_range / height_in_pixels
    return pixels * data_per_pixel


def plot(data, plot_name):
    df = pd.DataFrame(data)

    g = sns.FacetGrid(
        df,
        col="Architecture",
        sharey=False,
        sharex=True,
        height=3,
        aspect=4,
        col_wrap=2,
        col_order=["System A", "System B", "System C", "System E"],
    )
    g.map_dataframe(
        sns.barplot,
        x="Scale Factor",
        y="Time",
        hue="Algo",
        errorbar=None,
        estimator=sum,
        width=0.925,
        order=["10", "50", "100"],
        hue_order=["HJ", "SSMJ", "SSMJ w/o Bloom Filter"],
        palette=["#e02b35", "#082a54", "#59a89c"],
    )
    label_fontsize = 26
    for arch, ax in g.axes_dict.items():
        # print(df)
        # print(df_cum)
        ax.bar_label(
            ax.containers[0],
            fmt="%.1f",
            fontsize=label_fontsize,
            rotation=0,
            weight="bold",
        )
        ax.bar_label(
            ax.containers[1],
            fmt="%.1f",
            fontsize=label_fontsize,
            rotation=0,
            weight="bold",
        )
        ax.bar_label(
            ax.containers[2],
            fmt="%.1f",
            fontsize=label_fontsize,
            rotation=0,
            weight="bold",
        )
        i = 0
        times = [[], [], [], []]
        for p in ax.patches:
            h, w, x = p.get_height(), p.get_width(), p.get_x()
            xy = (x + w / 2.0, h / 2)
            times[int(i / 3)].append(h)
            i += 1

        label_height_data = points_to_data_units(ax, label_fontsize)
        padding_data = points_to_data_units(ax, 50)  # extra 5 points padding
        ymin, ymax = ax.get_ylim()
        # Increase ymax by, say, 5-10% to add space above the highest data point
        new_ymax = ymax * 1.5
        ax.set_ylim(ymin, new_ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        
        # major_ticks = ax.get_yticks()
        # ax.set_yticks(major_ticks, minor=False)
     

        i = 0
        for p in ax.patches:
            h, w, x = p.get_height(), p.get_width(), p.get_x()
            rotation = 0 if h < 4.0 else 90
            labelsize = label_fontsize-3
            # xy = (x + w / 2.0, h / 2)
            xy = (x + w / 2.0, h + label_height_data + padding_data)
            if i < 3:
                if i == 0:
                    ax.annotate(
                    text="BL",
                    xy=xy,
                    color = "#717070",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=labelsize,
                    # fontweight="bold",
                )
                else:
                    ax.annotate(
                    text="BL",
                    xy=(x + w / 2.0, h / 2),
                    color = "#C9C9C9",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=labelsize,
                    # fontweight="bold",
                )
              

            elif i < 6:
                index = i - 3
                # Relative speedup: percentage reduction in execution time
                speedup = ((times[0][index] - times[1][index]) / times[0][index]) * 100
                ax.annotate(
                    text=f"{speedup:.1f}%",
                    xy=xy,
                    color="red" if speedup < 0 else "green",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=labelsize,
                    fontweight="bold",
                )
            elif i < 9:
                index = i - 6
                # Relative speedup: percentage reduction in execution time
                speedup = ((times[0][index] - times[2][index]) / times[0][index]) * 100
                ax.annotate(
                    text=f"{speedup:.1f}%",
                    xy=xy,
                    color="red" if speedup < 0 else "green",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=labelsize,
                    fontweight="bold",
                )
            i += 1

    g.set_titles(col_template="{col_name}", row_template="", size=35, fontweight="bold")
    g.set_ylabels(label=f"Time [{time_unit}]", fontsize=28)
    g.set_xlabels(label=f"Scale Factor", fontsize=28)

    for ax in g.axes.flat:
        ax.tick_params(axis="y", labelsize=26, width=2)
        ax.tick_params(axis="x", labelsize=26, width=4, length=5)

    # g.add_legend()
    # sns.move_legend(
    #     g,
    #     "lower center",
    #     bbox_to_anchor=(0.38, 1),
    #     ncol=3,
    #     title=None,
    #     frameon=True,
    # )
    # g.tight_layout()
    g.figure.subplots_adjust(wspace=0.075, hspace=1)
    g.savefig(plot_name, dpi=300, bbox_inches="tight")
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
