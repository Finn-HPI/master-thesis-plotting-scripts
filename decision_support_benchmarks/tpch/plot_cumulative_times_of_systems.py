import seaborn as sns
import pandas as pd
import argparse
import glob
import os

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


def plot(data, plot_name):
    df = pd.DataFrame(data)

    g = sns.FacetGrid(
        df,
        col="Architecture",
        sharey=False,
        sharex=True,
        height=3,
        aspect=1.27,
        # col_wrap=5,
        col_order=["System A", "System B", "System C", "System E"],
    )
    g.map_dataframe(
        sns.barplot,
        x="Scale Factor",
        y="Time",
        hue="Algo",
        errorbar=None,
        estimator=sum,
        order=["10", "50", "100"],
        hue_order=["HJ", "SSMJ", "SSMJ w/o Bloom Filter"],
        palette=["#e02b35", "#082a54", "#59a89c"],
    )
    label_fontsize = 7
    for arch, ax in g.axes_dict.items():
        # print(df)
        # print(df_cum)
        ax.bar_label(ax.containers[0], fmt="%.1f", fontsize=label_fontsize+1, rotation=0,weight='bold')
        ax.bar_label(ax.containers[1], fmt="%.1f", fontsize=label_fontsize+1, rotation=0,weight='bold')
        ax.bar_label(ax.containers[2], fmt="%.1f", fontsize=label_fontsize+1, rotation=0,weight='bold')
        i = 0
        times = [[], [], [], []]
        for p in ax.patches:
            h, w, x = p.get_height(), p.get_width(), p.get_x()
            xy = (x + w / 2.0, h / 2)
            times[int(i / 3)].append(h)
            i += 1

        i = 0
        for p in ax.patches:
            h, w, x = p.get_height(), p.get_width(), p.get_x()
            rotation = 0 if h < 4.0 else 90
            labelsize = label_fontsize - 1 if h < 4.0 else label_fontsize + 2
            xy = (x + w / 2.0, h / 2)
            if i < 3:
                if i == 0:
                    ax.annotate(
                        text="Bl.",
                        xy=xy,
                        color="#dad8d6",
                        ha="center",
                        va="center",
                        rotation=0,
                        fontsize=label_fontsize+1,
                        fontweight='bold'
                    )
                else:
                    ax.annotate(
                        text="Baseline",
                        xy=xy,
                        color="#dad8d6",
                        ha="center",
                        va="center",
                        rotation=rotation,
                        fontsize=labelsize,
                        fontweight='bold'
                    )
            elif i < 6:
                # if i == 3: continue
                index = i - 3
                speedup = times[0][index] / times[1][index]
                ax.annotate(
                    text=f"{speedup:.2f}x",
                    xy=xy,
                    color="white",
                    ha="center",
                    va="center",
                    rotation=rotation,
                    fontsize=labelsize,
                    fontweight="bold",
                )
            elif i < 9:
                index = i - 6
                speedup = times[0][index] / times[2][index]
                ax.annotate(
                    text=f"{speedup:.2f}x",
                    xy=xy,
                    color="white",
                    ha="center",
                    va="center",
                    rotation=rotation,
                    fontsize=labelsize,
                    fontweight="bold",
                )
            i += 1

    g.set_titles(col_template="{col_name}", row_template="", size=14, fontweight="bold")
    g.set_ylabels(label=f"Time [{time_unit}]", fontsize=14)
    g.set_xlabels(label=f"Scale Factor", fontsize=14)
    
    for ax in g.axes.flat:
        ax.tick_params(axis='both', labelsize=14)  # Change '10' to desired font size

    # g.add_legend()
    # sns.move_legend(
    #     g,
    #     "lower center",
    #     bbox_to_anchor=(0.38, 1),
    #     ncol=3,
    #     title=None,
    #     frameon=True,
    # )

    g.savefig(plot_name, dpi=500)
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
