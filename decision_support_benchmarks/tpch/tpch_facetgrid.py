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
    print(df.head())

    sns.set_theme(style="white")

    g = sns.FacetGrid(
        df,
        col="Scale Factor",
        row="Architecture",
        margin_titles=True,
        sharey=False,
        sharex=True,
        height=2.5,
        aspect=3,
        row_order=systems_ordered,
    )

    g.map_dataframe(
        sns.barplot,
        x="Query",
        y="Time",
        hue="Algo",
        palette=["#e02b35", "#082a54", "#59a89c"],
    )
    g.set_ylabels(label=f"Time [{time_unit}]")
    g.set_titles(
        col_template="SF {col_name}", row_template="", size=18, fontweight="bold"
    )

    for ax in g.axes.flatten():
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.tick_params(axis="y", which="both", left=True, top=False)

    ax = g.axes[0, 0]
    title_text_obj = ax.title
    fontsize = title_text_obj.get_fontsize()
    row_names = g.row_names
    for i, row_val in enumerate(row_names):
        ax = g.axes[i, 0]  # Leftmost subplot in this row
        # Add text outside the plot area
        ax.text(
            -0.18,
            0.5,
            full_name[row_val],
            transform=ax.transAxes,
            ha="center",
            rotation=90,
            va="center",
            fontsize=13,
        )

    # Uncomment to add cumulative join times!

    # # Add cumulative bar plot as an inset in the upper right
    # lable_fontsize = 10
    # for (arch, sf), ax in g.axes_dict.items():
    #     df_sub = df[(df["Scale Factor"] == sf) & (df["Architecture"] == arch)]
    #     df_cum = df_sub.groupby("Algo", as_index=False)["Time"].sum()

    #     # Create an inset axis in the upper-right corner
    #     ax_inset = ax.inset_axes(
    #         [1.075, 0, 0.1, 0.9]
    #     )  # , loc="lower right", borderpad=1)

    #     sns.barplot(
    #         data=df_cum,
    #         y="Time",
    #         x="Algo",
    #         palette=["#e02b35", "#082a54", "#59a89c"],
    #         ax=ax_inset,
    #         width=0.8,
    #     )

    #     min_y = 999999
    #     for bar, time, algo in zip(ax_inset.patches, df_cum["Time"], df_cum["Algo"]):
    #         min_y = min(min_y, bar.get_y() + bar.get_height() / 2 + 0.2)

    #     # ax_inset.bar_label(ax_inset.containers[0], fontsize=10, rotation=90)
    #     # ax_inset.bar_label(ax_inset.containers[1], fontsize=10, rotation=90)
    #     # ax_inset.bar_label(ax_inset.containers[2], fontsize=10, rotation=90)

    #     # Annotate cumulative times on bars
    #     for bar, time, algo in zip(ax_inset.patches, df_cum["Time"], df_cum["Algo"]):

    #         if algo in ["SSMJ", "SSMJ w/o BF"]:
    #             hj_time = df_cum[df_cum["Algo"] == "HJ"]["Time"].values
    #             if hj_time.size > 0:  # Check if HJ exists
    #                 hj_time = hj_time[0]
    #                 speedup = hj_time / time
    #                 if speedup < 1:
    #                     ax_inset.text(bar.get_x() + bar.get_width() / 2,
    #                               min_y,
    #                               f"{speedup:.2f}x",
    #                               ha="center", va="center", fontsize=lable_fontsize, color="#a00000", fontweight="bold", rotation=90)
    #                 else:
    #                     ax_inset.text(bar.get_x() + bar.get_width() / 2,
    #                               min_y,
    #                               f"{speedup:.2f}x",
    #                               ha="center", va="center", fontsize=lable_fontsize, color='lightgreen', fontweight="bold", rotation=90)
    #         else:
    #              ax_inset.text(bar.get_x() + bar.get_width() / 2,
    #                               min_y ,
    #                               "1x",
    #                               ha="center", va="center", fontsize=lable_fontsize, fontweight="bold", rotation=90)

    #     # # Keep inset clean
    #     ax_inset.set_xticklabels([])
    #     ax_inset.tick_params(axis='y', left=True)
    #     ax_inset.set_xlabel(f"Cum. times\n[{time_unit}]", fontsize=8, fontweight="bold")
    #     ax_inset.set_ylabel("")
    #     ax_inset.spines["top"].set_visible(False)
    #     ax_inset.spines["right"].set_visible(False)

    g.fig.tight_layout()
    g.add_legend()
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.45, 1.1), ncol=3, frameon=True)
    g.tight_layout()

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

                # for label, time in [('HJ', time_hj), ('SSMJ', time_ssmj), ('SSMJ w/o BF', time_ssmj_no_bf)]:
                for label, time in [
                    ("HJ", time_hj),
                    ("SSMJ", time_ssmj),
                    ("SSMJ w/o Bloom Filter", time_ssmj_no_bf),
                ]:
                    data["Query"].append(query_name)
                    data["Scale Factor"].append(sf)
                    data["Architecture"].append(server[arch])
                    data["Algo"].append(label)
                    data["Time"].append(time)

    plot(data, args.output)
