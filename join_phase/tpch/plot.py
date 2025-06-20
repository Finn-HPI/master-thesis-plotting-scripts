import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import glob
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


def extract_join_time(text):
    # Regex to capture the pattern: 'FindJoinPartner' followed by time in ms, µs, etc.
    pattern = r"FindJoinPartner\s+(\d+)\s*(ms|µs|us|s|ns)\s*(\d+)\s*(ms|µs|us|s|ns)"
    matches = re.findall(pattern, text)

    # List to store all extracted times in nanoseconds
    times_in_ns = []

    for match in matches:
        # Extract the numbers and units
        number1 = int(match[0])
        unit1 = match[1]
        number2 = int(match[2])
        unit2 = match[3]

        # Convert the time values to nanoseconds
        time_in_ns = number1 * time_units[unit1] + number2 * time_units[unit2]

        # Append the result
        times_in_ns.append(time_in_ns)
    return times_in_ns[0]


def extract_times_for_query(dir, query):
    pattern = os.path.join(dir, f"TPC-H_{query}*-PQP.txt")
    matching_files = glob.glob(pattern)

    file_sums = {}

    for file in matching_files:
        with open(file, "r") as f:
            times = [extract_join_time(line) for line in f if "|" in line]
            file_sums[file] = sum(times)

    sum_values = list(file_sums.values())
    average_sum = sum(sum_values) / len(sum_values) if sum_values else 0

    return average_sum


def plot(data, plot_name):
    df = pd.DataFrame(data)
    sns.set_theme(style="white")

    # Define a consistent color palette for the algorithms
    algo_order = sorted(df["Algo"].unique())
    # palette = sns.color_palette("tab10", n_colors=len(algo_order))
    palette = ['#377eb8', '#ff7f00', '#4daf4a']
    color_mapping = dict(zip(algo_order, palette))

    # Create subplots: 1 row, 2 columns
    fig, ax = plt.subplots(
        1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [8, 0.75]}
    )

    # Bar plot per query
    ax3 = sns.barplot(
        data=df, x="Query", y="Time", hue="Algo", ax=ax[0], palette=color_mapping
    )
    ax3.tick_params(axis="both", which="both", bottom=True, top=False, labelsize=12)
    
    sns.move_legend(
        ax3, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=True, fontsize=12
    )

    # Cumulative times
    cumulative_times = df.groupby("Algo")["Time"].sum().reset_index()
    cumulative_times["Time"] = cumulative_times["Time"].astype(int)

    sns.barplot(
        data=cumulative_times,
        x="Algo",
        y="Time",
        hue="Algo",
        ax=ax[1],
        palette=color_mapping,
        order=[
            "Binary Search",
            "Linear (128) + Binary Search",
            "Exponential Search",
        ],
        dodge=False,  # No grouped bars, just one per Algo
    )

    
    ax[1].set_ylim(0, cumulative_times["Time"].max() * 1.1)
    ax[0].set_ylabel(f"Join phase time [{time_unit}]", fontsize=16)
    ax[0].set_xlabel("TPC-H queries", fontsize=16)
    ax[0].tick_params(axis="y", which="both", left=True)
    ax[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    ax[1].set(ylabel=None)
    ax[1].set(xlabel=f"Cum. times\n[{time_unit}]")
    ax[1].title.set_fontsize(10)

    ax[1].set_xticks([])
    ax[1].yaxis.set_tick_params(labelleft=False)

    sns.despine(
        fig=None,
        ax=ax[1],
        top=True,
        right=True,
        left=True,
        bottom=False,
        offset=None,
        trim=False,
    )
    sns.despine(
        fig=None,
        ax=ax[0],
        top=True,
        right=True,
        left=False,
        bottom=False,
        offset=None,
        trim=False,
    )

    # Annotate bars in the cumulative plot
    for p in ax[1].containers:
        ax[1].bar_label(p, fontsize=14, label_type='center', rotation=90, color='white',fontweight='semibold')

    plt.tight_layout()

    plt.savefig(plot_name, dpi=300)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", required=True, help="Output file name (e.g., plot.png)."
    )
    args = parser.parse_args()

    prefix = "TPC-H_"

    queries = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22]
    scale_factors = ["10"]
    # architectures = ["arm", "avx2", "avx512", "power"]

    data = {"Query": [], "Algo": [], "Time": []}

    time_unit = "ms"
    time_divide = time_units[time_unit]

    server = {
        "arm": "NVIDIA GH200 (ARM Neoverse-V2)",
        "avx2": "HPE XL225n Gen10 (AMD EPYC 7742)",
        "avx512": "Fujitsu RX2530 M5 (Intel Xeon Gold 5220S)",
        "power": "IBM S1024 (IBM Power10)",
    }

    for query in queries:
        query_name = f"{query:02d}"
        time_lin_bin = (
            extract_times_for_query("sf50_linear_and_binary_search", query_name) / time_divide
        )
        time_jump_search = (
            extract_times_for_query("sf50_exponential_search", query_name) / time_divide
        )
        time_binary_search = (
            extract_times_for_query("sf50_binary_search", query_name) / time_divide
        )

        for label, time in [
            ("Binary Search", time_binary_search),
            ("Linear (128) + Binary Search", time_lin_bin),
            ("Exponential Search", time_jump_search),
        ]:
            data["Query"].append(query_name)
            data["Algo"].append(label)
            data["Time"].append(time)

    plot(data, args.output)
