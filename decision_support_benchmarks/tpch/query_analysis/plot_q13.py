import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from statistics import geometric_mean
import sys
from scipy.stats import gstd
from scipy.stats import gmean
import statistics
import glob
from matplotlib.lines import Line2D
import os

import re

# Define time multipliers in nanoseconds
time_units = {
    "ns": 1,  # Nanoseconds
    "µs": 1_000,  # Microseconds to nanoseconds
    "us": 1_000,  # ASCII microsecond (fallback)
    "ms": 1_000_000,  # Milliseconds to nanoseconds
    "s": 1_000_000_000,  # Seconds to nanoseconds
}

time_unit = "ns"
time_divide = time_units[time_unit]  # Nano to Milli


def parse_time_to_ns(time_str):
    pattern = r"(\d+)\s*(ns|µs|us|ms|s)"
    matches = re.findall(pattern, time_str)

    total_ns = sum(int(value) * time_units[unit] for value, unit in matches)
    return total_ns


def extract_operator_runtimes(text):
    total_time = int(text.split("|")[0].rstrip("ns"))
    # Match the section starting from "Operator step runtimes:" until the first period
    match = re.search(r"Operator step runtimes:\s*(.*?)(?=\.)", text)
    if not match:
        return []

    runtimes_section = match.group(1)

    # Split on ', ' to separate the name-time pairs
    pairs = re.findall(r"(\S+)\s+(.+?)(?:,|$)", runtimes_section)

    return [("Total", total_time)] + [
        (name, parse_time_to_ns(time.strip()))
        for name, time in pairs
        if parse_time_to_ns(time.strip()) > 0
    ]


def extract_filter_count(s: str) -> int:
    match = re.search(r"filtered:\s*(\d+)", s)
    if match:
        return int(match.group(1))
    else:
        return -1


def extract_input_table_data(text: str):
    pattern = (
        r"left_rows: (\d+), left_chunks: (\d+), right_rows: (\d+), right_chunks: (\d+)"
    )
    pattern2 = r"Radix bits: (\d+)"
    pattern3 = r"Output: (\d+) rows in (\d+) chunk"

    match = re.search(pattern, text)
    match2 = re.search(pattern2, text)
    match3 = re.search(pattern3, text)

    result = {
        "left_rows": 0,
        "left_chunks": 0,
        "right_rows": 0,
        "right_chunks": 0,
        "output_rows": 0,
        "output_chunks": 0,
        "radix_bits": 0,
    }
    if match:
        left_rows, left_chunks, right_rows, right_chunks = map(int, match.groups())
        result["left_rows"] = left_rows
        result["left_chunks"] = left_chunks
        result["right_rows"] = right_rows
        result["right_chunks"] = right_chunks
    if match2:
        result["radix_bits"] = int(match2.group(1))
    if match3:
        result["output_rows"] = int(match3.group(1))
        result["output_chunks"] = int(match3.group(2))
    return result


def extract_basic_info(prefix, dir, query):
    pattern = os.path.join(dir, prefix + f"{query}-*-PQP.txt")
    # Find all matching files
    matching_files = glob.glob(pattern)

    file_filter_counts = {}
    file_input_table_data = {}

    for file in matching_files:
        with open(file, "r") as f:
            filter_counts, input_tables = zip(
                *[
                    (
                        extract_filter_count(line),
                        extract_input_table_data(line),
                    )
                    for line in f
                    if "|" in line
                ]
            )
            filter_counts = list(filter_counts)
            file_filter_counts[file] = filter_counts
            file_input_table_data[file] = input_tables

    filter_counts = {}
    item_input_data = {}

    for file, filtered in file_filter_counts.items():
        item = 0
        for filtered_data in filtered:
            if item not in filter_counts:
                filter_counts[item] = []  # Initialize if item doesn't exist
            filter_counts[item].append(filtered_data)
            item += 1

    for file, input_data in file_input_table_data.items():
        item = 0
        for table_input in input_data:
            if item not in item_input_data:
                item_input_data[item] = {
                    "left_rows": [],
                    "left_chunks": [],
                    "right_rows": [],
                    "right_chunks": [],
                    "output_rows": [],
                    "output_chunks": [],
                    "radix_bits": [],
                }
            item_input_data[item]["left_rows"].append(table_input["left_rows"])
            item_input_data[item]["left_chunks"].append(table_input["left_chunks"])
            item_input_data[item]["right_rows"].append(table_input["right_rows"])
            item_input_data[item]["right_chunks"].append(table_input["right_chunks"])
            item_input_data[item]["output_rows"].append(table_input["output_rows"])
            item_input_data[item]["output_chunks"].append(table_input["output_chunks"])
            item_input_data[item]["radix_bits"].append(table_input["radix_bits"])
            item += 1

    return {"filter_counts": filter_counts, "table_input": item_input_data}


def extract_times_for_query(prefix, dir, query):
    pattern = os.path.join(dir, prefix + f"{query}-*-PQP.txt")
    # Find all matching files
    matching_files = glob.glob(pattern)

    file_times = {}

    for file in matching_files:
        with open(file, "r") as f:
            times = [extract_operator_runtimes(line) for line in f if "|" in line]
            times = list(times)
            file_times[file] = times

    item_times = {}

    for file, times in file_times.items():
        item = 0
        for time_data in times:
            for name, time in time_data:  # Loop over name, time pairs
                if item not in item_times:
                    item_times[item] = {}  # Initialize if item doesn't exist
                if name not in item_times[item]:
                    item_times[item][
                        name
                    ] = []  # Initialize the name key if it doesn't exist
                item_times[item][name].append(time)  # Accumulate the time for the name

            item += 1
    return {"item_times": item_times}


def fill_data(data, basic_info, times, label, query, scale_factor):
    for item_num, time in times["item_times"].items():
        filtered = int(
            statistics.mean(basic_info["filter_counts"][item_num])
        )  # We use mean as filtered sometimes contains zeros.
        left_rows = int(
            statistics.mean(basic_info["table_input"][item_num]["left_rows"])
        )
        left_chunks = int(
            statistics.mean(basic_info["table_input"][item_num]["left_chunks"])
        )
        right_rows = int(
            statistics.mean(basic_info["table_input"][item_num]["right_rows"])
        )
        right_chunks = int(
            statistics.mean(basic_info["table_input"][item_num]["right_chunks"])
        )
        output_rows = int(
            statistics.mean(basic_info["table_input"][item_num]["output_rows"])
        )
        output_chunks = int(
            statistics.mean(basic_info["table_input"][item_num]["output_chunks"])
        )
        radix_bits = int(
            statistics.mean(basic_info["table_input"][item_num]["radix_bits"])
        )
        for name, step_time in time.items():
            if name == "Total" or name == "GatherRowIds":
                continue
            mean = statistics.mean(step_time) / time_divide
            geo_mean = statistics.geometric_mean(step_time) / time_divide
            data["Join Type"].append(label)
            data["Operator Num"].append(item_num)
            data["Step"].append(name)
            data["Time"].append(mean / scale_factor)  # normalize by scale factor
            data["GeoMeanTime"].append(
                geo_mean / scale_factor
            )  # normalize by scale factor
            data["Query"].append(query)
            data["Filtered"].append(filtered)
            data["LeftRows"].append(left_rows)
            data["RightRows"].append(right_rows)
            data["OutputRows"].append(output_rows)
            data["LeftChunks"].append(left_chunks)
            data["RightChunks"].append(right_chunks)
            data["OutputChunks"].append(output_chunks)
            data["RadixBits"].append(radix_bits)
            data["ScaleFactor"].append(scale_factor)


def compute_mean_normalized(data, col, scale_column="ScaleFactor"):
    normalized = data[col] / data[scale_column]
    geo_std = gstd(normalized)
    return gmean(normalized), geo_std


def annotate(data, **kws):
    operator_time = sum(data["Time"])

    left_rows, lrows_std = compute_mean_normalized(data, "LeftRows")
    right_rows, rrows_std = compute_mean_normalized(data, "RightRows")
    output_rows, orows_std = compute_mean_normalized(data, "OutputRows")

    time_sums = data.groupby("ScaleFactor")["Time"].sum()
    operator_time = gmean(time_sums)
    operator_time_gstd = gstd(time_sums)
    ax = plt.gca()

    gap = 0.05
    x = 0.05
    y = 0.95

    ax.text(
        x,
        y,
        f"|R| = {int(left_rows)} * SF (GSD={lrows_std:.3f})",
        transform=ax.transAxes,
    )
    y -= gap
    ax.text(
        x,
        y,
        f"|S| = {int(right_rows)} * SF, (GSD={rrows_std:.3f}",
        transform=ax.transAxes,
    )
    y -= gap
    ax.text(
        x,
        y,
        f"|O| = {int(output_rows)} * SF (GSD={orows_std:.3f})",
        transform=ax.transAxes,
    )
    y -= gap
    ax.text(
        x,
        y,
        f"Total = SF * {operator_time:.2f} {time_unit} (GSD={operator_time_gstd:.3f})",
        transform=ax.transAxes,
    )


def plot(data, query, plot_name):
    df = pd.DataFrame(data)

    g = sns.FacetGrid(df, col="Join Type", row="Operator Num", height=4, aspect=1.5)
    g.map(
        sns.barplot,
        "Query",
        "Time",
        "Step",
        errorbar=lambda x: (x.min(), x.max()),
        estimator=gmean,
        err_kws={"color": "black", "alpha": 0.4, "linewidth": 2},
    )

    g.set_ylabels(label=f"SF normalized time [{time_unit}]")
    g.set_xlabels(label=f"")
    g.set_titles("{col_name}")
    g.map_dataframe(annotate)

   
    # Define the color mapping for each step (this is where the step_to_color comes in)
    step_to_color = {
        "BuildSideMaterializing": (0.1, 0.4, 0.6),
        "ProbeSideMaterializing": (0.8, 0.4, 0.05),
        "Clustering": (0.14, 0.5, 0.14),
        "Building": (0.67, 0.12, 0.12),
        "Probing": (0.46, 0.32, 0.59),
        "LeftSideMaterialize": (0.1, 0.4, 0.6),
        "RightSideMaterialize": (0.8, 0.4, 0.05),
        "LeftSidePartition": (0.44, 0.27, 0.24),
        "RightSidePartition": (0.71, 0.37, 0.64),
        "LeftSideSortBuckets": (0.44, 0.44, 0.44),
        "RightSideSortBuckets": (0.66, 0.67, 0.11),
        "GatherRowIds": (0.08, 0.6, 0.73),
        "FindJoinPartner": (0.1, 0.4, 0.6),
        "OutputWriting": (0.2, 0.2, 0.2),
    }

    g.fig.subplots_adjust(right=0.85)

    # Set the color for each bar based on its category
    for ax, (name, group) in zip(
        g.axes.flat, df.groupby(["Join Type", "Operator Num"])
    ):
        for idx, (p, step) in enumerate(zip(ax.patches, group["Step"])):
            p.set_facecolor(step_to_color[step])
            for container in ax.containers:
                labels = [f"{int(bar.get_height())}" for bar in container]
                ax.bar_label(container, labels=labels, padding=5)
        ax.legend()

    g.savefig(plot_name + ".png", dpi=500, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="name")
    args = parser.parse_args()

    query = "13"
    time_unit = "us"
    time_divide = time_units[time_unit]
    plot_name = "test"

    clean_times_path = "./"

    scale_factors = [10, 50, 100]

    data = {
        "Join Type": [],
        "Operator Num": [],
        "Step": [],
        "Time": [],
        "GeoMeanTime": [],
        "Query": [],
        "Filtered": [],
        "LeftRows": [],
        "RightRows": [],
        "LeftChunks": [],
        "RightChunks": [],
        "OutputRows": [],
        "OutputChunks": [],
        "RadixBits": [],
        "ScaleFactor": [],
    }

    prefix = "TPC-H_"
    for sf in scale_factors:
        dirs = ["hj_sf" + str(sf), "ssmj_sf" + str(sf), "ssmj_sf" + str(sf) + "_nb"]
        labels = ["HJ", "SSMJ", "SSMJ w/o Bloom Filter"]

        # Loop through the directories and labels
        for dir_path, label in zip(dirs, labels):
            times_dir_path = os.path.join(clean_times_path, dir_path)
            basic_info = extract_basic_info(prefix, dir_path, query)
            times = extract_times_for_query(prefix, times_dir_path, query)
            fill_data(data, basic_info, times, label, f"TPC-H {query}", sf)

    df = pd.DataFrame(data)
    plot(data, f"TPC-H {query}", args.output)
