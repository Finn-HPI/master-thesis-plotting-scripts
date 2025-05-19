import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statistics import geometric_mean
import sys
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


def fill_data(data, basic_info, times, label, query):
    for item_num, time in times["item_times"].items():
        filtered = int(
            statistics.mean(basic_info["filter_counts"][item_num])
        )  # We use mean as filtered sometimes contains zeros.
        left_rows = int(statistics.mean(basic_info["table_input"][item_num]["left_rows"]))
        left_chunks = int(
            statistics.mean(basic_info["table_input"][item_num]["left_chunks"])
        )
        right_rows = int(statistics.mean(basic_info["table_input"][item_num]["right_rows"]))
        right_chunks = int(
            statistics.mean(basic_info["table_input"][item_num]["right_chunks"])
        )
        output_rows = int(
            statistics.mean(basic_info["table_input"][item_num]["output_rows"])
        )
        output_chunks = int(
            statistics.mean(basic_info["table_input"][item_num]["output_chunks"])
        )
        radix_bits = int(statistics.mean(basic_info["table_input"][item_num]["radix_bits"]))
        for name, step_time in time.items():
            if name == "Total":
                continue
            mean = statistics.mean(step_time) / time_divide
            geo_mean = statistics.geometric_mean(step_time) / time_divide
            data["Join Type"].append(label)
            data["Operator Num"].append(item_num)
            data["Step"].append(name)
            data["Time"].append(mean)
            data["GeoMeanTime"].append(geo_mean)
            data["Query"].append(query)
            data["Filtered"].append(filtered)
            data["LeftRows"].append(left_rows)
            data["RightRows"].append(right_rows)
            data["OutputRows"].append(output_rows)
            data["LeftChunks"].append(left_chunks)
            data["RightChunks"].append(right_chunks)
            data["OutputChunks"].append(output_chunks)
            data["RadixBits"].append(radix_bits)


def annotate(data, **kws):
    operator_time = sum(data['Time'])
    filtered = data["Filtered"].values[0]
    left_rows = data["LeftRows"].values[0]
    left_chunks = data["LeftChunks"].values[0]
    right_rows = data["RightRows"].values[0]
    right_chunks = data["RightChunks"].values[0]
    output_rows = data["OutputRows"].values[0]
    output_chunks = data["OutputChunks"].values[0]
    radix_bits = data["RadixBits"].values[0]
    ax = plt.gca()

    gap = 0.05
    x = 0.05
    y = 0.95

    ax.text(
        x, y, f"|R|={left_rows:,} in {left_chunks:,} chunks", transform=ax.transAxes
    )
    y -= gap
    ax.text(
        x, y, f"|S|={right_rows:,} in {right_chunks:,} chunks", transform=ax.transAxes
    )
    y -= gap
    ax.text(
        x, y, f"|O|={output_rows:,} in {output_chunks:,} chunks", transform=ax.transAxes
    )
    y -= gap
    if filtered > 0:
        ax.text(x, y, f"Bloom Filtered = {filtered:,}", transform=ax.transAxes)
        y -= gap
    ax.text(x, y, f"Radix bits = {radix_bits}", transform=ax.transAxes)
    y -= gap
    ax.text(
        x, y, f"Time = {operator_time:.2f} {time_unit}", transform=ax.transAxes
    )


def plot(data, query, plot_name):
    df = pd.DataFrame(data)

    g = sns.FacetGrid(df, row="Join Type", col="Operator Num", height=4, aspect=1.5)
    g.map(sns.barplot, "Query", "Time", "Step")
    g.set_ylabels(label=f"Time [{time_unit}]")
    g.set_xlabels(label=f"")
    g.set_titles("[{row_name}] Join {col_name}")
    g.map_dataframe(annotate)

    # Define categories and the steps associated with each category
    hj_steps = [
        "BuildSideMaterializing",
        "ProbeSideMaterializing",
        "Clustering",
        "Building",
        "Probing",
    ]
    partition_steps = [
        "LeftSideMaterialize",
        "RightSideMaterialize",
        "LeftSidePartition",
        "RightSidePartition",
        "LeftSideSortBuckets",
        "RightSideSortBuckets",
        "GatherRowIds",
        "FindJoinPartner",
    ]
    output_writing_steps = ["OutputWriting"]

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

    # Create a dictionary to map steps to categories
    step_to_category = {}
    for step in hj_steps:
        step_to_category[step] = "HJ"
    for step in partition_steps:
        step_to_category[step] = "SSMJ"  # You can change this category name if needed
    for step in output_writing_steps:
        step_to_category[step] = "OutputWriting"

    g.fig.subplots_adjust(right=0.85)

    # Set the color for each bar based on its category
    for ax, (name, group) in zip(
        g.axes.flat, df.groupby(["Join Type", "Operator Num"])
    ):
        for idx, (p, step) in enumerate(zip(ax.patches, group["Step"])):
            # Set the color of the current bar based on its step
            p.set_facecolor(step_to_color[step])

            # Add annotation with the 'Time' value
            ax.annotate(
                f'{group["Time"].iloc[idx]:.2f} {time_unit}',
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=0,
            )

            # Add a point marking the geometric mean
            geo_mean_time = group["GeoMeanTime"].iloc[
                idx
            ]  # Get geometric mean time for this step
            ax.scatter(
                p.get_x() + p.get_width() / 2.0,
                geo_mean_time,
                color="red",
                marker="*",
                s=30,
                zorder=3,
                label="GeoMean",
            )

        # Ensure "GeoMean" is only added to the legend once per axis
        handles, labels = ax.get_legend_handles_labels()
        if "GeoMean" not in labels:
            ax.legend(handles, labels, loc="upper right", fontsize=10)

    cumulative_time = df.groupby("Join Type")["Time"].sum().reset_index()
    max_cumulative_time = cumulative_time["Time"].max() * 1.05

    # Create extra bars with cumulative times
    for row_idx, (group, subset) in enumerate(cumulative_time.iterrows()):
        # Find the last subplot in this row (last column in the row)
        last_ax = (
            g.axes[row_idx, -1] if g.axes.ndim > 1 else g.axes[row_idx]
        )  # Handle 1D case

        # Create an extra axis next to the last facet column in the row
        extra_ax = g.fig.add_axes(
            [
                last_ax.get_position().x1 + 0.01,  # x position (right of last column)
                last_ax.get_position().y0,  # y position (same as facet row)
                0.01,  # width
                last_ax.get_position().height,
            ]
        )  # height (same as facet row)

        # Plot cumulative time as a single bar
        extra_ax.bar(["Total"], [subset["Time"]], color=(0.075, 0.004, 0.31))

        # Set y-axis limits to match the highest extra bar value
        extra_ax.set_ylim(0, max_cumulative_time)

        # Format extra plot
        extra_ax.set_xticks([])
        extra_ax.set_ylabel("")  # Remove duplicate labels
        extra_ax.set_title("cum. time", fontsize=10)

        # Get the subset of data for the current row based on the 'Join Type'
        row_data = df[df["Join Type"] == subset["Join Type"]]

        # Get the unique steps in the current row
        unique_steps = row_data["Step"].unique()

        # Create a custom legend for the row based on the steps present
        legend_handles = []
        for step in unique_steps:
            # Create a handle for each step with the corresponding color
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=step_to_color[step],
                    markersize=10,
                    label=step,
                )
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Geometric Mean",
            )
        )

        # Add the custom legend to the extra plot, positioned to the right edge of the extra bar
        extra_ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.5, 1),
            title=f"{subset['Join Type']}",
            fontsize=10,
        )

    g.savefig(plot_name + ".png", dpi=500, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 9 or (len(sys.argv) - 4) % 2 != 1:
        print(
            "Usage: python script.py <query> <time_unit> <plot_name> <clean_times_path> <dir1> <label1> [<dir2> <label2> ...]"
        )
        sys.exit(1)

    query = sys.argv[1]
    time_unit = sys.argv[2]
    plot_name = sys.argv[3]

    clean_times_path = sys.argv[4]

    # Build directories and labels list from remaining arguments
    dirs_labels = sys.argv[5:]

    # Ensure the directories and labels are paired
    dirs = dirs_labels[::2]
    labels = dirs_labels[1::2]

    time_divide = time_units[time_unit]
    prefix = "TPC-H_"

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
    }

    query_num = query
    

    # # Loop through the directories and labels
    for dir_path, label in zip(dirs, labels):
        times_dir_path = os.path.join(clean_times_path, dir_path)
        basic_info = extract_basic_info(prefix, dir_path, query_num)
        times = extract_times_for_query(prefix, times_dir_path, query_num)
        fill_data(data, basic_info, times, label, f"TPC-H {query_num}")
    plot(data, f"TPC-H {query_num}", plot_name)
