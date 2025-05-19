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


def to_latex_table(data):
    df = pd.DataFrame(data)
    # Sort df so the order is consistent per group
    df_sorted = df.sort_values(by=["Query", "Algo"])

    # Group by Query
    results = []

    for _, group in df_sorted.groupby("Query"):
        group = group.reset_index(drop=True)
        results.append(
            {
                "Query": group.loc[0, "Query"],
                group.loc[0, "Algo"]: group.loc[0, "Time"],
                group.loc[2, "Algo"]: group.loc[2, "Time"],
                group.loc[1, "Algo"]: group.loc[1, "Time"],
                f"Speedup {group.loc[2, "Algo"]} (cf. {group.loc[0, "Algo"]}) [%]": f'x{(
                    group.loc[0, "Time"] / group.loc[2, "Time"]
                ):.4f}',
                f"Speedup {group.loc[1, "Algo"]} (cf. {group.loc[0, "Algo"]}) [%]": f'x{(
                    group.loc[0, "Time"] / group.loc[1, "Time"]
                ):.4f}',
                f"Speedup {group.loc[1, "Algo"]} (cf. {group.loc[2, "Algo"]}) [%]": f'x{(
                    group.loc[2, "Time"] / group.loc[1, "Time"]
                ):.4f}',
            }
        )

    # Turn into DataFrame
    slowdown_df = pd.DataFrame(results)
    print(slowdown_df.to_latex(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
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

    to_latex_table(data)
