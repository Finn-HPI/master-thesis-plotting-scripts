import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import statistics
import argparse
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from statistics import geometric_mean
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str, help="The system")
    args = parser.parse_args()

    prefix = "TPC-H_"

    queries = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22]
    scale_factors = ["10", "50", "100"]

    data = {"Query": [], "Scale Factor": [], "Algo": [], "Time": [], "Time Unit": []}

    time_unit = "s"
    time_divide = time_units[time_unit]

    system_name = {
        "arm": "System C",
        "avx2": "System B",
        "avx512": "System E",
        "power": "System A",
    }

    system_name_lower_case = {
        "arm": "system_c",
        "avx2": "system_b",
        "avx512": "system_e",
        "power": "system_a",
    }

    system = args.system

    for sf in scale_factors:
        for query in queries:
            query_name = f"{query:02d}"
            time_hj = (
                extract_times_for_query(system + "/hj_sf" + sf, query_name)
                / time_divide
            )
            time_ssmj = (
                extract_times_for_query(system + "/ssmj_sf" + sf, query_name)
                / time_divide
            )
            time_ssmj_no_bf = (
                extract_times_for_query(system + "/ssmj_sf" + sf + "_nb", query_name)
                / time_divide
            )
            for label, time in [
                ("HJ", time_hj),
                ("SSMJ", time_ssmj),
                ("SSMJ w/o BF", time_ssmj_no_bf),
            ]:
                data["Query"].append(query_name)
                data["Scale Factor"].append(sf)
                data["Algo"].append(label)
                data["Time"].append(time * 1000 if sf == "10" else time)
                data["Time Unit"].append("ms" if sf == "10" else "s")

    df = pd.DataFrame(data)

    df["Scale Factor"] = df["Scale Factor"].astype(int)

    # Pivot the table
    pivoted = df.pivot_table(
        index=["Scale Factor", "Algo", "Time Unit"], columns="Query", values="Time"
    ).reset_index()

    # Rename columns for LaTeX
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns=lambda x: f"Q{x}" if isinstance(x, int) else x)
    pivoted = pivoted.rename(columns={"Scale Factor": "SF"})

    # Optional: sort
    pivoted = pivoted.sort_values(by=["SF", "Algo"])

    # Export to LaTeX
    latex_table = pivoted.to_latex(
        index=False,
        float_format="%.2f",
        column_format="ll" + "r" * (pivoted.shape[1] - 2),
    )
    latex_table = latex_table.replace("\\toprule", "\\hline")
    latex_table = latex_table.replace("\\midrule", "\\hline")
    latex_table = latex_table.replace("\\bottomrule", "\\hline")
    latex_table = latex_table.replace("50 & HJ", "\\hline 50 & HJ")
    latex_table = latex_table.replace("100 & HJ", "\\hline 100 & HJ")

    lines = latex_table.splitlines()[3:]

    latex_table = r"""
\begin{table}[h!]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{l|l|c|c|c|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r}
\hline
\multirow{2}{*}{SF} & \multirow{2}{*}{Join Algorithm} & Time & \multicolumn{19}{c}{TPC-H Query}\\\cline{4-22}
&  & Unit & 02 & 03 & 04 & 05 & 07 & 08 & 09 & 10 & 11 & 12 & 13 & 14 & 16 & 17 & 18 & 19 & 20 & 21 & 22 \\
"""

    latex_table += "\n".join(lines)

    #  Wrap in resizebox
    latex_table = latex_table.replace("\\end{tabular}", "\\end{tabular}}")
    latex_table += (
        f"\n\\caption{{TPC-H join times on \\systemA{{{system_name[system]}}}}}"
    )
    latex_table += f"\n\\label{{tab:tpch_results_{system_name_lower_case[system]}}}"
    latex_table += "\n\\end{table}\n"

    print(latex_table)
