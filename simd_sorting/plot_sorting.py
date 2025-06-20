import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="directory (e.g. arm or power)")
parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png).")
args = parser.parse_args()

dir = args.dir
files = [f'{dir}/SIMD sort [double].csv', f'{dir}/SIMD sort [int64_t].csv', f'{dir}/boost::pdqsort.csv', f'{dir}/std::sort.csv']

# Create an empty list to store dataframes
dataframes = []

# Read all CSV files and append them to the list
index = 0
for file in files:
    df = pd.read_csv(file)
    df["throughput_mtuples_per_sec"] = df["num_tuples"] / df["time_us"]
    df["file"] = file.replace('.csv', '').replace(dir + '/', '')
    df["hue"] = index
    index += 1
    dataframes.append(df)

# Concatenate all dataframes
final_df = pd.concat(dataframes)

# Plot
plt.figure(figsize=(10, 6))
plt.rcParams.update({
    "font.size": 24,  # Increase overall font size
    "axes.labelsize": 24,  # Increase axis labels
    "xtick.labelsize": 100,  # Increase x-tick labels
    "ytick.labelsize": 24,  # Increase y-tick labels
    "legend.fontsize": 24   # Increase legend font size
})
# sns.set_theme()
# sns.set_style("whitegrid")
sns.set_theme(style="white")
plt.grid(True)

palette = sns.color_palette()
palette[2] = palette[3]
palette[3] = palette[4]
palette = palette[:4]

ax = sns.pointplot(data=final_df, x="scale", y="throughput_mtuples_per_sec", hue="file", palette=palette, markers=['s', '^', 'h', 'H'], linestyles=["-", "-", "--", "--"], scale=1.5)

ax.tick_params(axis="y", which="both", left=True, labelsize=24)  # Ensure y-ticks are visible
ax.tick_params(axis="x", which="both", bottom=True, labelsize=24)  # Ensure x-ticks are visible
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
ax.grid(True, axis="y", linestyle="-", alpha=0.5, linewidth=2)
ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.5, linewidth=2)

# plt.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(final_df['file'].unique()))
plt.legend([],[], frameon=False)

plt.ylabel("Throughput [M. tuples / s]", size=26)
plt.xlabel("Number of tuples [$2^{20}$]", size=26)

plt.tight_layout()
plt.savefig(args.output, dpi=300, bbox_inches="tight")
# plt.show()

