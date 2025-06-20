import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png).")

args = parser.parse_args()

# Create an empty list to store dataframes
dataframes = []

directories = ['avx2', 'avx512', 'arm', 'power']

name = {
    "power": r"$\bf{(a)}\ \bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "avx2": r"$\bf{(b)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "arm": r"$\bf{(c)}\ \bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "avx512": r"$\bf{(d)}\ \bf{System}\ \bf{D}$" + "\n(Intel Xeon Gold 5220S)"
}

# Read all CSV files from given directories
for directory in directories:
    for file in sorted(os.listdir(directory)):
        if file.endswith(".csv"):
            filepath = os.path.join(directory, file)
            df = pd.read_csv(filepath)
            df["throughput_mtuples_per_sec"] = df["num_tuples"] / df["time_us"]
            df["file"] = file.replace('.csv', '')
            df["directory"] = name[os.path.basename(directory)]
            dataframes.append(df)

# Concatenate all dataframes
final_df = pd.concat(dataframes)

# Set plotting styles
plt.figure(figsize=(14, 6))
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

sns.set_theme(style="white")
plt.grid(True)

palette = sns.color_palette()
palette[2] = palette[3]
palette[3] = palette[4]
palette = palette[:4] # We only need four colors

# Create a FacetGrid for separate plots per directory
g = sns.FacetGrid(final_df, col="directory", sharey=False, col_wrap=2, height=5, aspect=2, col_order=list(name.values()))
g.map_dataframe(sns.pointplot, x="scale", y="throughput_mtuples_per_sec", hue="file", palette=palette, markers=['s', '^', 'h', 'H'], scale=1.5)
g.set_titles(col_template="{col_name}", row_template="", size=20)#, weight="bold")
# Remove axis titles (set them globally instead)
for ax in g.axes.flat:
    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.set_xlabel("")  # Remove individual x-axis titles
    ax.set_ylabel("")  # Remove individual y-axis titles

# Manually set global axis labels
plt.figtext(0.5, -0.02, "Number of tuples [$2^{20}$]", ha="center", fontsize=30)
plt.figtext(-0.02, 0.5, "Throughput [M. tuples / s]", va="center", rotation="vertical", fontsize=30)

# Add y-ticks explicitly, making sure all plots have visible y-axis ticks
for ax in g.axes.flat:
    ax.tick_params(axis="y", which="both", left=True, labelsize=22)  # Ensure y-ticks are visible
    ax.tick_params(axis="x", which="both", bottom=True, labelsize=22)  # Ensure x-ticks are visible

g.add_legend()
sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=True,
    )

plt.tight_layout()

g.savefig(args.output, dpi=300)
