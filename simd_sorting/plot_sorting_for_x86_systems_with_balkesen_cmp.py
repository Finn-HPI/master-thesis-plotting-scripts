import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png).")
args = parser.parse_args()

name = {
    "avx2": r"$\bf{(a)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "cx16": r"$\bf{(b)}\ \bf{System}\ \bf{D}$" + "\n(Intel Xeon Gold 5220S)",
    "nx05": r"$\bf{(c)}\ \bf{System}\ \bf{E}$" + "\n(Intel Xeon Platinum 8352Y)"
}

# Function to read CSV files from a directory and process them
def load_data_from_directory(directory):
    # Create an empty list to store dataframes
    dataframes = []
    index = 0

    for file in sorted(os.listdir(directory)):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df["throughput_mtuples_per_sec"] = df["num_tuples"] / df["time_us"]
            df["file"] = file.replace('.csv', '')
            df["directory"] = name[os.path.basename(directory)] # Add the directory as a new column
            dataframes.append(df)
            index += 1
    
    return pd.concat(dataframes)  # Concatenate all dataframes

directories = ['avx2', 'cx16', 'nx05']

# Read and combine all data from the specified directories
final_df = pd.DataFrame()  # Initialize an empty dataframe

# Process each directory
for directory in directories:
    dir_data = load_data_from_directory(directory)
    final_df = pd.concat([final_df, dir_data])


plt.figure(figsize=(14, 6))
fontsize = 18
plt.rcParams.update({
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize
})

sns.set_theme(style="white")

palette = sns.color_palette()
tmp = palette[1]
palette[1] = palette[0]
palette[0] = palette[2]
palette[2] = tmp
palette = palette[:5]


g = sns.FacetGrid(final_df, col="directory", sharey=False, sharex=True, col_wrap=3, height=6, aspect=1.5)

g.map_dataframe(sns.pointplot, x="scale", y="throughput_mtuples_per_sec", hue="file", palette=palette, markers=['o', 's', '^', 'h', 'H'], linestyles=["-", "-", "-", "--", "--"], markersize=5)
g.set_titles(col_template="{col_name}", row_template="", size=fontsize)#, weight="bold")

for ax in g.axes.flat:
    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.set_xlabel("")  # Remove individual x-axis titles
    ax.set_ylabel("")  # Remove individual y-axis titles


plt.figtext(0.5, -0.02, "Number of tuples (in $2^{20}$)", ha="center", fontsize=20)
plt.figtext(-0.02, 0.5, "Sort throughput [M. tuples / s]", va="center", rotation="vertical", fontsize=20)

for ax in g.axes.flat:
    ax.tick_params(axis="y", which="both", left=True, labelsize=fontsize)  # Ensure y-ticks are visible
    ax.tick_params(axis="x", which="both", bottom=True, labelsize=fontsize)  # Ensure x-ticks are visible


g.add_legend()
sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=True,
    )

plt.tight_layout()
g.savefig(args.output, dpi=600)
# plt.show()
