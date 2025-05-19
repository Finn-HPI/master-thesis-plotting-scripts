import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse



# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="Directory (e.g., arm)")
parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png).")
args = parser.parse_args()

# Create an empty list to store dataframes
dataframes = []

dir = args.dir
files = [f'{dir}/|vec| = 2.csv', f'{dir}/|vec| = 4.csv', f'{dir}/|vec| = 8.csv']

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
plt.figure(figsize=(10, 7))
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

sns.pointplot(data=final_df, x="scale", y="throughput_mtuples_per_sec", hue="file", palette=palette, markers=['s', '^', 'h', 'H'], markersize=5)
plt.xlabel("Scale")
plt.ylabel("Sort throughput [M. tuples / s]")

plt.legend()
# plt.legend(title="", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(final_df['file'].unique()))
# plt.legend([],[], frameon=True)


# Adjust layout to prevent cutting off the legend
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.subplots_adjust(bottom=0.08)  # Adjust this to reduce the white space

# Setting equispaced x-ticks
min_scale = final_df["scale"].min()
max_scale = final_df["scale"].max()
num_ticks = len(final_df["scale"].unique())  # Number of unique scale values

x_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Set a linear scale for x-axis and make the ticks equispaced
plt.xticks(ticks=range(len(x_values)), labels=[str(x) for x in x_values])
plt.xlabel("Number of tuples (in $2^{20}$)")

plt.savefig(args.output, dpi=600)
# plt.show()

