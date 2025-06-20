import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,  # Increase overall font size
    "axes.labelsize": 18,  # Increase axis labels
    "xtick.labelsize": 16,  # Increase x-tick labels
    "ytick.labelsize": 16,  # Increase y-tick labels
    "legend.fontsize": 16   # Increase legend font size
})

def load_csv(filepath):
    """Load CSV file and convert time_us to seconds."""
    df = pd.read_csv(filepath)
    df["time_s"] = df["time_us"] / 1_000_000  # Convert microseconds to seconds
    return df

def plot_comparison(dir, output):
    """Plot execution time comparison from four CSV files."""
    # Load the data
    df1 = load_csv(f'{dir}/balkesen_no_numa.csv')
    df2 = load_csv(f'{dir}/balkesen_numa.csv')
    df3 = load_csv(f'{dir}/ssmj_double.csv')
    df4 = load_csv(f'{dir}/ssmj_int64_t.csv')

    # Add a column to identify the source of data
    df1["Algo"] = 'm-way w/o NUMA'
    df2["Algo"] = 'm-way with NUMA'
    df3["Algo"] = 'SSMJ [double]'
    df4["Algo"] = 'SSMJ [int64_t]'

    # Combine all dataframes
    combined_df = pd.concat([df1, df2, df3, df4])
    
    sns.set_theme(style="white")

    # Plot using seaborn
    plt.figure(figsize=(11, 6))
    # , linestyles=["-", "--", ":", "-."]
    ax = sns.pointplot(data=combined_df, x="scale", y="time_s", hue="Algo", markers=["o", "s", "D", "^"], alpha=0.9, scale=1.35)
   
    labelsize = 30
   
    ax.tick_params(axis='x', labelsize=24) 
    ax.tick_params(axis='y', labelsize=24) 
    
    ax.tick_params(axis="x", which="both", bottom=True, top=False)
    ax.tick_params(axis="y", which="both", left=True, top=False, width=1.5)

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
    ax.grid(True, axis="y", linestyle="-", alpha=0.5, linewidth=1.5)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.5, linewidth=1.5)
    
    # Labels and title
    plt.xlabel("m $\\cdot$ |R| = |S| [1600 * 1e6]", fontsize=labelsize)
    plt.ylabel("Join duration [s]", fontsize=labelsize)

    plt.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    # plt.legend([],[], frameon=False)
    plt.tight_layout()
    
    # Show plot
    plt.savefig(output, dpi=500)
    # plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="directory (cx30_avx2 or nx05_avx512)")
    parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png).")
    args = parser.parse_args()
    
    plot_comparison(args.dir, args.output)

if __name__ == "__main__":
    main()
