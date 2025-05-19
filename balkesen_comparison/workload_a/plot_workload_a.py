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
    plt.figure(figsize=(10, 8))
    ax = sns.pointplot(data=combined_df, x="scale", y="time_s", hue="Algo", markers=["o", "s", "D", "^"], linestyles=["-", "--", ":", "-."])
   
    ax.tick_params(axis="y", which="both", left=True)  # Ensure y-ticks are visible
    ax.tick_params(axis="x", which="both", bottom=True)  # Ensure x-ticks are visible
    ax.grid(axis='y', which='major', linewidth=0.5)
    
    # Labels and title
    plt.xlabel("m * |R| = |S| [1600 * 1e6]")
    plt.ylabel("Join duration [s]")

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
