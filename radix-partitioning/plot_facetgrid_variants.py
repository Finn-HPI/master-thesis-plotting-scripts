import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Plots radix partitioning times for different optimization (software-managed buffers, streaming stores, prefetching).
# The software managed buffers use a fixed size of one cache-line.

name = {
    "power": r"$\bf{(a)}\ \bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "avx2": r"$\bf{(b)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "arm": r"$\bf{(c)}\ \bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "avx512": r"$\bf{(d)}\ \bf{System}\ \bf{D}$" + "\n(Intel Xeon Gold 5220S)"
}

def load_data(csv_files, labels, directory):
    """Loads and processes CSV data for a specific directory."""
    all_data = []
    
    for csv_file, label in zip(csv_files, labels):
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)
        
        # Ensure num_partitions is sorted and treated as an integer
        df = df.sort_values(by='num_partitions')
        df['num_partitions'] = df['num_partitions'].astype(int)

        df['total_time'] = df['time_histogram'] + df['time_init'] + df['time_partition']
        
        # Add metadata
        df['variant'] = label 
        df['platform'] = name[directory]
        
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

colors = ['#2066a8',  '#a00000','#1f6f6f','#2066a8',  "#5e4c5f"]
colors_med = ['#3594cc','#c46666', '#54a1a1','#3594cc', '#999999']
colors_light = ['#8cc5e3', '#d8a6a6', '#9fc8c8','#8cc5e3','#ffbb7f']

def plot_facetgrid(all_data, output):
    """Plots the partitioning time as line plots in a FacetGrid."""
    sns.set_style('white')

    g = sns.FacetGrid(all_data, col="platform", sharex=True, sharey=False, margin_titles=True,height=2.8, aspect=2, col_wrap=2, col_order=list(name.values()))
    g.figure.subplots_adjust(wspace=0, hspace=0)

    g.set_titles(col_template="{col_name}", row_template="")

    def stacked_barplot(data, **kwargs):
        """Helper function to plot stacked bars within each facet."""
        x = np.arange(len(data['num_partitions'].unique()))
        width = 0.25  # Width of each bar
        for i, variant in enumerate(sorted(data['variant'].unique(), key=str)):  # Ensure correct label order
            df_subset = data[data['variant'] == variant]
            bottom_init = df_subset['time_histogram'].values
            
            plt.bar(x + i * width, df_subset['time_histogram'], width, label=f'{variant} - Histogram', color=colors_light[i], edgecolor=colors_light[i],linewidth=0)
            plt.bar(x + i * width, df_subset['time_init'], width, bottom=bottom_init, label=f'{variant} - Init', color=colors_med[i], edgecolor=colors_med[i],linewidth=0)
            plt.bar(x + i * width, df_subset['time_partition'], width, bottom=bottom_init + df_subset['time_init'].values, label=f'{variant} - Partition', color=colors[i], edgecolor=colors[i],linewidth=0)

        plt.xticks(x + (len(data['variant'].unique()) - 1) * width / 2, df_subset['num_partitions'])
        plt.xlabel("Number of partitions")
        plt.ylabel("Time [ms]")
    
    g.map_dataframe(stacked_barplot)
    
    for ax in g.axes.flat:
        ax.tick_params(axis="y", which="both", left=True)  # Ensure y-ticks are visible
        ax.tick_params(axis="x", which="both", bottom=True)  # Ensure x-ticks are visible
    
    g.set_axis_labels("Number of partitions", "Time [ms]")
    
    g.add_legend()
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.375, 1), ncol=3, title=None, frameon=True,
    )

    # Save or show the plot
    if output:
        g.savefig(output, dpi=500, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot partitioning time as a FacetGrid with line plots.")
    parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png). If not provided, the plot is shown.")

    args = parser.parse_args()
    
    directories = ['avx2', 'avx512', 'arm', 'power']
    variant_labels = ['SW Buffers', 'SW Buffers + Streaming Store', 'SW Buffers + Streaming Stores (prefetched)']

    # Load and merge data from all directories
    all_data = pd.concat([load_data(
        csv_files=[
            "radix_partition_swb.csv",
            "radix_partition_swb_nt.csv",
            "radix_partition_swb_nt_pf.csv"
        ],
        labels=variant_labels,
        directory=dir_name
    ) for dir_name in directories], ignore_index=True)
    
    print(all_data)

    # Generate FacetGrid plot
    plot_facetgrid(all_data, args.output)
