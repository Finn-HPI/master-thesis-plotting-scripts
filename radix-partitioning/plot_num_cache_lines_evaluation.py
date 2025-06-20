import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math


name = {
    "power": r"$\bf{(a)}\ \bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "avx2": r"$\bf{(b)}\ \bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "arm": r"$\bf{(c)}\ \bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "avx512": r"$\bf{(d)}\ \bf{System}\ \bf{D}$" + "\n(Intel Xeon Gold 5220S)"
}


superscript_map = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '10': '¹⁰', '11': '¹¹', '12': '¹²', '13': '¹³', '14': '¹⁴'
}

def convert_to_superscript(num):
    exponent = int(math.log2(num))
    if exponent <= 14:
        superscript_exponent = superscript_map[str(exponent)]
        return f"2{superscript_exponent}"
    else:
        return f"2^{exponent}"  # Fallback if exponent exceeds 14

def load_data(csv_files, labels, directory):
    """Loads and processes CSV data for a specific directory."""
    all_data = []
    
    for csv_file, label in zip(csv_files, labels):
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)
        
        # Ensure num_partitions is sorted and treated as an integer
        df = df.sort_values(by='num_partitions')
        df['num_partitions'] = df['num_partitions'].apply(convert_to_superscript)

        # Compute total partitioning time
        df['total_time'] = df['time_histogram'] + df['time_init'] + df['time_partition']
        
        df['num_cache_lines'] = label  # Cache line count (1-6)
        df['platform'] = name[directory]  # Architecture (AVX2, AVX512, ARM, POWER)
        
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def plot_facetgrid(all_data, output):
    sns.set_style('white')

    g = sns.FacetGrid(all_data, col="platform", sharex=True, sharey=False, margin_titles=True,height=2.5, aspect=2, col_wrap=2, col_order=list(name.values()))

    palette = sns.color_palette('tab10', n_colors=all_data["num_partitions"].nunique())

    g.map_dataframe(sns.lineplot, x="num_cache_lines", y="total_time", hue="num_partitions", 
                    style="num_partitions", markers=True, palette=palette)
    
    g.set_titles(col_template="{col_name}", row_template="", size=12)#, weight="bold")
    g.set_axis_labels("Number of cache-lines per partition buffer", "Total Partitioning Time [ms]")
    
    for ax in g.axes.flat:
        ax.tick_params(axis="y", which="both", left=True)  # Ensure y-ticks are visible
        ax.tick_params(axis="x", which="both", bottom=True)  # Ensure x-ticks are visible
    
    # Function to plot vertical lines and marker at the fastest partitioning time
    def plot_min_time_vline(data, **kwargs):
        ax = plt.gca()
        unique_partitions = sorted(data["num_partitions"].unique())  # Ensure consistent color assignment

        # Extract the seaborn legend elements for style and markers
        lines = ax.get_lines()
        line_styles = {line.get_label(): line.get_linestyle() for line in lines}
        markers = {line.get_label(): line.get_marker() for line in lines}
        
        lines = {}

        for i, partition in enumerate(unique_partitions):
            subset = data[data["num_partitions"] == partition]
            min_row = subset.loc[subset["total_time"].idxmin()]
            x_pos = min_row["num_cache_lines"]
            color = palette[i]  # Get the matching color
            marker = markers.get(str(partition), "o")  # Default marker if unknown

            # Plot vertical line using the extracted style and color
            if x_pos not in lines:
                ax.axvline(x_pos, color='gray', alpha=0.8, linewidth=1, linestyle='--')
                lines[x_pos] = True
            
            # Mark the point with the same marker as the lineplot
            ax.scatter(x_pos, min_row["total_time"], color=color, marker=marker, s=100, edgecolor="black")

    g.map_dataframe(plot_min_time_vline)
    
    font_size = 16
    for ax in g.axes.flat:
        ax.set_xlabel("")  # Remove individual y-axis titles
        ax.set_ylabel("")  # Remove individual y-axis titles
        font_size = ax.title.get_fontsize()

    # Manually set global axis labels
    plt.figtext(0.5, 0, "Number of cache-lines per buffer", ha="center", fontsize=font_size+4)
    plt.figtext(0, 0.5, "Total Partitioning Time [ms]", va="center", rotation="vertical", fontsize=font_size+4)
    
    g.add_legend()
    g._legend.set_title("Fan-out", prop={'size': font_size+2})
    for text in g._legend.get_texts():
        text.set_fontsize(font_size+2)

    # Save or show the plot
    if output:
        g.savefig(output, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# Example Usage: python3 plot_num_cache_lines_evaluation.py --output plot.png
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output file name (e.g., plot.png). If not provided, the plot is shown.")

    args = parser.parse_args()
    
    directories = ['avx2', 'avx512', 'arm', 'power']
    num_cache_line_labels = [1,2,3,4,5,6]

    # Load and merge data
    all_data = pd.concat([load_data(
        csv_files=[
            "radix_partition_1.csv",
            "radix_partition_2.csv",
            "radix_partition_3.csv",
            "radix_partition_4.csv",
            "radix_partition_5.csv",
            "radix_partition_6.csv"
        ],
        labels=num_cache_line_labels,
        directory=dir_name
    ) for dir_name in directories], ignore_index=True)

    plot_facetgrid(all_data, args.output)
