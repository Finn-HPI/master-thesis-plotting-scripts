import os
import argparse
import pandas as pd
from scipy.stats import gmean
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt

name = {
    'JoinHash': 'HJ',
    'JoinSortMerge': 'SMJ',
    'JoinSimdSortMerge': 'SSMJ'
}

system_name = {
    "cp03": r"$\bf{System}\ \bf{A}$" + "\n(IBM Power10)",
    "cx30": r"$\bf{System}\ \bf{B}$" + "\n(AMD EPYC 7742)",
    "ga02": r"$\bf{System}\ \bf{C}$" + "\n(ARM Neoverse-V2)",
    "nx05": r"$\bf{System}\ \bf{E}$" + "\n(Intel Xeon Platinum 83252Y)"
}

def read_benchmark_file(filepath):
    with open(filepath, "r") as f:
        # Skip until we find the header
        while True:
            pos = f.tell()
            line = f.readline()
            if line.startswith("name,"):
                f.seek(pos)
                break

        # Read the remaining lines into a DataFrame
        df = pd.read_csv(f)
        df = df[df['name'].str.contains('_mean', na=False)]
        df['name'] = df['name'].str.replace('_mean', '', regex=False)
        df['algo'] = df['name'].str.split('/').str[0].str.extract(r'<(.*?)>')[0].map(name)

        # Define conditions for special cases
        is_small = df['name'].str.startswith('BM_Join_Small')
        is_medium = df['name'].str.startswith('BM_Join_Medium')

        # Compute 'R'
        df['R'] = np.select(
            [is_small, is_medium],
            [1/1024, 100/1024],
            default=df['name'].str.split('/').str[1].astype(float)
        )

        # Compute 'S'
        df['S'] = np.select(
            [is_small, is_medium],
            [
                df['R'] * df['name'].str.split('/').str[1].astype(float),
                df['R'] * df['name'].str.split('/').str[1].astype(float)
            ],
            default=df['R'] * df['name'].str.split('/').str[2].astype(float)
        )
        df['S_scale'] = np.select(
            [is_small, is_medium],
            [
                df['name'].str.split('/').str[1].astype(float),
                df['name'].str.split('/').str[1].astype(float)
            ],
            default=df['name'].str.split('/').str[2].astype(float)
        )
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--smj', action='store_true', help='Enable SMJ (default: disabled)')
parser.add_argument('--hj', action='store_true', help='Enable HJ (default: disabled)')
parser.add_argument("--output", required=True, help="")
args = parser.parse_args()

print(f"SMJ enabled: {args.smj}")
print(f"HJ enabled: {args.hj}")

file = 'join_complete2.csv'

all_dfs = []
directories = ['cp03', 'cx30', 'ga02' ,'nx05']
for dir in directories:
    df = read_benchmark_file(dir + '/' + file)
    df['system'] = system_name[dir]
    all_dfs.append(df)
    
combined_df = pd.concat(all_dfs, ignore_index=True)

if args.hj and args.smj:
    palette=['#d8a6a6', '#a8dadc']
elif args.hj:
    palette=['#d8a6a6']
elif args.smj:
    palette=['#a8dadc']

def compute_speedup(data, **kwargs):
    data = data.copy()
    # Extract SSMJ baseline times per system
    baseline = data[data['algo'] == 'SSMJ'][['system', 'cpu_time']].rename(columns={'cpu_time': 'base_time'})

    # Merge baseline back into the full DataFrame on the 'system' column
    data = data.merge(baseline, on='system', how='left')

    # Compute speedup as percent
    data['speedup_pct'] = (data['base_time'] - data['cpu_time']) / data['base_time'] * 100
    

    # Only keep the algorithms we want to show
    algos = []
    if args.hj:
        algos.append('HJ')
    if args.smj:
        algos.append('SMJ')
    plot_data = data[data['algo'].isin(algos)]

    # Draw the barplot centered around 0
    # ax = sns.barplot(estimator='mean', errorbar=lambda x: (x.min(), x.max()), data=plot_data, x='algo', y='speedup_pct', palette=['#e63946', '#a8dadc'])
    sns.violinplot(data=plot_data, x='algo', y='speedup_pct', inner=None, density_norm='width', hue='algo', palette=palette, legend=False)
    # sns.boxplot(data=plot_data, x='algo', y='speedup_pct', palette=['#e63946', '#a8dadc'])
    sns.stripplot(data=plot_data, x='algo', y='speedup_pct', jitter=True, hue='system', size=5)

    # sns.swarmplot(data=plot_data, x='algo', y='speedup_pct', hue='system', dodge=True)
    # y_min, y_max = ax.get_ylim()
    # y_range = y_max - y_min
    # padding = 0.125 * y_range  # space above the bar for label outside
    # offset = 20

    # for container in ax.containers:
    #     for rect in container:
    #         height = rect.get_height()
    #         x = rect.get_x() + rect.get_width() / 2
    #         label = f"{int(round(height))}%"

    #         if height >= 0:
    #             # Positive bar
    #             if height > 2 * padding:
    #                 # Label inside the bar
    #                 ax.text(x, height / 2, label, ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    #             else:
    #                 # Label just above the bar
    #                 ax.text(x, height + offset, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    #         else:
    #             # Negative bar
    #             if abs(height) > 2 * padding:
    #                 # Label inside the bar
    #                 ax.text(x, height / 2, label, ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    #             else:
    #                 # Label just below the bar
    #                 ax.text(x, height - offset, label, ha='center', va='top', fontsize=10, fontweight='bold')

    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Zero baseline

# Usage with seaborn FacetGrid
g = sns.FacetGrid(combined_df, row="R", col="S_scale", sharex=True, sharey=False, height=1.75, aspect=1.5)
g.map_dataframe(compute_speedup)
g.set_titles("{row_name}|{col_name}")

for i, ax in enumerate(g.axes.flatten()):
    
    title = ax.get_title()
    parts = title.split('|')
    R = parts[0].strip()
    scale = parts[1].strip()
    S = float(R) * float(scale)
    
    def format_unit(value):
        value = float(value)
        if value < 1:
            value *= 1024
            unit = "K"
        else:
            unit = "M"
        return f'{value:.0f} {unit}' if value.is_integer() else f'{value:.2f} {unit}'

    R = format_unit(R)
    S = format_unit(S)
        
    new_title = f"{R} ⋈ {S}"
    ax.set_title(new_title)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    # ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", which="both", bottom=True, top=False)
    ax.tick_params(axis="y", which="both", left=True, top=False, width=1.5)

    major_ticks = ax.get_yticks()
    # Compute minor ticks (midpoints between major ticks)
    minor_ticks = (major_ticks[:-1] + major_ticks[1:]) / 2

    # # Add minor ticks
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks, minor=False)
    # ymin, ymax = ax.get_ylim()
    # # ax.set_ylim(bottom=ymin, top=ymax)

    # # Customize minor tick appearance (shorter lines)
    ax.tick_params(axis="y", which="minor", length=3, width=1)
    ax.grid(True, axis="y", linestyle="-", alpha=0.5, linewidth=1)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.5, linewidth=1)


g.set_axis_labels("", "")
plt.figtext(0, 0.5, "Speedup [%]", va="center", rotation="vertical")

g.add_legend()
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.45, 1), ncol=4, title=None, frameon=True,
)

g.tight_layout()
# plt.show()
g.savefig(args.output, dpi=500)