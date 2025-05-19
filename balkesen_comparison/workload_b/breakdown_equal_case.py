import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

plt.rcParams.update({
    "font.size": 16,  # Increase overall font size
    "axes.labelsize": 18,  # Increase axis labels
    "xtick.labelsize": 16,  # Increase x-tick labels
    "ytick.labelsize": 16,  # Increase y-tick labels
    "legend.fontsize": 16   # Increase legend font size
})

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help="Path to the breakdown file.")
parser.add_argument("--output", type=str, default='breakdown_b.png', help="Output file name (e.g., plot.png).")
args = parser.parse_args()

df = pd.read_csv(args.file)

# Normalize by num_tuples
stages = ["partition", "sort", "merge", "join"]
df["total_cycles"] = df["partition"] + df["sort"] + df["merge"] + df["join"]
for stage in stages:
    # compute time in seconds for each stage
    df[stage] = (df[stage] / df["total_cycles"]) * (df["time_us"] / 1e6)

# Set plot style
sns.set_theme(style="white")
plt.grid(False)

# Create stacked bar plot
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
# colors = sns.color_palette("pastel", len(stages))
colors = ['#036EDB', '#039393', '#DB6E03', '#930303']
hatches = ['//','xx','\\\\','oo']
bottoms = [0] * len(df)
x = range(0, len(df) * 2, 2)

total_times = df[stages].sum(axis=1)

for i, stage in enumerate(stages):
    ax.bar(x, df[stage], label=stage, bottom=bottoms, color=colors[i], hatch=hatches[i], width=1.25, edgecolor="black", linewidth=1.5)
    bottoms += df[stage]

# Add total time labels on top of bars
for i, total in enumerate(total_times):
    ax.text(x[i], bottoms[i], f"{total:.2f}", ha='center', va='bottom', fontsize=14, fontweight='bold')
    
ax.tick_params(axis="y", which="both", left=True)  # Ensure y-ticks are visible
ax.tick_params(axis="x", which="both", bottom=True)  # Ensure x-ticks are visible

plt.ylim(top=1.075 * max(total_times))

# Labels and title
ax.set_ylabel("Join duration [s]")
ax.set_xticks(x)
ax.set_xticklabels(df["algo"], rotation=0)


ax.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

plt.tight_layout()

plt.savefig(args.output, dpi=500)
# plt.show()
