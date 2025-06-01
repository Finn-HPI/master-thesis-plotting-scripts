import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--include_a",
    type=str2bool,
    default=True,
    help="Include distribution A:  R non-unique in [0, |R|) and R = S",
)
parser.add_argument(
    "-b",
    "--include_b",
    type=str2bool,
    default=True,
    help="Include distribution B: (B) R unique in [0, |R|), S non-unique sampled from R",
)
parser.add_argument(
    "-c",
    "--include_c",
    type=str2bool,
    default=True,
    help="Include distribution C: R unique in [0, |R|), S non-unique sampled from R (according to Zipf distr.)",
)
parser.add_argument(
    "--output", required=True, help="Output file name (e.g., plot.png)."
)
args = parser.parse_args()

names = {
    "BM_JoinLinBin8": "Linear Search (8) + Binary Search",
    "BM_JoinLinBin64": "Linear Search (64) + Binary Search",
    "BM_JoinLinBin128": "Linear Search (128) + Binary Search",
    "BM_JoinBin": "Binary Search",
    "BM_JoinExp": "Exponential Search",
}

show_scale = 10
time_unit = "ms"

dist_name = {
    "key_distribution_a": "(A) R non-unique in [0, |R|) and R = S\n",
    "key_distribution_b": "(B) R unique in [0, |R|), S non-unique sampled from R\n",
    "key_distribution_c": "(C) R unique in [0, |R|), S non-unique sampled from R\n(according to Zipf distr.)",
}


def load_dataset(dir):
    data = pd.read_csv(dir + "/cmp.csv", skiprows=10)

    # Extract x from the 'name' column
    data["x"] = data["name"].apply(
        lambda x: int(x.split("/")[3])
    )  # Extract x (e.g., "1" from "algo/size_r/size_s/1/iterations")

    data["rsize"] = data["name"].apply(lambda x: int(x.split("/")[1]))
    data["ssize"] = data["name"].apply(lambda x: int(x.split("/")[2]))
    data["algo"] = data["name"].apply(lambda x: names[x.split("/")[0]])
    data["Comparisons"] = data["Comparisons"].apply(lambda x: (x / 10e6))
    data["dist"] = dist_name[dir]

    # Read time_data
    time_data = pd.read_csv(dir + "/perf.csv", skiprows=10)

    # Extract the x from the 'name' column
    time_data["x"] = time_data["name"].apply(
        lambda x: int(x.split("/")[3])
    )  # Extract x (e.g., "1" from "algo/size_r/size_s/1/iterations")
    time_data["rsize"] = time_data["name"].apply(lambda x: int(x.split("/")[1]))
    time_data["ssize"] = time_data["name"].apply(lambda x: int(x.split("/")[2]))
    time_data["algo"] = time_data["name"].apply(lambda x: x.split("/")[0])

    # Add scale column for both dataframes
    data["scale"] = data["rsize"].apply(
        lambda x: (
            1
            if x == 1048576
            else (10 if x == 10485760 else (50 if x == 52428800 else None))
        )
    )
    time_data["scale"] = time_data["rsize"].apply(
        lambda x: (
            1
            if x == 1048576
            else (10 if x == 10485760 else (50 if x == 52428800 else None))
        )
    )

    data = data.rename(columns={"cpu_time": "unused"})
    data = data.merge(
        time_data[
            [
                "name",
                "cpu_time",
                "cycles",
                "instructions",
            ]
        ],
        on="name",
        how="left",
    )
    data = data[data["scale"] == show_scale]
    return data


datasets = []
if args.include_a:
    datasets.append("key_distribution_a")
if args.include_b:
    datasets.append("key_distribution_b")
if args.include_c:
    datasets.append("key_distribution_c")
    

data = pd.concat([load_dataset(dir) for dir in datasets])
print(data.head())

g = sns.FacetGrid(
    data, col="dist", col_wrap=3, height=5, aspect=1.5, sharex=False, sharey=False
)

label_size=20
# g.set_titles(col_template="{col_name}", weight="bold", size=30)
g.set_titles(col_template="", weight="bold", size=label_size)
 
ymin = data["Comparisons"].min()
ymax = data["Comparisons"].max() + 2


def custom_plot(data, **kwargs):
    ax = plt.gca()
    ax2 = ax.twinx()
    sns.pointplot(
        data=data,
        x="x",
        y="cpu_time",
        hue="algo",
        ax=ax,
        markersize=5,
        markers=["o", "s", "D", "^", "v"],
    )
    sns.barplot(
        data=data, x="x", y="Comparisons", hue="algo", ax=ax2, alpha=0.4, legend=False
    )

    ax2.set_ylim(ymin, ymax)
    ax2.set_yticks(np.arange(0, ymax + 1, step=5))
    
    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14) 
    ax2.tick_params(axis='y', labelsize=14) 

    if ax == g.axes.flat[-1]:
        #    ax2.set_ylabel("Number of key comparisons [$10^6$] $\\mathit{{(bars)}}$", fontsize=16)
        ax2.set_ylabel("#Key comparisons [M]\n(bars)", fontsize=label_size)
        # ax2.set_ylabel('')
        # ax2.set_yticklabels([])
    else:
        ax2.set_ylabel("")
        ax2.set_yticklabels([])


g.map_dataframe(custom_plot)

custom_xlabels = {
    "(A) R non-unique in [0, |R|) and R = S\n": "Occurrences of each non-unique value",
    "(B) R unique in [0, |R|), S non-unique sampled from R\n": "Occurrences of each non-unique value",
    "(C) R unique in [0, |R|), S non-unique sampled from R\n(according to Zipf distr.)": "Skew [%]",
}

g.set_axis_labels("", f"(lines)\nJoin phase [{time_unit}]", fontsize=label_size)
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_xlabel(custom_xlabels.get(title, "x"), fontsize=label_size)

# g.add_legend()
# sns.move_legend(
#     g,
#     "lower center",
#     bbox_to_anchor=(0.5, 1),
#     ncol=5,
#     title=None,
#     frameon=True,
# )

plt.tight_layout()

g.savefig(args.output, dpi=500)
# plt.show()
