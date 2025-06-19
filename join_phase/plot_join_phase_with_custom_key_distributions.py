import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

time_unit = "ms"

dist_name = {
    "key_distribution_a": "(I)",
    "key_distribution_b": "(II)",
    "key_distribution_c": "(III)",
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
    data["Comparisons"] = data["Comparisons"].apply(lambda x: (x))
    data["dist"] = dist_name[dir]
    data = data[data["rsize"] == 10485760]

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
    data = data[data["scale"] == 10]
    return data


datasets = []
if args.include_a:
    datasets.append("key_distribution_a")
if args.include_b:
    datasets.append("key_distribution_b")
if args.include_c:
    datasets.append("key_distribution_c")


data = pd.concat([load_dataset(dir) for dir in datasets])
# sns.set_theme(style="white")
g = sns.FacetGrid(
    data, col="dist", col_wrap=2, height=6, aspect=2, sharex=False, sharey=False
)

label_size = 30
g.set_titles(col_template="{col_name}", weight="semibold", size=35)
# g.set_titles(col_template="", weight="bold", size=label_size)

ymin = data["Comparisons"].min()
ymax = data["Comparisons"].max() + 2



def custom_plot(data, **kwargs):
    ax = plt.gca()
    
    sns.pointplot(
        data=data,
        x="x",
        y="cpu_time",
        hue="algo",
        ax=ax,
        # markersize=5,
        scale=1.5,
        markers=["o", "s", "D", "^", "v"],
    )
    
    ax2 = ax.twinx()
    sns.barplot(
        data=data, x="x", y="Comparisons", hue="algo", ax=ax2, alpha=0.4, legend=False
    )
    # ticks = [1e6, 1e7, 1e8]
    # ax2.set_yticks(ticks)
    ax2.set_yscale('log')
    # ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None, numticks=10))
    # ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
    # ax2.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels if you want
    # ax2.get_yaxis().set_major_formatter(ticker.LogFormatter(base=10))

    ax2.set_ylim(ymin, ymax * 1.25)
    # ax2.set_yticks(np.arange(0, ymax + 1, step=5))
    ax.set_zorder(ax2.get_zorder() + 1)

    # Make ax background transparent so barplot shows through
    ax.patch.set_visible(False)


    ticklablesize= 22
    ax.tick_params(axis="x", labelsize=ticklablesize+4)
    ax.tick_params(axis="y", labelsize=ticklablesize+2)
    ax2.tick_params(axis="y", labelsize=ticklablesize+2)

    # if ax == g.axes.flat[-1]:
        #    ax2.set_ylabel("Number of key comparisons [$10^6$] $\\mathit{{(bars)}}$", fontsize=16)
    ax2.set_ylabel("#Key comparisons (bars)", fontsize=label_size)
        # ax2.set_ylabel('')
        # ax2.set_yticklabels([])
    # else:
    #     ax2.set_ylabel("")
    #     ax2.set_yticklabels([])


g.map_dataframe(custom_plot)
g.figure.subplots_adjust(wspace=0, hspace=0)

custom_xlabels = {
    "(I)": "Paramter $x$",
    "(II)": "Paramter $x$",
    "(III)": "Skew [%]",
}

g.set_axis_labels("", f"Join phase [{time_unit}] (lines)", fontsize=label_size)
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_xlabel(custom_xlabels.get(title, "x"), fontsize=label_size)

g.add_legend()
sns.move_legend(
    g,
    "upper left",
    bbox_to_anchor=(0.6, 0.45),
    ncol=1,
    title=None,
    frameon=False,
    fontsize=30,
)

plt.tight_layout()

g.savefig(args.output, dpi=300)
# plt.show()
