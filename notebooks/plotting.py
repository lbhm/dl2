import matplotlib as mpl

WONG_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

SNS_COLORS = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#949494",
    "#ECE133",
    "#56B4E9",
]

COLOR_DICT = {
    "raw": SNS_COLORS[0],
    "raw50": SNS_COLORS[0],
    "raw40": SNS_COLORS[0],
    "raw30": SNS_COLORS[0],
    "raw20": SNS_COLORS[0],
    "raw10": SNS_COLORS[0],
    "jpeg85": SNS_COLORS[1],
    "jpeg75": SNS_COLORS[2],
    "jpeg50": SNS_COLORS[3],
    "jpeg25": SNS_COLORS[4],
    "jpeg10": SNS_COLORS[5],
    "jpeg05": SNS_COLORS[6],
    "jpeg01": SNS_COLORS[7],
    "webp85": SNS_COLORS[1],
    "webp75": SNS_COLORS[2],
    "webp50": SNS_COLORS[3],
    "webp25": SNS_COLORS[4],
    "webp10": SNS_COLORS[5],
    "webp05": SNS_COLORS[6],
    "webp01": SNS_COLORS[7],
    
    "synthetic": SNS_COLORS[0],
    "pytorch": SNS_COLORS[1],
    "dali": SNS_COLORS[2],
    
    "r50": SNS_COLORS[0],
    "ResNet50": SNS_COLORS[0],
    "ResNet18": SNS_COLORS[1],
    "AlexNet": SNS_COLORS[2],
    
    "rx": SNS_COLORS[0],
    "tx": SNS_COLORS[1],
    
    "default": SNS_COLORS[0],
    "minio": SNS_COLORS[1],
    "compressed": SNS_COLORS[2],
    "combined": SNS_COLORS[3],
}


def latexify(base_size=11):
    """
    Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    base_size : float, optional, font size
    """

    params = {
        "axes.labelsize": base_size, # fontsize for x and y labels
        "axes.labelpad": 1,
        "axes.titlesize": base_size,
        "axes.titleweight": "bold",
        "backend": "ps",
        "font.family": "serif",
        "font.size": base_size,
        "legend.borderpad": 0.4,  # border whitespace, default 0.4
        "legend.borderaxespad": 0.4,  # the border between the axes and legend edge", default 0.5
        "legend.columnspacing": 0.9,  # column separation, default 2.0
        "legend.edgecolor": "black",
        "legend.fontsize": base_size * 0.9,
        "legend.framealpha": 0.5,
        "legend.handleheight": 0.7,  # the height of the legend handle, default 0.7
        "legend.handlelength": 1.25,  # the length of the legend lines, default 2.0
        "legend.handletextpad": 0.7,  # the space between the legend line and legend text, default 0.8
        "legend.labelspacing": 0.2,  # the vertical space between the legend entries, default 0.5
        "legend.title_fontsize": base_size * 0.9,
        "lines.linewidth" : 1.0,
        "ps.usedistiller": "xpdf",
        "text.usetex": True,
        "xtick.labelsize": base_size * 0.9,
        "ytick.labelsize": base_size * 0.9,
    }
    
    mpl.rcParams.update(params)


def autolabel_bars(ax, bars, offsets=None, precision=1):
    """
    Attach a text label above each bar in *bars*, displaying its height.
    """
    for i, bar in enumerate(bars):
        if offsets:
            height = bar.get_height() + offsets[i]
        else:
            height = bar.get_height()
        ax.annotate(f"{height:.{precision}f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    fontsize=mpl.rcParams["font.size"] * 0.75,
                    xytext=(0, 1),  # 1 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")
