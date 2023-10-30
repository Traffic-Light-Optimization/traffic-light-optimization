import argparse
import glob
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from matplotlib.animation import FuncAnimation

# Initialize an empty dictionary to store the sums
meanVec = {}

def setup_graphs(num):
    sns.set(
        style="darkgrid",
        rc={
            "figure.figsize": (7, 4.5),
            "text.usetex": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 15,
            "figure.autolayout": True,
            "axes.titlesize": 17,
            "axes.labelsize": 12,
            "lines.linewidth": 0.8,
            "lines.markersize": 6,
            "legend.fontsize": 8,
        },
    )
    colors = sns.color_palette("colorblind", num)
    sns.set_palette(colors)
    colors = cycle(colors)
    return colors

dashes_styles = cycle(["-"])

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")

def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")
    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)
    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x[:-ma], mean[:-ma], label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

def update(frame, args, line):
    plt.clf()  # Clear the current plot
    for i, file in enumerate(args.f):
        main_df = pd.DataFrame()
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f, sep=args.sep)[["step", y_axis_variable]].head(3*frame)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))
        plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)
    plt.title(args.t)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left')

if __name__ == "__main__":
    y_axis_variable =  "agents_mean_waiting_time (100)"
    y_name =  "agents_mean_waiting_time (100) (s)"

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
    )
    prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
    prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
    prs.add_argument("-t", type=str, default=y_axis_variable, help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default=y_axis_variable, help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
    prs.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
    prs.add_argument("-ylabel", type=str, default=y_name, help="Y axis label.\n")
    prs.add_argument("-output", type=str, default=None, help="PDF output filename.\n")
    prs.add_argument("-dur", type=int, default=3000, help="duration of gif in ms.\n")
    prs.add_argument("-steps", type=int, default=450, help="total number of steps.\n")

    

    args = prs.parse_args()
    if args.l is None:
        labels = cycle([s.split("/")[-1] for s in args.f]) if args.f is not None else cycle([str(i) for i in range(len(args.f))])
    else:
        labels = cycle(args.l)

    colors = setup_graphs(len(args.f) if len(args.f) > 1 else 25)
    line=[]

    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=range(math.ceil(args.ma/3),int(args.steps/3)), fargs=(args,line), repeat=False, interval=int(3*args.dur/(args.steps-args.ma)))
    ani.save("live_sim_plot.gif", writer="pillow")
    plt.show()
