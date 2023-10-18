import argparse
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv
from matplotlib.animation import FuncAnimation

def setup_graphs(num):
    sns.set(
        style="darkgrid",
        rc={
            "figure.figsize": (7.2, 4.45),
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
    colors = sns.color_palette("colorblind",  num)
    sns.set_palette(colors)
    colors = cycle(colors)
    return colors

dashes_styles = cycle(["-.", "-", "--"])

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")  # convert NaN string to NaN value

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

def update(frame, args, lines):
    ax.clear()

    for i, file in enumerate(args.f):
        df_ep = pd.DataFrame()

        for conn in range(1, args.conn + 1):
            episode_data = []
            ep = args.start - 1
            try:
                for episode_num in range(args.start, frame + 1):
                    ep += 1
                    f = file + str(conn) + f"_ep{episode_num}.csv"
                    df = pd.read_csv(f, sep=args.sep)[["step",  y_axis_variable]]
                    episode_sum = df[y_axis_variable].mean()
                    episode_data.append({"Episode": episode_num, y_axis_variable: episode_sum})

                if df_ep.empty:
                    df_ep = pd.DataFrame.from_records(episode_data)
                else:
                    df_ep = pd.concat((df_ep, pd.DataFrame.from_records(episode_data)))

            except Exception as e:
                pass

        if i >= len(lines):
            lines.append(ax.plot([], [], label=next(labels), color=next(colors), linestyle=next(dashes_styles))[0])

        # Plot DataFrame
        plot_df(df_ep, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)
        lines[i].set_data(df_ep[args.xaxis], df_ep[args.yaxis])

    ax.set_title(args.t)
    ax.set_ylabel(args.ylabel)
    ax.set_xlabel("Episodes")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

if __name__ == "__main__":
    y_axis_variable = "system_total_waiting_time"
    y_name = "system_total_waiting_time (s)"

    # y_variables = ["system_total_waiting_time", "system_accumulated_waiting_time (100)", "system_total_stopped", "system_mean_waiting_time", "system_accumulated_mean_waiting_time (100)", "system_mean_speed", "system_cars_present", "agents_total_accumulated_waiting_time (100)", "agents_total_stopped", "agents_mean_waiting_time (100)", "agents_mean_speed", "agents_cars_present"]
    # y_names = ["system_total_waiting_time (s)", "system_accumulated_waiting_time (100) (s)", "system_total_stopped (stopped vehicles)", "system_mean_waiting_time (s)", "system_accumulated_mean_waiting_time (100) (s)", "system_mean_speed (m/s)", "system_cars_present", "agents_total_accumulated_waiting_time (100) (s)", "agents_total_stopped (stopped vehicles)", "agents_mean_waiting_time (100) (s)", "agents_mean_speed (m/s)", "agents_cars_present"]
    
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
    )
    prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
    prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
    prs.add_argument("-t", type=str, default=y_axis_variable, help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default=y_axis_variable, help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default="Episode", help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument("-conn", type=int, default=1, help="Number of conns.\n")
    prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
    prs.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
    prs.add_argument("-ylabel", type=str, default=y_name, help="Y axis label.\n")
    prs.add_argument("-output", type=str, default=None, help="PDF output filename.\n")
    prs.add_argument("-start", type=int, default=1, help="Start episode.\n")
    prs.add_argument("-stop", type=int, default=10, help="Stop at episode.\n")

    args = prs.parse_args()
    filenames = args.f
    print(filenames)
    colors = setup_graphs(len(args.f) if len(args.f) > 1 else 25)

    if args.l == None:
      labels = cycle([s.split("/")[-1] for s in args.f]) if args.f is not None else cycle([str(i) for i in range(len(args.f))])
    else:
      labels = cycle(args.l)

    fig, ax = plt.subplots()
    lines = []  # Store lines for each model

    ani = FuncAnimation(fig, update, frames=range(args.start, args.stop + 1), fargs=(args, lines), repeat=False, interval=100)
    ani.save("live_plot.gif", writer="pillow")

    if args.output is not None:
        with PdfPages(args.output) as pdf:
            pdf.savefig(fig)
    else:
        plt.show()