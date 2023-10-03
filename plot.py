import argparse
import glob
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Initialize an empty dictionary to store the sums
meanVec = {}

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

def compare(compVec, pdf_pages):
  grouped_data = {}

  for key, value in compVec.items():
        # Extract the y-axis name from the key
        y_axis_name = key.split(",")[1]

        if y_axis_name not in grouped_data:
            grouped_data[y_axis_name] = []

        # Append the key-value pair to the corresponding group
        grouped_data[y_axis_name].append((key, value))

  # Sort each group by the sum (value) in ascending order
  for y_axis_name, group in grouped_data.items():
        sorted_group = sorted(group, key=lambda x: x[1])

        print(f"Group for Y-Axis: {y_axis_name}")
        for key, value in sorted_group:
            print(f"{key}: {value}")
        print("\n")

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
    
    # Calculate and store the sum of the second column
    sum_yaxis = df[yaxis].sum()
    count_yaxis = df[yaxis].count()
    # Calculate the average (avg_yaxis) by dividing sum_yaxis by count_yaxis
    avg_yaxis = sum_yaxis / count_yaxis
    # Create a label for the row entry in meanVec
    row_label = f"{label},{yaxis}"
    # Add the sum to the meanVec dictionary with the label as the key
    meanVec[row_label] = sum_yaxis # or avg_yaxis 

def getPDFName(filenames):
  pdf_name = ""
  if len(filenames) > 1:
    
    parts = filenames[0].split("/")[-1].split(".")[0].split("-", 1)
    pdf_name = parts[0]
    last = filenames[0].split("/")[-1].split(".")[0].split("-", 1)[1].split("_")[1]
    groups = ""
    for filename in filenames:
      group = filename.split("/")[-1].split(".")[0].split("-",1)[1].split("_")[0]
      groups = groups + f"-({group})"

    return pdf_name + groups + "_" + last

  else:
    pdf_name = filenames[0].split("/")[-1].split(".")[0]
  return pdf_name

if __name__ == "__main__":

  # List of five different y-axis variables
  y_variables = ["system_total_waiting_time", "system_accumulated_waiting_time (100)", "system_accumulated_waiting_time (delta)", "system_total_stopped", "system_mean_waiting_time", "system_accumulated_mean_waiting_time", "system_mean_speed", "system_cars_present", "agents_total_accumulated_waiting_time (100)", "agents_total_accumulated_waiting_time (delta)", "agents_total_stopped", "agents_mean_waiting_time (100)", "agents_mean_waiting_time (delta)", "agents_mean_speed", "agents_cars_present"]
  y_names = ["system_total_waiting_time (s)", "system_accumulated_waiting_time (100) (s)", "system_accumulated_waiting_time (delta) (s)","system_total_stopped (stopped vehicles)", "system_mean_waiting_time (s)", "system_accumulated_mean_waiting_time (s)", "system_mean_speed (m/s)", "system_cars_present", "agents_total_accumulated_waiting_time (100) (s)", "agents_total_accumulated_waiting_time (delta) (s)", "agents_total_stopped (stopped vehicles)", "agents_mean_waiting_time (100) (s)", "agents_mean_waiting_time (delta) (s)", "agents_mean_speed (m/s)", "agents_cars_present"]

  # Create a single PDF file to save all subplots
  para = argparse.ArgumentParser(
          formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
      )
  para.add_argument("-f", nargs="+", required=True, help="Measures files\n")
  pr = para.parse_args()
  filenames = pr.f
  pdf_name = getPDFName(filenames)

  pdf_filename = f"./plots/{pdf_name}.pdf"
  pdf_pages = PdfPages(pdf_filename)
  colors = setup_graphs(25)


  for y_axis_variable, y_name in zip(y_variables, y_names):
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

      args = prs.parse_args()
      labels = cycle([s.split("/")[-1] for s in args.f]) if args.f is not None else cycle([str(i) for i in range(len(args.f))])

      # Create a subplot for this y-axis variable
      plt.figure()

      # File reading and grouping
      for file in args.f:
          main_df = pd.DataFrame()
          for f in glob.glob(file + "*"):
              df = pd.read_csv(f, sep=args.sep)
              if main_df.empty:
                  main_df = df
              else:
                  main_df = pd.concat((main_df, df))

          # Plot DataFrame
          plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)

          plt.title(args.t)
          plt.ylabel(args.ylabel)
          plt.xlabel(args.xlabel)
          plt.ylim(bottom=0)

      plt.legend()

      # if args.output is not None:
      #     plt.savefig(args.output + file + ".pdf", bbox_inches="tight")

      # Save the current subplot to the PDF pages
      pdf_pages.savefig()

  # print(meanVec)

  compare(meanVec, pdf_pages)  # Call the compare function to add the data to the PDF

  # Close the PDF file
  pdf_pages.close()
