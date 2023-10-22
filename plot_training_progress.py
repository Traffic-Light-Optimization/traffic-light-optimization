import argparse
import glob
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv

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

def compare(compVec, pdf_name):
  grouped_data = {}
  csv_column_headings = ["Type", "Score", "Model", "Position"]

  for key, value in compVec.items():
        # Extract the y-axis name from the key
        y_axis_name = key.split(",")[1]

        if y_axis_name not in grouped_data:
            grouped_data[y_axis_name] = []

        # Append the key-value pair to the corresponding group
        grouped_data[y_axis_name].append((key, value))

  with open(f"./plots/{pdf_name}.csv", mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_column_headings)

        # Sort each group by the sum (value) in ascending order
        for y_axis_name, group in grouped_data.items():
              sorted_group = sorted(group, key=lambda x: x[1], reverse=True if y_axis_name in ["system_mean_speed", "agents_mean_speed"] else False)
              pos = 1

              for key, value in sorted_group:
                  model_name = key.split(",")[0]
                  entry = f"{y_axis_name},{value},{model_name},{pos}"
                  print(entry)
                  entry = [y_axis_name, value, model_name, pos]
                  writer.writerow(entry)
                  pos += 1

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
    # plt.figure(figsize=(8, 8))
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)
    

    # x = df[xaxis]
    # y = df[yaxis]
    # plt.plot(x, y, label=label, color=color, linestyle=next(dashes_styles))
    # # plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)
    
    # Calculate and store the sum of the second column
    sum_yaxis = df[yaxis].sum()
    count_yaxis = df[yaxis].count()
    # Create a label for the row entry in meanVec
    row_label = f"{label},{yaxis}"
    # Add the sum to the meanVec dictionary with the label as the key
    meanVec[row_label] = sum_yaxis # or avg_yaxis 

def getPDFName(filenames):
  pdf_name = ""
  if len(filenames) > 1:
    parts = filenames[0].split("/")[-1].split(".")[0].split("-", 2)
    pdf_name = parts[1]
    last = filenames[0].split("/")[-1].split(".")[0].split("-", 1)[1].split("_")[1]
    groups = ""
    for filename in filenames:
      group = filename.split("/")[-1].split(".")[0].split("-", 2)[2].split("_")[0]
      groups = groups + f"-[{group}]"

    return "training-progress-" + parts[1] + "-" + parts[0] + groups + "_" + last

  else:
    pdf_name = "training-progress-" + filenames[0].split("/")[-1].split(".")[0]
  return pdf_name

if __name__ == "__main__":

  # List of five different y-axis variables
  # y_variables = ["system_total_waiting_time", "system_accumulated_waiting_time (100)", "system_accumulated_waiting_time (delta)", "system_total_stopped", "system_mean_waiting_time", "system_accumulated_mean_waiting_time (100)", "system_accumulated_mean_waiting_time (delta)", "system_mean_speed", "system_cars_present", "agents_total_accumulated_waiting_time (100)", "agents_total_accumulated_waiting_time (delta)", "agents_total_stopped", "agents_mean_waiting_time (100)", "agents_mean_waiting_time (delta)", "agents_mean_speed", "agents_cars_present"]
  # y_names = ["system_total_waiting_time (s)", "system_accumulated_waiting_time (100) (s)", "system_accumulated_waiting_time (delta) (s)","system_total_stopped (stopped vehicles)", "system_mean_waiting_time (s)", "system_accumulated_mean_waiting_time (100) (s)", "system_accumulated_mean_waiting_time (delta) (s)", "system_mean_speed (m/s)", "system_cars_present", "agents_total_accumulated_waiting_time (100) (s)", "agents_total_accumulated_waiting_time (delta) (s)", "agents_total_stopped (stopped vehicles)", "agents_mean_waiting_time (100) (s)", "agents_mean_waiting_time (delta) (s)", "agents_mean_speed (m/s)", "agents_cars_present"]

  y_variables = ["system_total_waiting_time", "system_accumulated_waiting_time (100)", "system_total_stopped", "system_mean_waiting_time", "system_accumulated_mean_waiting_time (100)", "system_mean_speed", "system_cars_present", "agents_total_accumulated_waiting_time (100)", "agents_total_stopped", "agents_mean_waiting_time (100)", "agents_mean_speed", "agents_cars_present"]
  y_names = ["system_total_waiting_time (s)", "system_accumulated_waiting_time (100) (s)", "system_total_stopped (stopped vehicles)", "system_mean_waiting_time (s)", "system_accumulated_mean_waiting_time (100) (s)", "system_mean_speed (m/s)", "system_cars_present", "agents_total_accumulated_waiting_time (100) (s)", "agents_total_stopped (stopped vehicles)", "agents_mean_waiting_time (100) (s)", "agents_mean_speed (m/s)", "agents_cars_present"]
  
  # Create a single PDF file to save all subplots
  para = argparse.ArgumentParser(
          formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
      )
  para.add_argument("-f", nargs="+", required=True, help="Measures files\n")
  para.add_argument("-conn", type=int, default=1, help="Number of conns.\n")
  para.add_argument("-start", type=int, default=1, help="Start episode.\n")
  para.add_argument("-stop", type=int, default=10, help="Stop at episode.\n")
  para.add_argument("-t", type=str, default="Title", help="Plot title\n")
  para.add_argument("-l", nargs="+", default=None, help="File's legends\n")

  pr = para.parse_args()
  filenames = pr.f
  pdf_name = getPDFName(filenames)

  try:
      pdf_filename = f"./plots/{pdf_name}.pdf"
      pdf_pages = PdfPages(pdf_filename)
      
  except Exception as e:
      print(f"Error: {e}")
      pdf_name = "default"
      pdf_filename = f"./plots/{pdf_name}.pdf"
      pdf_pages = PdfPages(pdf_filename)

  colors = setup_graphs(len(pr.f) if len(pr.f) > 1 else 25)


  for y_axis_variable, y_name in zip(y_variables, y_names):
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
      if args.l == None:
        labels = cycle([s.split("/")[-1] for s in args.f]) if args.f is not None else cycle([str(i) for i in range(len(args.f))])
      else:
        labels = cycle(args.l)

      # Create a subplot for this y-axis variable
      plt.figure()

      # Plot each model on a single graph
      for file in args.f:
          df_ep = pd.DataFrame()
          
          for conn in range(1, pr.conn + 1):
              # Initialize an empty list to store episode data
              episode_data = []
              ep = pr.start - 1

              try:
                  # for f in glob.glob(file + str(conn) + "*"):
                  for episode_num in range(pr.start, pr.stop + 1):
                      try:
                          ep += 1
                          
                          # if "marl_train" in file:
                          #     f = file + str(conn) + f"_ep{episode_num}.csv"
                          #     df = pd.read_csv(f, sep=args.sep)[["step",  y_axis_variable]]
                          # else:
                          #     raise ValueError("The word 'marl_train' is not found in the variable 'file'.")

                          f = file + str(conn) + f"_ep{episode_num}.csv"
                          df = pd.read_csv(f, sep=args.sep)[["step",  y_axis_variable]]
                          
                        
                      except Exception as e:
                          try: 
                              f = file + str(conn) + ".csv"
                              df = pd.read_csv(f, sep=args.sep)[["step",  y_axis_variable]]
                          except Exception as e:
                              ep -= 1
                              f = file + str(conn) + f"_ep{ep}.csv"
                              # df = pd.read_csv(f, sep=args.sep)[["step",  y_axis_variable]]
                              temp = []
                              values = [item[ y_axis_variable] for item in episode_data]
                              average = sum(values) / len(values)
                              temp.append({y_axis_variable: average})
                              df = pd.DataFrame.from_records(temp)
                          
                      episode_sum = df[y_axis_variable].mean()
                      episode_data.append({"Episode": episode_num, y_axis_variable: episode_sum})
                      
                  if df_ep.empty:
                      df_ep = pd.DataFrame.from_records(episode_data)
                  else:
                      df_ep = pd.concat((df_ep, pd.DataFrame.from_records(episode_data)))
              
              except Exception as e:
                  pass
        
          # Plot DataFrame
          plot_df(df_ep, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)

          # Plot specific model f
          plt.title(args.t)
          plt.ylabel(args.ylabel)
          plt.xlabel("Episodes")
          plt.ylim(bottom=0)

      # plt.show()
      plt.legend()

      # Save the current subplot to the PDF pages
      pdf_pages.savefig()

  compare(meanVec, pdf_name)  # Call the compare function to add the data to the PDF

  # Close the PDF file
  pdf_pages.close()
