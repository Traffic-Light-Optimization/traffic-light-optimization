import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(
      style="darkgrid",
      rc={
          "figure.figsize": (7.2, 8),
          "text.usetex": False,
          "xtick.labelsize": 10,
          "ytick.labelsize": 12,
          "font.size": 15,
          "figure.autolayout": True,
          "axes.titlesize": 17,
          "axes.labelsize": 16,
          "lines.linewidth": 0.8,
          "lines.markersize": 6,
          "legend.fontsize": 8,
      },
  )

def calculate_total_score(df):
    position_points = {1: 2.5, 2: 1.5, 3: 1, 4: 0.7, 5: 0.5, 6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1, 10: 0}
    
    total_scores = {}
    
    for _, row in df.iterrows():
        model = row['Model']
        position = row['Position']

        points = position_points.get(position, 0)
        
        if model not in total_scores:
            total_scores[model] = 0
        
        total_scores[model] += points

    return total_scores

def main():
    parser = argparse.ArgumentParser(description='Plot total scores for models')
    parser.add_argument('-f', '--file', type=str, required=True, help='CSV file path')
    parser.add_argument("-t", type=str, default="Title", help="Plot title\n")
    parser.add_argument("-x", type=str, default="Not specified", help="X label\n")
    args = parser.parse_args()

    file_path = args.file
    output_path = file_path + "_rank.pdf"

    df = pd.read_csv(file_path)

    total_scores = calculate_total_score(df)

    models = list(total_scores.keys())
    scores = list(total_scores.values())

    # Create a color map to assign different colors to bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    plt.figure(figsize=(8, 8))
    bars = plt.bar(models, scores, color=colors)
    plt.xlabel(args.x)
    plt.ylabel('Total Score')
    plt.title(args.t)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    # Rotate the x-labels vertically
    plt.xticks(rotation='vertical')
    # Add total score labels on top of each bar
    # Add total score labels on top of each bar
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.05, str(round(score,2)), fontsize=10)


    # Save the plot as a PDF file
    plt.savefig(output_path, format='pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
