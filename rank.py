import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

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

def calculate_scores(df):
    position_points = {1: 2.5, 2: 1.5, 3: 1, 4: 0.7, 5: 0.5, 6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1, 10: 0}
    
    total_scores = {}
    system_scores = {}
    agent_scores = {}
    
    for _, row in df.iterrows():
        model = row['Model']
        position = row['Position']

        points = position_points.get(position, 0)
        
        if model not in total_scores:
            total_scores[model] = 0
        if model not in system_scores:
            system_scores[model] = 0
        if model not in agent_scores:
            agent_scores[model] = 0

        if row['Type'].startswith('system'):
            system_scores[model] += points
        elif row['Type'].startswith('agents'):
            agent_scores[model] += points
        total_scores[model] += points

    return total_scores, system_scores, agent_scores

def main():
    parser = argparse.ArgumentParser(description='Plot total scores for models')
    parser.add_argument('-f', '--file', type=str, required=True, help='CSV file path')
    parser.add_argument("-t", type=str, default="Title", help="Plot title\n")
    parser.add_argument("-xh", type=str, default="Not specified", help="X label\n")
    parser.add_argument("-xlist", type=str, default="Not specified", help="X list sperated by , \n")
    args = parser.parse_args()

    file_path = args.file

    df = pd.read_csv(file_path)

    total_scores, system_scores, agent_scores = calculate_scores(df)
    if args.xlist == "Not specified":
        models = list(total_scores.keys())
    else :
        print(args.xlist)
        models = args.xlist.split(",")
        print(models)
    total_scores_list = list(total_scores.values())
    system_scores_list = list(system_scores.values())
    agent_scores_list = list(agent_scores.values())

    scores = {
        'Total Score': total_scores_list,
        'System Score': system_scores_list,
        'Agents Score': agent_scores_list
    }

    
    for score_type, score_list in scores.items():
        # Sort models and scores by model names so that each model gets the same colour every time you run the script
        models = total_scores.keys()
        sorted_models = zip(models, score_list)
        sorted_models = sorted(sorted_models, reverse=True)
        models, score_list = zip(*sorted_models)

        # Sort models and scores based on score in descending order
        colors = plt.cm.viridis(np.linspace(0, 1, len(models))) 
        sorted_lists = zip(score_list, models, colors)
        sorted_lists = sorted(sorted_lists, reverse=True)
        score_list, models, colors = zip(*sorted_lists)

        plt.figure(figsize=(8, 8))
        bars = plt.bar(models, score_list, color=colors)
        plt.xlabel(args.xh)
        plt.ylabel(score_type)
        plt.title(args.t)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        # Rotate the x-labels vertically
        plt.xticks(rotation='vertical')
        # Add total score labels on top of each bar
        for bar, score in zip(bars, score_list):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.05, str(round(score,2)), fontsize=10)

        # Save the plot as a PDF file
        if score_type == "Total Score":
            output_path = os.path.splitext(file_path)[0] + "_rank_total.pdf"
        elif score_type == "System Score":
            output_path = os.path.splitext(file_path)[0] + "_rank_system.pdf"
        else:
            output_path = os.path.splitext(file_path)[0] + "_rank_agents.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Saved {output_path}")

        plt.show()

if __name__ == '__main__':
    main()
