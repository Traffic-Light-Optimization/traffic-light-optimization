import argparse
import os
import pandas as pd

def main(folder_path):
    # Get a list of all CSV files in the specified folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    # Group CSV files by their common prefixes
    file_groups = {}
    for csv_file in csv_files:
        prefix = os.path.splitext(csv_file)[0]
        if "average" in prefix:
            continue
        prefix = prefix.split("_conn")[0]
        if prefix in file_groups:
            file_groups[prefix].append(csv_file)
        else:
            file_groups[prefix] = [csv_file]

    for prefix, files in file_groups.items():
        # Read and combine the CSV files with the same prefix
        data_frames = [pd.read_csv(os.path.join(folder_path, file)) for file in files]
        combined_df = pd.concat(data_frames, ignore_index=True)

        # Calculate the mean for each column within the combined DataFrame
        averaged_df = combined_df.mean().reset_index()
        averaged_df = averaged_df.T

        # Create the output filename with "_average" at the end
        output_filename = os.path.join(folder_path, f"{prefix}-average_conn1_ep1.csv")

        # Save the averaged DataFrame to a new CSV file
        averaged_df.to_csv(output_filename, header=False, index=False)
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average CSV files in a folder")
    parser.add_argument("-f", help="Folder path containing CSV files")
    args = parser.parse_args()

    folder_path = args.f

    if not os.path.exists(folder_path):
        print(f"The specified folder '{folder_path}' does not exist.")
    else:
        main(folder_path)
