import os
import pandas as pd

def process_csv_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    
                    # Remove columns with less than 95% occupancy
                    occupancy_threshold = 0.95
                    df = df.dropna(thresh=int(occupancy_threshold * len(df)), axis=1)
                    
                    # Determine the maximum number of columns among all rows
                    max_num_columns = df.apply(lambda row: len(row), axis=1).max()
                    
                    # Remove extra columns (columns beyond the maximum)
                    df = df.iloc[:, :max_num_columns]
                    
                    # Save the modified DataFrame back to the file
                    df.to_csv(file_path, index=False)
                    
                    print(f"Processed {file} successfully!")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process CSV files in a directory")
    parser.add_argument("-f", "--folder", type=str, help="Directory path containing CSV files", required=True)
    args = parser.parse_args()

    folder_path = args.folder
    if os.path.isdir(folder_path):
        process_csv_files(folder_path)
    else:
        print(f"The provided path '{folder_path}' is not a directory.")
