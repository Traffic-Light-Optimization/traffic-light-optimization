import os
import re
import argparse

def rename_files(folder_path, conn):
    # Check if the specified folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Define a regular expression pattern to match the file names
    pattern = r'^(.*_conn)(\d+)(.*)$'

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the item is a file
        if os.path.isfile(file_path):
            print(f"Processing: {filename}")
            # Extract parts of the file name using regex
            match = re.match(pattern, filename)
            print(match)
            if match:
                print(f"Matched: {filename}")
                prefix, conn_number, suffix = match.groups()
                new_conn_number = str(int(conn_number) - conn)
                new_filename = f"{prefix}{new_conn_number}{suffix}"
                new_file_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a folder")
    parser.add_argument("-f", "--folder", help="Path to the folder containing the files")
    parser.add_argument("-conn", type=int, default=0, help="The value to subtract from the 'conn' number")
    args = parser.parse_args()

    if args.folder:
        print("Renaming files in folder:", args.folder)
        rename_files(args.folder, args.conn)
    else:
        print("Please specify a folder using the -f parameter.")

if __name__ == "__main__":
    print("Starting")
    main()
