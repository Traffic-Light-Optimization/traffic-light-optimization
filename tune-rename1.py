import os
import argparse
import re

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_file_path):
            # Extract the text within <...>
            match = re.search(r'<(.*?)>', filename)
            if match:
                text_inside_brackets = match.group(1)
                # Get the last 12 characters within the brackets
                new_text = text_inside_brackets[-12:]
                # Remove <...> brackets and replace with the new text
                new_filename = re.sub(r'<(.*?)>', new_text, filename)
                new_file_path = os.path.join(folder_path, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} to {new_filename}")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a folder.")
    parser.add_argument("-f", "--folder", required=True, help="Folder path where files need to be renamed.")
    args = parser.parse_args()

    folder_path = args.folder

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        rename_files_in_folder(folder_path)
    else:
        print(f"The specified folder '{folder_path}' does not exist or is not a directory.")

if __name__ == "__main__":
    main()
