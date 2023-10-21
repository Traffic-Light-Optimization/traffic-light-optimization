import os
import sys
import shutil

# Check if the command-line argument is provided
if len(sys.argv) < 2:
    print("Usage: python script.py -f <folder_path>")
    sys.exit(1)

# Get the folder path from the command-line argument
folder_path = sys.argv[2]

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"The folder '{folder_path}' does not exist.")
    sys.exit(1)

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if it's a file and the name starts with "sim"
    if os.path.isfile(file_path) and filename.startswith("sim") and filename.count("_") > 3:
        # Split the filename by "_"
        parts = filename.split("_")

        # Check if there are at least 3 parts (to avoid index errors)
        if len(parts) >= 3:
            # Remove the second last "_" and everything after it
            new_filename = "_".join(parts[:-2]) + ".csv"
            
            # Construct the new file path
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            shutil.move(file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

print("Processing complete.")
