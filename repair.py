import os
import pandas as pd

def get_average_comma_count(lines):
    total_comma_count = 0
    line_count = 0

    # Calculate the total comma count in lines 2 to 10
    for line in lines[1:11]:  # Lines are 0-indexed
        total_comma_count += line.count(',')
        line_count += 1

    # Calculate the average comma count (rounded to the nearest whole number)
    if line_count > 0:
        average_comma_count = round(total_comma_count / line_count)
    else:
        average_comma_count = 0

    return average_comma_count

def remove_additional_columns(file_path):
    try:
        # Read the CSV file as a text file line by line
        with open(file_path, 'r') as txt_file:
            lines = txt_file.readlines()

        # Determine the average comma count in lines 2 to 10
        average_comma_count = get_average_comma_count(lines)

        # Process and modify the lines
        modified_lines = []

        # Append the very first line to modified_lines
        modified_lines.append(lines[0])
                              
        for line in lines[1:]:
            comma_count = line.count(',')
            result = ''
            if comma_count > average_comma_count:
               # Split the line at the average comma count and keep the portion before it
                parts = line.split(',', average_comma_count + 1)
                result = parts[0]
                for i in range(average_comma_count):
                    result = result + ',' + parts[i+1]
                
                elements = line.split(',')
                first_element = elements[0].strip()  # Get the first element and remove leading/trailing whitespace

                try:
                    float(first_element)  # Try converting the first element to a float
                    # Join the parts with commas to reformat the line
                    modified_lines.append(result + '\n')
                except ValueError:
                    print("Not a number:", first_element)
            else:
                elements = line.split(',')
                first_element = elements[0].strip()  # Get the first element and remove leading/trailing whitespace

                try:
                    float(first_element)  # Try converting the first element to a float
                    # Join the parts with commas to reformat the line
                    modified_lines.append(line)
                except ValueError:
                    print("Not a number:", first_element)
                

        # Save the modified lines back to the file
        with open(file_path, 'w') as txt_file:
            txt_file.writelines(modified_lines)

        # print(f"Processed {file_path} successfully!")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_csv_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                remove_additional_columns(file_path)

                try:
                    df = pd.read_csv(file_path)

                    # Exclude the first column (assuming it contains column titles)
                    columns_to_convert = df.columns[1:]

                    # Convert the selected columns to numeric
                    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

                    #Remove rows with values greater than 30 times the column average
                    row_threshold = 30
                    # Calculate the mean for each column
                    # Calculate the mean for each numeric column
                    numeric_columns = df.select_dtypes(include=['number'])  # Select only numeric columns
                    column_means = numeric_columns.mean()

                    # Apply the comparison for each column individually
                    df = df[df.apply(lambda row: (row[numeric_columns.columns] <= column_means * row_threshold), axis=1).all(axis=1)]
                
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
