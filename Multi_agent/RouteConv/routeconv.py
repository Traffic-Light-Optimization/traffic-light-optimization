# Input and output file paths
input_file = 'input.txt'
output_file = 'output.txt'

# Open the input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Iterate through each line in the input file
    for line in infile:
        # Find the position of 'depart' attribute in the line
        depart_index = line.find('depart="')
        if depart_index != -1:
            # Extract the value of 'depart' attribute as a float
            depart_start = depart_index + len('depart="')
            depart_end = line.find('"', depart_start)
            if depart_end != -1:
                depart_value = float(line[depart_start:depart_end])
                
                # Subtract 57600 from the depart_value and round to 2 decimal places
                new_depart_value = round(depart_value - 25200.0, 2)
                
                # Replace the old depart value with the new one
                line = line[:depart_start] + str(new_depart_value) + line[depart_end:]
        
        # Write the modified line to the output file
        outfile.write(line)
