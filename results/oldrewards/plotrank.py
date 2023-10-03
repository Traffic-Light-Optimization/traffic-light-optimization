import matplotlib.pyplot as plt
import numpy as np

# Read the data from the text file
with open('rankcologne8.txt', 'r') as file:
    data = file.read().splitlines()

# Define the groups and labels
groups = ['system_total_waiting_time', 'system_total_stopped', 'system_mean_waiting_time', 'system_mean_speed', 'system_cars_present']
labels = [
    'cologne8-PPO-ideal-avgwaitavgspeed_conn1',
    'cologne8-PPO-ideal-defandspeed_conn1',
    'cologne8-PPO-ideal-all3_conn1',
    'cologne8-PPO-ideal-default_conn1',
    'cologne8-PPO-ideal-defandpress_conn1',
    'cologne8-PPO-ideal-avgwait_conn1',
    'cologne8-PPO-ideal-speed_conn1',
    'cologne8-PPO-ideal-defandmaxgreen_conn1'
]

# Create a dictionary to store data for each label and type
data_dict = {}
for label in labels:
    data_dict[label] = {
        'train': [],
        'last': [],
        'sim': []
    }

# Parse the data
current_group = None
current_label = None
current_type = None

for line in data:
    if line.startswith('Group for Y-Axis:'):
        current_group = line.split(': ')[-1]
    elif line.startswith('Label:'):
        current_label = line.split(': ')[-1]
    elif line.startswith('Label:'):
        current_type = line.split(', ')[1]
    elif line.startswith('Y-Axis:'):
        y_axis_value = float(line.split(': ')[-1])
        data_dict[current_label][current_type].append(y_axis_value)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors and markers for each type
colors = {'train': 'red', 'last': 'blue', 'sim': 'green'}
markers = {'train': 'o', 'last': 's', 'sim': 'x'}

# Plot the data
x = np.arange(len(labels))
width = 0.2

for i, label in enumerate(labels):
    for j, group in enumerate(groups):
        for k, type_name in enumerate(['train', 'last', 'sim']):
            values = data_dict[label][type_name]
            # Sort the values while keeping track of the original indices
            sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1])
            # Create a dictionary of ranks for each index
            ranks = {index: rank + 1 for rank, (index, _) in enumerate(sorted_values_with_indices)}
            # Get the rank for the current label
            rank = ranks[i] if i in ranks else len(ranks) + 1  # Assign a rank outside the range if not found
            ax.bar(
                x[i] + k * width,
                rank,  # Use the calculated rank
                width,
                label=f'{type_name} - {group}',
                color=colors[type_name],
                alpha=0.7,
                hatch=markers[type_name],  # Use hatch for markers
                edgecolor='black'  # Add edgecolor for better visibility
            )

# Customize the plot
ax.set_xlabel('Label')
ax.set_ylabel('Rank')
ax.set_title('Rankings of Labels for Different Types and Groups')
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', fontsize='small')

# Show the plot
plt.tight_layout()
plt.show()
