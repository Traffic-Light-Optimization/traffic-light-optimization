import sumo_rl
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action import greedy_action
from config_files.max_pressure.observation import MaxPressureObservationFunction
from config_files.max_pressure.action import max_pressure_action
from config_files.action_lane_relationships import Map_Junction_Action_Lanes
from config_files.net_route_directories import get_file_locations
import csv

type = "max_pressure" #greedy, max_pressure, fixed (fixed does not work)
map_name = "cologne8" #choose the map to simulate
map = get_file_locations(map_name) #obtain network, route, and additional files
gui = False #SUMO gui
num_seconds = 3600 #episode duration
delta_time = 5 #step duration
action_lanes = Map_Junction_Action_Lanes[map_name] #dict of relationships between actions and lanes for each intersection

env = sumo_rl.env(
    net_file=map["net"],
    route_file=map["route"],
    use_gui=gui,
    num_seconds=num_seconds,
    delta_time=delta_time,
    observation_class=GreedyObservationFunction if type == "greedy" else MaxPressureObservationFunction,
    additional_sumo_cmd = f"--additional-files {map['additional']}",
    fixed_ts = True if type == "fixed" else False
)

# Initialize a list to store the data
data = []

env.reset()
done = False
info = ""
num_agents = 8 # Have to set this manually for now
count = 1
step = 0  # Initialize the step variable

while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation
        if agent in action_lanes.keys():
            if type == "greedy":
                action = greedy_action(observation, action_lanes[agent]) if not done else None
            else:
                action = max_pressure_action(observation, action_lanes[agent]) if not done else None
        else:
            raise ValueError(f"Agent {agent} has not been implemented in the config file")
        env.step(action)

        if count % num_agents == 0:
            # Add the "step" column to the info dictionary
            info = {'step': step, **info}
            step += 5  # Increment the step by 5 for the next entry

            # Extract the headings from the info dictionary
            if data:
                headings = data[0].keys()
            else:
                headings = info.keys()

            # Append the info dictionary to the data list
            data.append(info.copy())

        count += 1

env.close()

# Create a CSV file and write the data to it
if data:
    with open(f"./results/{type}/{map_name}-{type}_conn1.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headings)
        writer.writeheader()
        writer.writerows(data)