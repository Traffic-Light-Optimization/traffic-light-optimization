import sumo_rl
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action_lane_relationships import Map_Junction_Action_Lanes
from config_files.net_route_directories import get_file_locations
import csv

map_name = "cologne8"
map = get_file_locations(map_name) #Use this variable to choose the network you want to use
action_lanes = Map_Junction_Action_Lanes[map_name]

env = sumo_rl.env(
    net_file=map["net"],
    route_file=map["route"],
    use_gui=True,
    num_seconds=3600,
    delta_time=5,
    out_csv_name="results",
    observation_class=GreedyObservationFunction,
    additional_sumo_cmd = f"--additional-files {map['additional']}"
)

def choose_action(obs: list, agent_id: str) -> int:
    max_queue = -1
    result = None
    for action, lanes in action_lanes[agent_id].items():
        queue = 0
        for lane in lanes:
            queue += obs[lane]
            if queue > max_queue:
                max_queue = queue
                result = action
    return result

# OLD CODE
# =======================
# env.reset()
# done = False
# info = ""
# num_agents = 8
# count = 1

# while not done:
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
#         done = termination or truncation
#         if agent in action_lanes.keys():
#             action = choose_action(observation, agent) if not done else None
#         else:
#             raise ValueError(f"Agent {agent} has not been implemented in the config file")
#         env.step(action)
     
#         if count%8 == 0:
#          print(info)
#         count = count + 1
    
# env.close()

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
            action = choose_action(observation, agent) if not done else None
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
    with open("./results/greedy/greedy_conn1.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headings)
        writer.writeheader()
        writer.writerows(data)
