from sumo_rl.environment.env import SumoEnvironment
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action import greedy_action
from config_files.max_pressure.observation import MaxPressureObservationFunction
from config_files.max_pressure.action import max_pressure_action
from config_files.gps.observation import GpsObservationFunction
from config_files.action_lane_relationships import Map_Junction_Action_Lanes
from config_files.net_route_directories import get_file_locations
import csv

type = "greedy" #greedy, max_pressure, fixed
map_name = "cologne1" #choose the map to simulate
map = get_file_locations(map_name) #obtain network, route, and additional files
gui = True #SUMO gui
hide_cars = True #Required for GPS observation
num_seconds = 3600 #episode duration
delta_time = 5 #step duration
action_lanes = Map_Junction_Action_Lanes[map_name] #dict of relationships between actions and lanes for each intersection

env = SumoEnvironment(
    net_file=map["net"],
    route_file=map["route"],
    use_gui=gui,
    num_seconds=num_seconds,
    delta_time=delta_time,
    # observation_class=GreedyObservationFunction if type == "greedy" else MaxPressureObservationFunction,
    observation_class=GpsObservationFunction,
    additional_sumo_cmd = f"--additional-files {map['additional']}",
    fixed_ts = True if type == "fixed" else False,
    hide_cars=hide_cars
)

data = [] #initialize a list to store the data
observations = env.reset()
done = False
while not done:
    if type == "greedy":
        actions = {agent: greedy_action(observations[agent], action_lanes[agent]) for agent in env.ts_ids}
    elif type == "max_pressure":
        actions = {agent: max_pressure_action(observations[agent], action_lanes[agent]) for agent in env.ts_ids}
    elif type == "fixed":
        actions = {}
    else:
        raise ValueError(f"{type} is an invalid type")
    observations, rewards, dones, infos = env.step(actions)
    data.append(infos.copy())
    done = dones['__all__']

headings = data[0].keys()
env.close()

# Create a CSV file and write the data to it
if data:
    with open(f"./results/{type}/{map_name}-{type}_conn1.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headings)
        writer.writeheader()
        writer.writerows(data)