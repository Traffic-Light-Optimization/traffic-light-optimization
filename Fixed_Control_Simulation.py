import sumo_rl
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action import greedy_action
from config_files.max_pressure.observation import MaxPressureObservationFunction
from config_files.max_pressure.action import max_pressure_action
from config_files.action_lane_relationships import Map_Junction_Action_Lanes
from config_files.net_route_directories import get_file_locations

type = "greedy" #greedy, max_pressure, fixed
map_name = "cologne1"
map = get_file_locations(map_name) #Use this variable to choose the network you want to use
action_lanes = Map_Junction_Action_Lanes[map_name]

env = sumo_rl.env(
    net_file=map["net"],
    route_file=map["route"],
    use_gui=True,
    num_seconds=3600,
    delta_time=5,
    observation_class=GreedyObservationFunction if type == "greedy" else MaxPressureObservationFunction,
    additional_sumo_cmd = f"--additional-files {map['additional']}",
    fixed_ts = True if type == "fixed" else False
)

env.reset()
done = False
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

env.close()