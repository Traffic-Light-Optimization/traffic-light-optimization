import sumo_rl
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action_lane_relationships import Map_Junction_Action_Lanes
from config_files.net_route_directories import get_file_locations

map = "ingolstadt21" #NB, don't forget to change this variable if you change the network, see fixed_control_configs/action_lane_relationships for map names
action_lanes = Map_Junction_Action_Lanes[map]
net_route_files = get_file_locations(map) # Select a map

env = sumo_rl.env(
    net_file=net_route_files["net"],
    route_file=net_route_files["route"],
    use_gui=True,
    num_seconds=3600,
    delta_time=5,
    observation_class=GreedyObservationFunction
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

env.reset()
done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if agent in action_lanes.keys():
            action = choose_action(observation, agent) if not done else None
        else:
            raise ValueError(f"Agent {agent} has not been implemented in the config file")
        env.step(action)
        done = termination or truncation

env.close()