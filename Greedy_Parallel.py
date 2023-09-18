import sumo_rl
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action import greedy_action
from config_files.action_lane_relationships import Map_Junction_Action_Lanes
from config_files.net_route_directories import get_file_locations

map_name = "cologne8"
map = get_file_locations(map_name) #Use this variable to choose the network you want to use
action_lanes = Map_Junction_Action_Lanes[map_name]

env = sumo_rl.parallel_env(
    net_file=map["net"],
    route_file=map["route"],
    use_gui=True,
    num_seconds=4000,
    delta_time=5,
    out_csv_name="results",
    observation_class=GreedyObservationFunction,
    additional_sumo_cmd = f"--additional-files {map['additional']}",
)

observations, infos = env.reset()
done = False
while not done:
    actions = {agent: greedy_action(observations[agent], action_lanes[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = all(terminations.values()) or all(truncations.values())

env.close()
