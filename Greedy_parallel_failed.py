import sumo_rl
from config_files.greedy.observation import GreedyObservationFunction
from config_files.greedy.action_lane_relationships import Map_Junction_Action_Lanes
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

observations = env.reset()
done = False
while not done:
    actions = {agent: choose_action(observations[agent], agent) for agent in env.agent_iter()}
    observations, rewards, dones, infos = env.step(actions)
    print(infos)
    done = any(dones.values())

env.close()
