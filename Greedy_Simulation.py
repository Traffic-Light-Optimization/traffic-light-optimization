import sumo_rl
from fixed_control_configs.greedy.observation import GreedyObservationFunction
from fixed_control_configs.greedy.action_lane_relationships import Map_Junction_Action_Lanes

map = "beyers" #Use this variable to choose the network you want to use
action_lanes = Map_Junction_Action_Lanes[map]
net_file = f"nets/{map}/{map}.net.xml"
route_file = f"nets/{map}/{map}.rou.xml"
if map == "beyers":
    route_file = f"nets/{map}/{map}_rand.rou.xml"
additional_file = f"nets/{map}/{map}.add.xml"

env = sumo_rl.env(
    net_file=f"Multi_agent/nets/{map}/{map}.net.xml",
    route_file=f"Multi_agent/nets/{map}/{map}.rou.xml",
    use_gui=True,
    num_seconds=3600,
    delta_time=5,
    observation_class=GreedyObservationFunction,
    additional_sumo_cmd = f"--additional-files Multi_agent/nets/{map}/{map}.add.xml"
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
        done = termination or truncation
        if agent in action_lanes.keys():
            action = choose_action(observation, agent) if not done else None
        else:
            raise ValueError(f"Agent {agent} has not been implemented in the config file")
        env.step(action)

env.close()