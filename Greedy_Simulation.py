import sumo_rl
from fixed_control_configs.greedy_observation import GreedyObservationFunction
from fixed_control_configs.action_lane_relationships import Junction_Action_Lanes

map = "beyers" #NB, don't forget to change this variable if you change the network, see fixed_control_configs/action_lane_relationships for map names
action_lanes = Junction_Action_Lanes[map]

env = sumo_rl.env(
    net_file="nets/beyers/beyers.net.xml",
    route_file="nets/beyers/beyers_rand.rou.xml",
    use_gui=True,
    num_seconds=3600,
    delta_time=5,
    observation_class=GreedyObservationFunction
)

def choose_action(obs: list, agent_id: str) -> int:
    max_density = -1
    result = None
    for action, lanes in action_lanes[agent_id].items():
        density = 0
        for lane in lanes:
            density += obs[lane]
            if density > max_density:
                max_density = density
                result = action
    return result

env.reset()
done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample() if not done else None
        if agent in action_lanes.keys():
            action = choose_action(observation, agent) if not done else None
        else:
            raise ValueError(f"Agent {agent} has not been implemented in the config file")
        env.step(action)
        done = termination or truncation

env.close()