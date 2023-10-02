import random

def max_pressure_action(obs: list[list[str]], action_lane_relationships) -> int: #obs is list of current vehicle ids in each lane
    max_pressure = -99999999
    result = None
    if all(observation == 0 for observation in obs):
        result = random.choice(list(action_lane_relationships.keys()))
    else:
        for action, lanes in action_lane_relationships.items():
            pressure = 0
            for lane in lanes:
                pressure += obs[lane]
                if pressure > max_pressure:
                    max_pressure = pressure
                    result = action
    return result