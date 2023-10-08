import random

def greedy_action(obs: list, action_lane_relationships: dict, current_action: int, time_since_phase_change: int) -> int:
    max_queue = -99999999
    result = None
    if all(observation == 0 for observation in obs) or time_since_phase_change > 1:
        actions = list(action_lane_relationships.keys())
        if len(actions) > 1:
            result = random.choice([action for action in actions if action != current_action])
        else:
            result = actions[0]
    else:
        for action, lanes in action_lane_relationships.items():
            queue = 0
            for lane in lanes:
                queue += obs[lane]
                if queue > max_queue:
                    max_queue = queue
                    result = action
    return result