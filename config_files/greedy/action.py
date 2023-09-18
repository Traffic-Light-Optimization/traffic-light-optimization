def greedy_action(obs: list, action_lane_relationships) -> int:
    max_queue = -1
    result = None
    for action, lanes in action_lane_relationships.items():
        queue = 0
        for lane in lanes:
            queue += obs[lane]
            if queue > max_queue:
                max_queue = queue
                result = action
    return result