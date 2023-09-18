def max_pressure_action(obs: list[list[str]], action_lane_relationships) -> int: #obs is list of current vehicle ids in each lane
    max_pressure = -99999999
    result = None
    for action, lanes in action_lane_relationships.items():
        pressure = 0
        for lane in lanes:
            pressure += obs[lane]
            if pressure > max_pressure:
                max_pressure = pressure
                result = action
    return result