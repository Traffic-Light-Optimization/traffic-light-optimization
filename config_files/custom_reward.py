# Custom reward function
# ======================
#Don't forget to replace the files that need to be replaced in your python sumo-rl pip package
def my_reward_fn(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
    diff_pressure = traffic_signal.diff_pressure_reward()
    # reward_highest_occupancy = traffic_signal.reward_highest_occupancy_phase()

    # print(f'Intersection ID: {traffic_signal.get_id()}, Diff wait: {diff_wait}, Diff speed: {5*diff_avg_speed}, Diff pressure: {0.5*diff_pressure}')
    
    reward = 1*diff_wait + 5*diff_avg_speed # + 10*diff_pressure 

    return reward