# Custom reward function
# ======================
#Don't forget to replace the files that need to be replaced in your python sumo-rl pip package
def my_reward_fn(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
    diff_pressure = traffic_signal.diff_pressure_reward()
    # print(f'Diff wait: {diff_wait}, Diff speed: {diff_avg_speed}, Diff pressure: {diff_pressure}')
    
    reward = 1*diff_wait + 1*diff_avg_speed # + 0.02*diff_pressure # Don't include diff pressure. It overshadows the other rewards
    # Implement exponetial punishement for waiting cars
    #Implement punishment if a phase hasn't been implemented in a while.
    # Implement reward for cars that never stopped at the robot
    return reward