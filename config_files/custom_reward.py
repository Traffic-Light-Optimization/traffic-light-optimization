# Custom reward function
# ======================
#Don't forget to replace the files that need to be replaced in your python sumo-rl pip package
def custom(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
    # reward_highest_occupancy = traffic_signal.reward_highest_occupancy_phase()
    print(f'Intersection ID: {traffic_signal.get_id()}, Diff wait: {diff_wait}, Diff speed: {5*diff_avg_speed}')
    reward = 1*diff_wait + 5*diff_avg_speed
    return reward

def default(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    return diff_wait

def speed(traffic_signal):
    diff_avg_speed = traffic_signal.diff_avg_speed_reward()
    return 5*diff_avg_speed

def pressure(traffic_signal):
    diff_pressure = traffic_signal.diff_pressure_reward()
    return 10*diff_pressure

def defandspeed(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
   
    reward = 1*diff_wait + 5*diff_avg_speed 

    return reward

def defandpress(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_pressure = traffic_signal.diff_pressure_reward()
   
    reward = 1*diff_wait + 0.5*diff_pressure 

    return reward

def all3(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
    diff_pressure = traffic_signal.diff_pressure_reward()
    # print(f'Intersection ID: {traffic_signal.get_id()}, Diff wait: {diff_wait}, Diff speed: {5*diff_avg_speed}, Diff pressure: {0.5*diff_pressure}')
    reward = 1*diff_wait + 5*diff_avg_speed + 0.5*diff_pressure 

    return reward

# Create a dictionary to map reward options to functions
reward_functions = {
    'custom': custom,
    'default': default,
    'speed': speed,
    'pressure': pressure,
    'defandspeed': defandspeed,
    'defandpress': defandpress,
    'all3': all3
}

