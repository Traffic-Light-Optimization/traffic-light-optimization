# Custom reward functions
# ======================

def custom(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
    phase_time_rwd = traffic_signal.time_since_phase_chosen_reward() #normalized by max green times num green phases
    reward = 1*diff_wait + 50*diff_avg_speed + 0.1*phase_time_rwd
    return reward

def default(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    return diff_wait

def defandmaxgreen(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward()
    max_green_punishment = traffic_signal.max_green_reward()
    return diff_wait + max_green_punishment

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

def defandspeedwithmaxgreen(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward()
    max_green = traffic_signal.max_green_reward()

    reward = 1*diff_wait + 5*diff_avg_speed + 0.0001*max_green

    return reward

def defandspeedwithphasetimes(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward()
    time_since_phase = traffic_signal.time_since_phase_chosen_reward()

    reward = 1*diff_wait + 5*diff_avg_speed + time_since_phase

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
    reward = 1*diff_wait + 5*diff_avg_speed + 0.5*diff_pressure 

    return reward

def avgwait(traffic_signal):
    diff_avg_wait = traffic_signal._diff_avg_waiting_time_reward() # average instead of accumulated waiting time
    reward = 1*diff_avg_wait

    return reward

def avgwaitavgspeed(traffic_signal):
    diff_avg_wait = traffic_signal._diff_avg_waiting_time_reward() # average instead of accumulated waiting time
    diff_avg_speed = traffic_signal.diff_avg_speed_reward()
    reward = 1*diff_avg_wait + 0.05*diff_avg_speed

    return reward

def defandaccumlatedspeed(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_speed = traffic_signal.diff_speed_reward() 
   
    reward = 1*diff_wait + 1*diff_speed/100 

    return reward

# Create a dictionary to map reward options to functions
reward_functions = {
    'custom': custom,
    'default': default,
    'defandmaxgreen': defandmaxgreen,
    'speed': speed,
    'pressure': pressure,
    'defandspeed': defandspeed,
    'defandpress': defandpress,
    'all3': all3,
    'avgwait': all3,
    'avgwaitavgspeed': all3,
    'defandaccumlatedspeed': defandaccumlatedspeed,
    'defandspeedwithmaxgreen': defandspeedwithmaxgreen,
    'defandspeedwithphasetimes': defandspeedwithphasetimes
}

