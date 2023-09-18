import sumo_rl
import os
import re

from config_files.custom_observation import CustomObservationFunction
from config_files.custom_reward import my_reward_fn
from config_files.net_route_directories import get_file_locations

from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
from supersuit.multiagent_wrappers import pad_observations_v0
from supersuit.multiagent_wrappers import pad_action_space_v0


# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation repeats and improves on the previous model by repeating the simulation for a number of episodes
# This whole process is repeated for nTrials trials with different hyperparameters.

numSeconds = 800 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 8 # Number of episodes
parallelEnv = 1
totalTimesteps = numSeconds*simRepeats*parallelEnv # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
type = 'Parallel' # Set to AEC for AEC type (AEC does not work)
mdl = 'PPO' # Set to DQN for DQN model
seed = '12345' # or 'random'
gui = True # Set to True to see the SUMO-GUI
add_system_info = True
net_route_files = get_file_locations("cologne8") # Select a map

current_directory = os.getcwd()
new_directory = current_directory + "/results/rand/"
files = os.listdir(new_directory)
pattern = r'^results_rand.*\.csv$'
# Delete files matching the pattern
for file in files:
    if re.match(pattern, file):
        file_path = os.path.join(new_directory, file)
        os.remove(file_path)
        print("Deleted results")

rand_path = './results/rand/results_rand'

#Try random phase simulation:
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file=net_route_files["net"],
                                route_file=net_route_files["route"],
                                use_gui=gui,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=rand_path,
                                sumo_seed = seed, # or = 'random'
                                # time_to_teleport = 80
                                reward_fn=my_reward_fn,
                                observation_class=CustomObservationFunction
                                )
else:
    env = sumo_rl.env(net_file=net_route_files["net"],
                                route_file=net_route_files["route"],
                                use_gui=gui,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=rand_path,
                                sumo_seed = seed, # or = 'random'
                                # time_to_teleport = 80
                                reward_fn=my_reward_fn,
                                observation_class=CustomObservationFunction
                                )

env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
env = VecMonitor(env)

#num_agents = env.num_agents()

num_agents = 8 #Set manually for now
avg_rewards = []
obs = env.reset()

done = False
while not done:
    actions = [env.action_space.sample() for _ in range(num_agents)]  # Sample actions for all agents in all environments
    observations, rewards, dones, info = env.step(actions)
    avg_rewards.append(sum(rewards)/len(rewards))
    done = dones.any()
    # print(info)

print(f"\nMean reward for random phases simulation= {sum(avg_rewards)/len(avg_rewards)}\n")
env.close()
