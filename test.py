import optuna
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import sumo_rl
from stable_baselines3.common.evaluation import evaluate_policy 
import os
import re
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import parallel_to_aec
from supersuit.multiagent_wrappers import pad_observations_v0
from supersuit.multiagent_wrappers import pad_action_space_v0

from config_files.custom_observation import CustomObservationFunction
from config_files.custom_reward import my_reward_fn
from config_files.net_route_directories import get_file_locations
from config_files.delete_results import deleteResults
from stable_baselines3.common.monitor import Monitor

# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation repeats and improves on the previous model by repeating the simulation for a number of episodes
# This whole process is repeated for nTrials trials with different hyperparameters.

numSeconds = 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 8 # Number of episodes
parallelEnv = 1
totalTimesteps = numSeconds*simRepeats*parallelEnv # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
type = 'Parallel' # Set to AEC for AEC type (AEC does not work)
mdl = 'PPO' # Set to DQN for DQN model
seed = '12345' # or 'random'
gui = False # Set to True to see the SUMO-GUI
add_system_info = True
net_route_files = get_file_locations("cologne8") # Select a map


rand_path = f'./results/rand/results_rand'

#Try random phase simulation:
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file=net_route_files["net"],
                                route_file=net_route_files["route"],
                                use_gui=True,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=rand_path,
                                sumo_seed = seed,
                                observation_class=CustomObservationFunction,
                                reward_fn=my_reward_fn,
                                )
else:
    env = sumo_rl.env(net_file=net_route_files["net"],
                                route_file=net_route_files["route"],
                                use_gui=True,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=rand_path,
                                sumo_seed = seed,
                                observation_class=CustomObservationFunction,
                                reward_fn=my_reward_fn,
                                )
avg_rewards = []
obs, info = env.reset()
done = False
while not done:
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        avg_rewards.append(sum(rewards.values())/len(rewards))
        done = any(terminations.values()) or any(truncations.values())
        print(info)
        # sysInfo = env._get_system_info()
        # print(sysInfo)
        # print()
        # print()
    sysInfo = env._get_system_info()
    print("SystemInfo")
    print(sysInfo)

print(f"\nMean reward for random phases= {sum(avg_rewards)/len(avg_rewards)}\n")
env.close()