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

# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation repeats and improves on the previous model by repeating the simulation for a number of episodes
# This whole process is repeated for nTrials trials with different hyperparameters.

numSeconds = 3650 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 1 # Number of episodes
totalTimesteps = numSeconds*simRepeats # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
nTrials = 1; #Number of random trials to perform for hyperparameter tuning. 
disableMeanRewardCalculation = True # Set to false if nTrials = 1 to speed up simulation. 
type = 'Parallel' # Set to AEC for AEC type (AEC does not work)
mdl = 'PPO' # Set to DQN for DQN model
seed = '0' # or 'random'
best_score = -99999999
gui = False # Set to True to see the SUMO-GUI
add_system_info = False

#NET FILES:
#=============
# net_file="./nets/2x2grid/2x2.net.xml",
# route_file="./nets/2x2grid/2x2.rou.xml"
# net_file= "./nets/ingolstadt7/ingolstadt7.net.xml" 
# route_file= "./nets/ingolstadt7/ingolstadt7.rou.xml"
#net_file="./nets/beyers/beyers.net.xml"
#route_file= "./nets/beyers/beyers.rou.xml"
net_file= "./nets/cologne3/cologne3.net.xml" 
route_file= "./nets/cologne3/cologne3.rou.xml"

# Remove results
current_directory = os.getcwd()
new_directory = current_directory + "/results/sim/"
files = os.listdir(new_directory)
pattern = r'^results_sim.*\.csv$'
# Delete files matching the pattern
for file in files:
    if re.match(pattern, file):
        file_path = os.path.join(new_directory, file)
        os.remove(file_path)
        print("Deleted results")
new_directory = current_directory + "/results/rand/"
files = os.listdir(new_directory)
pattern = r'^results_rand.*\.csv$'
# Delete files matching the pattern
for file in files:
    if re.match(pattern, file):
        file_path = os.path.join(new_directory, file)
        os.remove(file_path)
        print("Deleted results")

sim_path = f'./results/sim/results_sim'

# creates a SUMO environment with multiple intersections, each controlled by a separate agent.
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file=net_file,
                                route_file=route_file,
                                use_gui=True,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_sim', #f'CSV/{type}/{mdl}/results_sim',
                                sumo_seed = seed, # or = 'random'
                                time_to_teleport = 80
                                )
else:
    env = sumo_rl.env(net_file=net_file,
                                route_file=route_file,
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_sim', #f'CSV/{type}/{mdl}/results_sim',
                                sumo_seed = seed, # or = 'random'
                                time_to_teleport = 80
                                )
    env = aec_to_parallel(env)

env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
env = VecMonitor(env)

if mdl == 'PPO':
  model = PPO.load(f"./models/best_multi_agent_model_{type}_{mdl}")
else:
  model = DQN.load(f"./models/best_multi_agent_model_{type}_{mdl}")

print("Evaluating")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)

#print(f"\nMean reward for our trained model = {mean_reward}\n")

env.close()

rand_path = f'./results/rand/results_rand'

#Try random phase simulation:
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file=net_file,
                                route_file=route_file,
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=rand_path,
                                sumo_seed = seed 
                                )
else:
    env = sumo_rl.env(net_file=net_file,
                                route_file=route_file,
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=rand_path,
                                sumo_seed = seed 
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

print(f"\nMean reward for random phases= {sum(avg_rewards)/len(avg_rewards)}\n")
env.close()