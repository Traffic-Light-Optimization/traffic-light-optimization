# PARAMETERS
#======================
# So, in each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation duration then occurs for totalTimesteps*deltaTime = numSeconds seconds.
# This whole process is repeated for nTrials trials with different hyperparameters.
numSeconds = 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 2 # Number of times 
totalTimesteps = numSeconds*simRepeats # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
nTrials = 1; #Number of random trials to perform. 
disableMeanRewardCalculation = True # Set to false if nTrials = 1 to speed up simulation. 
type = 'Parallel' # Set to AEC for AEC type (AEC does not work)
mdl = 'DQN' # Set to DQN for DQN model
seed = '0' # or = '14154153'
best_score = -99999999
# net_file="../nets/2x2grid/2x2.net.xml";
# route_file="../nets/2x2grid/2x2.rou.xml"
net_file= "./nets/cologne1/cologne1.net.xml" 
route_file= "./nets/cologne1/cologne1.rou.xml"


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

# Remove results
current_directory = os.getcwd()
files = os.listdir(current_directory)
pattern = r'^results_sim_conn.*\.csv$'
# Delete files matching the pattern
for file in files:
    if re.match(pattern, file):
        file_path = os.path.join(current_directory, file)
        os.remove(file_path)
        print("Deleted results")

# creates a SUMO environment with multiple intersections, each controlled by a separate agent.
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file=net_file,
                                route_file=route_file,
                                use_gui=True,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_sim', #f'CSV/{type}/{mdl}/results_sim',
                                sumo_seed = seed # or = 'random'
                                )
else:
    env = sumo_rl.env(net_file=net_file,
                                route_file=route_file,
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_sim', #f'CSV/{type}/{mdl}/results_sim',
                                sumo_seed = seed # or = 'random'
                                )
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
# env = VecMonitor(env)

if mdl == 'PPO':
  model = PPO.load(f"best_multi_agent_model_{type}_{mdl}")
else:
  model = DQN.load(f"best_multi_agent_model_{type}_{mdl}")

print("Evaluating")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

print(f"\nMean reward for our trained model = {mean_reward}\n")

env.close()

#Try random phase simulation:
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file=net_file,
                                route_file=route_file,
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_rand',
                                sumo_seed = seed # or = 'random'
                                )
else:
    env = sumo_rl.env(net_file=net_file,
                                route_file=route_file,
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_rand',
                                sumo_seed = seed # or = 'random'
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