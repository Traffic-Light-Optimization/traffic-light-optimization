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
from config_files.delete_results import deleteSimulationResults, deleteRandResults
from config_files.gps.observation import GpsObservationFunction
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
map = "cologne8"
seed = '12345' # or 'random'
gui = True # Set to True to see the SUMO-GUI
hide_cars = False # Required true for GPS observation, won't affect other observation functions, just changes colours of cars
add_system_info = True
net_route_files = get_file_locations(map) # Select a map

# Remove results
deleteSimulationResults(map, type, mdl)
# deleteRandResults(map, type, mdl)

sim_path = f"./results/sim/results_sim-{map}-{type}-{mdl}"

# creates a SUMO environment with multiple intersections, each controlled by a separate agent.
if type == 'Parallel':
    env = sumo_rl.parallel_env(
        net_file=net_route_files["net"],
        route_file=net_route_files["route"],
        use_gui=gui,
        num_seconds=numSeconds, 
        delta_time=deltaTime, 
        out_csv_name=sim_path, #f'CSV/{type}/{mdl}/results_sim',
        sumo_seed = seed, # or = 'random'
        # time_to_teleport = 80
        reward_fn=my_reward_fn,
        observation_class=CustomObservationFunction,
        hide_cars = hide_cars
    )
else:
    env = sumo_rl.env(
        net_file=net_route_files["net"],
        route_file=net_route_files["route"],
        use_gui=gui,
        num_seconds=numSeconds, 
        delta_time=deltaTime, 
        out_csv_name=sim_path, #f'CSV/{type}/{mdl}/results_sim',
        sumo_seed = seed, # or = 'random'
        # time_to_teleport = 80
        reward_fn=my_reward_fn,
        observation_class=CustomObservationFunction,
        hide_cars = hide_cars
    )
    env = aec_to_parallel(env)

env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
env = VecMonitor(env)

# if mdl == 'PPO':
#   model = PPO.load(path=f"./models/best_multi_agent_model_{type}_{mdl}",
#                    env=env,
#                   #  print_system_info=True
#                    )
# else:
#   model = DQN.load(f"./models/best_multi_agent_model_{type}_{mdl}")

if mdl == 'PPO':
      model = PPO(
          env=env,
          policy="MlpPolicy",
      )
elif mdl == 'DQN':
      model = DQN(
          env=env,
          policy="MlpPolicy",
      )

print("Evaluating")
# Run old simulation simulation
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True, render=True)
# env.close()

# Run a manual simulation
model.set_parameters(f"./models/best_multi_agent_model_{map}_{type}_{mdl}", exact_match=True, device='auto')
avg_rewards = []
obs = env.reset()
done = False
while not done:
    actions = model.predict(obs, deterministic=False)[0]
    obs, rewards, dones, infos = env.step(actions)
    avg_rewards.append(sum(rewards)/len(rewards))
    done = dones.any()

print(f"\nMean reward for manual simulation= {sum(avg_rewards)/len(avg_rewards)}\n")
env.close()

# rand_path = f'./results/rand/results_rand-{map}-{type}-{mdl}'

#Try random phase simulation:
# if type == 'Parallel':
#     env = sumo_rl.parallel_env(
#         net_file=net_route_files["net"],
#         route_file=net_route_files["route"],
#         use_gui=gui,
#         num_seconds=numSeconds, 
#         delta_time=deltaTime, 
#         out_csv_name=rand_path,
#         sumo_seed = seed,
#         observation_class=CustomObservationFunction,
#         reward_fn=my_reward_fn,
#     )
# else:
#     env = sumo_rl.env(
#         net_file=net_route_files["net"],
#         route_file=net_route_files["route"],
#         use_gui=gui,
#         num_seconds=numSeconds, 
#         delta_time=deltaTime, 
#         out_csv_name=rand_path,
#         sumo_seed = seed,
#         observation_class=CustomObservationFunction,
#         reward_fn=my_reward_fn,
#     )
# avg_rewards = []
# obs, info = env.reset()
# done = False
# while not done:
#     while env.agents:
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#         observations, rewards, terminations, truncations, infos = env.step(actions)
#         avg_rewards.append(sum(rewards.values())/len(rewards))
#         done = all(terminations.values()) or all(truncations.values())

# print(f"\nMean reward for random phases= {sum(avg_rewards)/len(avg_rewards)}\n")
# env.close()