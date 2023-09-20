import optuna
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
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
from pettingzoo.utils.wrappers import ClipOutOfBoundsWrapper

from config_files.custom_observation import CustomObservationFunction
from config_files.custom_reward import my_reward_fn
from config_files.net_route_directories import get_file_locations
from config_files.delete_results import deleteTrainingResults

# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation repeats and improves on the previous model by repeating the simulation for a number of episodes
# This whole process is repeated for nTrials trials with different hyperparameters.

numSeconds = 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 32 # Number of episodes
parallelEnv = 1
totalTimesteps = numSeconds*simRepeats*parallelEnv # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
type = 'Parallel' # Set to AEC for AEC type (AEC does not work)
mdl = 'PPO' # Set to DQN for DQN model
map = "cologne8"
seed = '12345' # or 'random'
gui = False # Set to True to see the SUMO-GUI
add_system_info = True
net_route_files = get_file_locations(map) # Select a map

#Delete results
deleteTrainingResults(map, type, mdl)

# START TRAINING
# =====================
if __name__ == "__main__":
    results_path = f'./results/train/results-{map}-{type}-{mdl}'
    print(results_path)

    # creates a SUMO environment with multiple intersections, each controlled by a separate agent.
    if type == 'Parallel':
      env = sumo_rl.parallel_env(
          net_file=net_route_files["net"],
          route_file=net_route_files["route"],
          use_gui=gui,
          num_seconds=numSeconds, 
          delta_time=deltaTime, 
          out_csv_name=results_path,
          sumo_seed = seed,
          add_system_info = add_system_info,
          # time_to_teleport=120,
          reward_fn=my_reward_fn,
          observation_class=CustomObservationFunction
      )
    else:
      env = sumo_rl.env(
          net_file=net_route_files["net"],
          route_file=net_route_files["route"],
          use_gui=gui,
          num_seconds=numSeconds, 
          delta_time=deltaTime, 
          out_csv_name=results_path,
          sumo_seed = seed,
          add_system_info = add_system_info,
          #time_to_teleport=80,
          reward_fn=my_reward_fn,
          observation_class=CustomObservationFunction
      )
      env = aec_to_parallel(env)
       
    env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
    env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
    env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
    env = ss.concat_vec_envs_v1(env, parallelEnv, num_cpus=1, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
    env = VecMonitor(env)

    if mdl == 'PPO':
      model = PPO(
          "MlpPolicy",
          env=env,
          verbose=3, 
          gamma=0.95, # gamma=trial.suggest_float("gamma", 0.9, 0.99),
          n_steps=256,  # n_steps=int(trial.suggest_int("n_steps", 100, 500)), # This is the number of steps of interaction (state-action pairs) that are used for each update of the policy.
          ent_coef=0.0905168, # ent_coef=trial.suggest_float("ent_coef", 0.01, 0.1),
          learning_rate=0.00062211,  #learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),
          vf_coef=0.042202,
          max_grad_norm=0.9,
          gae_lambda=0.99,
          n_epochs=5,  #n_epochs=int(trial.suggest_int("n_epochs", 5, 10, step=1)),
          clip_range=0.3,
          batch_size= 256,  #batch_size=int(trial.suggest_int("batch_size", 128, 512, step=128)),
      )
    elif mdl == 'DQN':
      model = DQN(
          env=env,
          policy="MlpPolicy",
          learning_rate=1e-3, #learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),
          batch_size= 256, #batch_size=int(trial.suggest_int("batch_size", 128, 512, step=128)),
          gamma= 0.95,
          learning_starts=0,
          buffer_size=50000,
          train_freq=1,
          target_update_interval=500, #update the target network every ``target_update_interval`` environment steps.
          exploration_fraction=0.05,
          exploration_final_eps=0.01,
          verbose=3,
      )

    model.learn(total_timesteps=totalTimesteps, progress_bar=True)

    model.save(f"./models/best_multi_agent_model_{map}_{type}_{mdl}")
    print("model saved")

    env.close() # Verify that this does not break the code
