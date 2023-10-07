from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
import supersuit as ss
import sumo_rl
from supersuit.multiagent_wrappers import pad_observations_v0
from supersuit.multiagent_wrappers import pad_action_space_v0

from config_files.observation_class_directories import get_observation_class
from config_files.net_route_directories import get_file_locations
from config_files.delete_results import deleteTrainingResults
from config_files import custom_reward

# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The simulation repeats and improves on the previous model by repeating the simulation for a number of episodes

numSeconds = 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 8 #This parameter determines how much time in the simulation passes with each step.
max_green = 60
simRepeats = 40 # Number of episodes
parallelEnv = 10
# evaluation_interval = 500 #How many seconds in you want to evaluate the model that is being trained to save the best one
num_cpus = 4
yellow_time = 3 # min yellow time
totalTimesteps = numSeconds*simRepeats*parallelEnv # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
map = "cologne8"
mdl = 'PPO' # Set to DQN for DQN model
observation = "ideal" #camera, gps, custom
reward_option = 'all3' # 'custom', 'default', 'defandmaxgreen','speed','defandspeed','defandpress','all3','avgwait','avgwaitavgspeed','defandaccumlatedspeed', 'defandmaxgreen'
seed = '12345' # or 'random'
gui = False # Set to True to see the SUMO-GUI
net_route_files = get_file_locations(map) # Select a map

#Model save path
model_save_path = f"./models/{map}_{mdl}_{observation}_{reward_option}"

#Delete results
deleteTrainingResults(map, mdl, observation, reward_option)

#Get observation class
observation_class =  get_observation_class("model", observation)

# Get the corresponding reward function based on the option
reward_function = custom_reward.reward_functions.get(reward_option)

# START TRAINING
# =====================
if __name__ == "__main__":
    results_path = f'./results/train/train-{map}-{mdl}-{observation}-{reward_option}'
    print(results_path)

    # creates a SUMO environment with multiple intersections, each controlled by a separate agent.
    env = sumo_rl.parallel_env(
        net_file=net_route_files["net"],
        route_file=net_route_files["route"],
        use_gui=gui,
        num_seconds=numSeconds, 
        delta_time=deltaTime, 
        max_green=max_green,
        out_csv_name=results_path,
        sumo_seed = seed,
        yellow_time = yellow_time,
        reward_fn=reward_function,
        add_per_agent_info = True,
        observation_class=observation_class,
        hide_cars = True if observation == "gps" else False,
        additional_sumo_cmd=f"--additional-files {net_route_files['additional']}" if observation == "camera" else None,
        sumo_warnings=False
    )
    env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be vectorized.
    env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be vectorized.
    env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment for each agent, allowing it to be used with standard single-agent RL methods.
    env = ss.concat_vec_envs_v1(vec_env=env, num_vec_envs=parallelEnv, num_cpus=num_cpus, base_class="stable_baselines3") # creates parallel simulations for training
    env = VecMonitor(env)

    if mdl == 'PPO':
      model = PPO(
          "MlpPolicy",
          env=env,
          verbose=0, 
          gamma=0.95, 
          n_steps=256,
          ent_coef=0.0905168,
          learning_rate=0.00062211, 
          vf_coef=0.042202,
          max_grad_norm=0.9,
          gae_lambda=0.99,
          n_epochs=6, 
          clip_range=0.3,
          batch_size= 256,
      )
    elif mdl == 'DQN':
      model = DQN(
          env=env,
          policy="MlpPolicy",
          learning_rate=1e-3, 
          batch_size= 256, 
          gamma= 0.95,
          learning_starts=0,
          buffer_size=50000,
          train_freq=1,
          target_update_interval=500,
          exploration_fraction=0.05,
          exploration_final_eps=0.01,
          verbose=0,
      )

    model.learn(total_timesteps=totalTimesteps, progress_bar=True)

    model.save(model_save_path)

    env.close()
    # eval_env.close()

