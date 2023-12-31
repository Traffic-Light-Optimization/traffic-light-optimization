from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import sumo_rl
from supersuit.multiagent_wrappers import pad_observations_v0
from supersuit.multiagent_wrappers import pad_action_space_v0
import subprocess
from config_files.observation_class_directories import get_observation_class
from config_files.net_route_directories import get_file_locations
from config_files.delete_results import deleteSimulationResults
from config_files import reward_directories

# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for totalTimesteps. 

numSeconds = 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 8 #This parameter determines how much time in the simulation passes with each step.
max_green = 60
simRepeats = 10 # Number of episodes
parallelEnv = 1
num_cpus = 1
map = 'cologne8'
mdl = 'PPO' # Set to DQN for DQN model
# observation = 'ideal' #camera, gps, custom
# seed = "12345" # or 'random'
gui = False # Set to True to see the SUMO-GUI
yellow_time = 3 # min yellow time
reward_option = 'all3'  # default # defandmaxgreen # all3 #speed #pressure #defandspeed # defandpress
add_system_info = True
net_route_files = get_file_locations(map) # Select a map
seeds = ["89393","7356","12345", "2134", "1204234"]

# Remove results
# deleteSimulationResults(map, mdl, observation, reward_option)

observations = ["ob1", "ob4", "ob5", "ob8"]

for observation in observations:

    mean_reward = []
    i = 0

    for seed in seeds:
        i = i + 1
        # Get observation class
        observation_class = get_observation_class("model", observation)

        # Get the corresponding reward function based on the option
        reward_function = reward_directories.reward_functions.get(reward_option)

        sim_path = f"./results/observations/cologne8/sim-{map}-{mdl}-{observation}-{reward_option}_conn1_ep{i}"

        # creates a SUMO environment with multiple intersections, each controlled by a separate agent.
        env = sumo_rl.parallel_env(
            net_file=net_route_files["net"],
            route_file=net_route_files["route"],
            use_gui=gui,
            time_to_teleport=100,
            num_seconds=numSeconds, 
            delta_time=deltaTime, 
            max_green=max_green,
            out_csv_name=sim_path,
            sumo_seed = seed,
            yellow_time = yellow_time,
            reward_fn=reward_function,
            add_per_agent_info = True,
            observation_class=observation_class,
            hide_cars = True if observation == "gps" else False,
            additional_sumo_cmd=f"--additional-files {net_route_files['additional']}" if observation == "camera" else None,
        )

        env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
        env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
        env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
        env = ss.concat_vec_envs_v1(env, parallelEnv, num_cpus=num_cpus, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
        env = VecMonitor(env)

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

        # Run a manual simulation
        model.set_parameters(f"./results/observations/cologne8/{map}_{mdl}_{observation}_{reward_option}", exact_match=True, device='auto') # Set to best_model for hyper parameter tuning models
        avg_rewards = []
        obs = env.reset()
        done = False
        while not done:
            actions = model.predict(obs, deterministic=True)[0]
            obs, rewards, dones, infos = env.step(actions)
            avg_rewards.append(sum(rewards)/len(rewards))
            done = dones.any()

        print(f"\nMean reward for manual simulation {i} = {sum(avg_rewards)/len(avg_rewards)}\n")
        mean_reward.append(sum(avg_rewards)/len(avg_rewards))
        env.close()


    print(f"=======================================================\nMean reward for all simulations= {sum(mean_reward)/len(mean_reward)}\n=======================================================")

#Rename files
script_path = './rename.py'
folder_path = './results/observations/cologne8/'
subprocess.call(['python', script_path, '-f', folder_path])