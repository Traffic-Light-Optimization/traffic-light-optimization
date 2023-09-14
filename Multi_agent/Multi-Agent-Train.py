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

# PARAMETERS
#======================
# In each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation repeats and improves on the previous model by repeating the simulation for a number of episodes
# This whole process is repeated for nTrials trials with different hyperparameters.

numSeconds = 3650 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 5 # Number of episodes
totalTimesteps = numSeconds*simRepeats # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
type = 'Parallel' # Set to AEC for AEC type (AEC does not work)
mdl = 'DQN' # Set to DQN for DQN model
seed = '2343' # or 'random'
best_score = -99999999
gui = False # Set to True to see the SUMO-GUI
add_system_info = True

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

# Delete results
current_directory = os.getcwd()
new_directory = current_directory + "/results/train/"
files = os.listdir(new_directory)
pattern = r'^results.*\.csv$'
# Delete files matching the pattern
for file in files:
    if re.match(pattern, file):
        file_path = os.path.join(new_directory, file)
        os.remove(file_path)
    print("Deleted results")

# Define optuna parameters
study_name = f"multi-agent-tuned-using-optuma-{type}-{mdl}"
storage_url = f"sqlite:///optuna/multi-tuned-{type}-{mdl}-db.sqlite3"
file_to_delete = f"./optuna/multi-tuned-{type}-{mdl}-db.sqlite3"

# Check if the file exists before attempting to delete it
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"{file_to_delete} has been deleted.")
else:
    print(f"{file_to_delete} does not exist in the current directory.")


# Custom reward function
# ======================
#Don't forget to replace the files that need to be replaced in your python sumo-rl pip package
def my_reward_fn(traffic_signal):
    diff_wait = traffic_signal._diff_waiting_time_reward() # Default reward
    diff_avg_speed = traffic_signal.diff_avg_speed_reward() 
    diff_pressure = traffic_signal.diff_pressure_reward()
    reward = 0.5*diff_wait + 0.3*diff_avg_speed + 0.2*diff_pressure
    return reward

# Custom observation space
# =========================
from sumo_rl.environment.observations import ObservationFunction
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np

class CustomObservationFunction(ObservationFunction):
    """Custom observation function for traffic signals."""

    ## Default observation 

        # The default observation for each traffic signal agent is a vector:
        # obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

            # phase_one_hot is a one-hot encoded vector indicating the current active green phase
            # min_green is a binary variable indicating whether min_green seconds have already passed in the current phase
            # lane_i_density is the number of vehicles in incoming lane i dividided by the total capacity of the lane
            # lane_i_queue is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
#Replace with custom observation
# ---------------------------------------
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation
# -------------------------------------
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
# Replace with custom observation space
# -------------------------------------
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
# -----------------------------------

# START TRAINING
# =====================
if __name__ == "__main__":
    results_path = f'./results/train/results-{type}-{mdl}'
    print(results_path)

    # creates a SUMO environment with multiple intersections, each controlled by a separate agent.
    if type == 'Parallel':
      env = sumo_rl.parallel_env(net_file=net_file,
                                route_file=route_file,
                                use_gui=gui,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=results_path,
                                sumo_seed = seed,
                                add_system_info = add_system_info,
                                # time_to_teleport=120,
                                # reward_fn=my_reward_fn,
                                # obs_func=CustomObservationFunction
                                )
    else:
       env = sumo_rl.env(net_file=net_file,
                                route_file=route_file,
                                use_gui=gui,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=results_path,
                                sumo_seed = seed,
                                add_system_info = add_system_info,
                                #time_to_teleport=80,
                                # reward_fn=my_reward_fn,
                                # obs_func=CustomObservationFunction
                                )
       env = aec_to_parallel(env)
       
    env = pad_action_space_v0(env) # pad_action_space_v0 function pads the action space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
    env = pad_observations_v0(env) # pad_observations_v0 function pads the observation space of each agent to be the same size. This is necessary for the environment to be compatible with stable-baselines3.
    env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
    env = ss.concat_vec_envs_v1(env, 3, num_cpus=1, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
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

    model.save(f"./models/best_multi_agent_model_{type}_{mdl}")
    print("model saved")

    env.close() # Verify that this does not break the code
