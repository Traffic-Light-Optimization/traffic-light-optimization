numSeconds = 3600 # 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 5 #This parameter determines how much time in the simulation passes with each step.
simRepeats = 20 # Number of times 
totalTimesteps = 10000 # (numSeconds/deltaTime)*simRepeats # This is the total number of steps in the environment that the agent will take for training. It’s the overall budget of steps that the agent can interact with the environment.
nTrials = 1; #Number of random trials to perform. 
disableMeanRewardCalculation = True # Set to false if nTrials = 1. 
type = 'Parallel' # Set to AEC for AEC type
mdl = 'PPO' # Set to DQN for DQN model
seed = 'random' # or = '14154153'
best_score = -99999999

#The model will learn by taking a total of totalTimesteps steps in the environment, and it will update its policy every n_steps steps.
# For each trial this is repeted for n_epochs epochs.

#Summary
#----------
# So, in each timestep (delta_time), the agent takes an action, and the environment (the traffic simulation) advances by delta_time seconds. 
# The agent continues to take actions for total_timesteps. 
# The policy is updated every n_steps steps, and each update involves going through the batch of interactions n_epochs times.
# The simulation duration then occurs for totalTimesteps*deltaTime = numSeconds seconds.
# This whole process is repeated for nTrials trials with different hyperparameters.

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

# Remove results
current_directory = os.getcwd()
files = os.listdir(current_directory)
pattern = r'^results_conn.*\.csv$'
# Delete files matching the pattern
for file in files:
    if re.match(pattern, file):
        file_path = os.path.join(current_directory, file)
        os.remove(file_path)

print("Deleted results")

# Define optuna parameters
study_name = f"multi-agent-tuned-using-optuma-{type}-{mdl}"
storage_url = f"sqlite:///multi-tuned-{type}-{mdl}-db.sqlite3"
file_to_delete = f"multi-tuned-{type}-{mdl}-db.sqlite3"

# Check if the file exists before attempting to delete it
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"{file_to_delete} has been deleted.")
else:
    print(f"{file_to_delete} does not exist in the current directory.")

def objective(trial):
    print()
    print()
    print(f"[2] Create environment for trial {trial.number}")
    print("--------------------------------------------")

    #results_path = f'./CSV/{type}/{mdl}/results',
    results_path = 'results'
    print(results_path)

    # creates a SUMO environment with multiple intersections, each controlled by a separate agent.
    if type == 'Parallel':
      env = sumo_rl.parallel_env(net_file="../nets/2x2grid/2x2.net.xml", #
                                route_file="../nets/2x2grid/2x2.rou.xml",
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=results_path,
                                # sumo_seed = seed # or = 'random'
                                )
    else:
       env = sumo_rl.env(net_file="../nets/2x2grid/2x2.net.xml",
                                route_file="../nets/2x2grid/2x2.rou.xml",
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=results_path,
                                # sumo_seed = seed # or = 'random'
                                )
       env = aec_to_parallel(env)
    
    env = ss.pettingzoo_env_to_vec_env_v1(env) # pettingzoo_env_to_vec_env_v1 function vectorizes the PettingZoo environment, allowing it to be used with standard single-agent RL methods.
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3") # function creates 4 copies of the environment and runs them in parallel. This effectively increases the number of agents by 4 times, as each copy of the environment has its own set of agents.
    env = VecMonitor(env)

    if mdl == 'PPO':
      model = PPO(
          "MlpPolicy",
          env=env,
          verbose=3, #Change to 2
          gamma=0.95,
          n_steps=256, # This is the number of steps of interaction (state-action pairs) that are used for each update of the policy.
          ent_coef=0.0905168,
          learning_rate=0.00062211,
          vf_coef=0.042202,
          max_grad_norm=0.9,
          gae_lambda=0.99,
           n_epochs=5,
          clip_range=0.3,
          batch_size= 256,
          # gamma=trial.suggest_float("gamma", 0.9, 0.99),
          # n_steps=int(trial.suggest_int("n_steps", 100, 500)),
          # ent_coef=trial.suggest_float("ent_coef", 0.01, 0.1),
          #learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),
          # vf_coef=trial.suggest_float("vf_coef", 0.01, 0.1),
          # max_grad_norm=trial.suggest_float("max_grad_norm", 0.5, 1),
          # gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
          #n_epochs=int(trial.suggest_int("n_epochs", 5, 10, step=1)),
          # clip_range=trial.suggest_float("clip_range", 0.1, 0.4),
          #batch_size=int(trial.suggest_int("batch_size", 128, 512, step=128)),
      )
    else:
      model = DQN(
          env=env,
          policy="MlpPolicy",
          learning_rate=1e-3,
          #learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),
          batch_size= 256,
          #batch_size=int(trial.suggest_int("batch_size", 128, 512, step=128)),
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

    if not disableMeanRewardCalculation:
      mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
      print(f"Mean reward: {mean_reward} (params: {trial.params})")

      # Check if the current model is better than the best so far
      if mean_reward > best_score:
          best_score = mean_reward
          # Save the best model to a file
          model.save(f"best_multi_agent_model_{type}_{mdl}")
    else:
        mean_reward = -1
        model.save(f"best_multi_agent_model_{type}_{mdl}")

    env.close() # Verify that this does not break the code

    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(
        storage=storage_url, 
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=nTrials)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
