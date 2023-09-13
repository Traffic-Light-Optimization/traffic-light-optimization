numSeconds = 3600 # This parameter determines the total duration of the SUMO traffic simulation in seconds.
deltaTime = 4 #This parameter determines how much time in the simulation passes with each step.
type = 'Parallel' # Set to AEC for AEC type
mdl = 'PPO' # Set to DQN for DQN model
seed = 'random' # or = '14154153'

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import sumo_rl
from stable_baselines3.common.evaluation import evaluate_policy 

# creates a SUMO environment with multiple intersections, each controlled by a separate agent.
if type == 'Parallel':
   env = sumo_rl.parallel_env(net_file="../nets/2x2grid/2x2.net.xml",
                                route_file="../nets/2x2grid/2x2.rou.xml",
                                use_gui=True,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name='results_sim', #f'CSV/{type}/{mdl}/results_sim',
                                sumo_seed = seed # or = 'random'
                                )
else:
    env = sumo_rl.env(net_file="../nets/2x2grid/2x2.net.xml",
                                route_file="../nets/2x2grid/2x2.rou.xml",
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
   env = sumo_rl.parallel_env(net_file="../nets/2x2grid/2x2.net.xml",
                                route_file="../nets/2x2grid/2x2.rou.xml",
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=f'CSV/random/{type}/{mdl}/results',
                                sumo_seed = seed # or = 'random'
                                )
else:
    env = sumo_rl.env(net_file="../nets/2x2grid/2x2.net.xml",
                                route_file="../nets/2x2grid/2x2.rou.xml",
                                use_gui=False,
                                num_seconds=numSeconds, 
                                delta_time=deltaTime, 
                                out_csv_name=f'CSV/random/{type}/{mdl}/results',
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