from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import sumo_rl
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

env = sumo_rl.parallel_env(net_file="nets/2x2grid/2x2.net.xml",
                           route_file="nets/2x2grid/2x2.rou.xml",
                           use_gui=True,
                           num_seconds=1000,
                           delta_time=5
                           )
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
env = VecMonitor(env)

model = PPO.load('ppo_multi')

print("Evaluating")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)

print(f"\nMean reward = {mean_reward}\n")

env.close()

#Try random phase simulation:
env = sumo_rl.parallel_env(net_file="nets/2x2grid/2x2.net.xml",
                           route_file="nets/2x2grid/2x2.rou.xml",
                           use_gui=True,
                           num_seconds=1000,
                           delta_time=5
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

print(f"\nMean reward = {sum(avg_rewards)/len(avg_rewards)}\n")
env.close()