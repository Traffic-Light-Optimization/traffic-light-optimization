from sumo_rl.environment.env import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                      route_file='nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                      out_csv_name='Results.csv',
                      use_gui=True,
                      single_agent=True,
                      num_seconds=1000,
                      sumo_seed=42)

model = PPO.load('ppo_single')

obs, info = env.reset()
rewards = []
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    done = terminated or truncated

print(f"\nMean reward = {sum(rewards)/len(rewards)}\n")