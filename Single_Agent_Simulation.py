from sumo_rl.environment.env import SumoEnvironment
from stable_baselines3 import DQN

env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                      route_file='nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                      out_csv_name='Results.csv',
                      use_gui=True,
                      single_agent=True,
                      num_seconds=10000)

model = DQN.load('dqn_single')

obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated