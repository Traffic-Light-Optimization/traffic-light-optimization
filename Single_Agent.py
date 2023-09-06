from sumo_rl.environment.env import SumoEnvironment

env = SumoEnvironment(net_file='Single.net.xml',
                      route_file='Single.rou.xml',
                      out_csv_name='Results.csv',
                      use_gui=True,
                      single_agent=True,
                      num_seconds=10000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(reward)
    done = terminated or truncated