import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import sumo_rl
import tqdm
import rich

if __name__ == "__main__":
    env = sumo_rl.parallel_env(net_file="nets/2x2grid/2x2.net.xml",
                               route_file="nets/2x2grid/2x2.rou.xml",
                               use_gui=True,
                               num_seconds=3600,
                               delta_time=5,
                               out_csv_name='results'
                               )
    env = ss.pettingzoo_env_to_vec_env_v1(env) #Vectorizes the agents to train pettingzoo environments with standard single agent RL methods
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3") #Used for running multiple simulations in parallel for faster training
    env = VecMonitor(env)
    print("Environment created")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
    )

    print("Starting training")
    model.learn(total_timesteps=10000, progress_bar=True) #Each timestep refers to a single step in the environment (delta_time seconds)
    print("Training finished.")

    model.save('ppo_multi')
    print('Model saved.')
    env.close()