import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import sumo_rl
import tqdm
import rich

def objective(trial):
    env = sumo_rl.parallel_env(net_file="nets/2x2grid/2x2.net.xml",
                               route_file="nets/2x2grid/2x2.rou.xml",
                               use_gui=True,
                               num_seconds=3600,
                               delta_time=5
                               )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=trial.suggest_float("gamma", 0.9, 0.99),
        n_steps=int(trial.suggest_float("n_steps", 100, 500)),
        ent_coef=trial.suggest_float("ent_coef", 0.01, 0.1),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),
        vf_coef=trial.suggest_float("vf_coef", 0.01, 0.1),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.5, 1),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        n_epochs=int(trial.suggest_float("n_epochs", 1, 10)),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.4),
        batch_size=int(trial.suggest_float("batch_size", 100, 500)),
    )

    model.learn(total_timesteps=10000)
    rewards = env.get_episode_rewards()
    return sum(rewards) / len(rewards)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
