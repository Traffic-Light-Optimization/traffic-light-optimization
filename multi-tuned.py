import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import sumo_rl
import os
from stable_baselines3.common.evaluation import evaluate_policy

file_to_delete = "db.sqlite3"

# Check if the file exists before attempting to delete it
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"{file_to_delete} has been deleted.")
else:
    print(f"{file_to_delete} does not exist in the current directory.")


def objective(trial):
    env = sumo_rl.parallel_env(net_file="nets/2x2grid/2x2.net.xml",
                               route_file="nets/2x2grid/2x2.rou.xml",
                               use_gui=False,
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
        gamma=0.95,
        n_steps=128,
        ent_coef=0.0905168,
        # learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        clip_range=0.3,
        # gamma=trial.suggest_float("gamma", 0.9, 0.99),
        # n_steps=int(trial.suggest_int("n_steps", 100, 500)),
        # ent_coef=trial.suggest_float("ent_coef", 0.01, 0.1),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),
        # vf_coef=trial.suggest_float("vf_coef", 0.01, 0.1),
        # max_grad_norm=trial.suggest_float("max_grad_norm", 0.5, 1),
        # gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        n_epochs=int(trial.suggest_int("n_epochs", 5, 10, step=1)),
        # clip_range=trial.suggest_float("clip_range", 0.1, 0.4),
        batch_size=int(trial.suggest_int("batch_size", 128, 512, step=128)),
    )

    model.learn(total_timesteps=10000)
    # rewards = env.get_episode_rewards()
    # rewards =random.randint(1, 100) 
    # return sum(rewards) / len(rewards)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="multi-tuned-using-optuma",
        direction="maximize"
    )
    study.optimize(objective, n_trials=100)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
