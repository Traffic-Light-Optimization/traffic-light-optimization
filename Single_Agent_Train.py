from stable_baselines3 import PPO
from sumo_rl.environment.env import SumoEnvironment
import supersuit as ss


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="nets/2way-single-intersection/single-intersection.net.xml",
        route_file="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        single_agent=True,
        use_gui=False,
        num_seconds=1000,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
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
    model.learn(total_timesteps=10000, progress_bar=True)

    model.save('ppo_single')
    print("Saved model")
