import argparse
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.bandit_env import MultiArmedBanditEnv
from envs.continuous_bandit_env import ContinuousBanditEnv
from envs.socket_env import SocketAppEnv
from config import get_config


def make_env(name: str) -> Callable[[], gym.Env]:
    """Return a function that creates the requested environment."""
    if name == "bandit":
        def _f() -> gym.Env:
            return MultiArmedBanditEnv([0.43, 0.57])
        return _f
    if name == "continuous":
        def _f() -> gym.Env:
            return ContinuousBanditEnv(optimal_action=0.2)
        return _f
    if name == "socket":
        cfg = get_config()

        def _f() -> gym.Env:
            return SocketAppEnv(
                cfg.env.max_steps,
                device=cfg.env.device,
                action_dim=cfg.env.action_dim,
                state_dim=cfg.env.state_dim,
                action_host=cfg.env.action_host,
                action_port=cfg.env.action_port,
                reward_host=cfg.env.reward_host,
                reward_port=cfg.env.reward_port,
                embedding_model=cfg.env.embedding_model,
                combined_server=cfg.env.combined_server,
                start_servers=cfg.env.start_servers,
                enable_logging=cfg.env.enable_logging,
                use_world_model=cfg.env.use_world_model,
                world_model_host=cfg.env.world_model.host,
                world_model_port=cfg.env.world_model.port,
                world_model_path=cfg.env.world_model.model_path,
                world_model_type=cfg.env.world_model.model_type,
                world_model_interval=cfg.env.world_model.interval_steps,
            )
        return _f
    raise ValueError(f"Unknown env '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable Baselines3 PPO trainer")
    parser.add_argument(
        "--env", choices=["bandit", "continuous", "socket"], default="continuous",
        help="Environment to train on")
    parser.add_argument(
        "--timesteps", type=int, default=5000,
        help="Number of training timesteps")
    args = parser.parse_args()

    env_fn = make_env(args.env)
    vec_env = make_vec_env(env_fn, n_envs=1)
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print("Learned action:", action)
    vec_env.close()


if __name__ == "__main__":
    main()
