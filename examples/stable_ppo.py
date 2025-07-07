import argparse
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.bandit_env import MultiArmedBanditEnv
from envs.continuous_bandit_env import ContinuousBanditEnv
from envs.socket_env import create_socket_env
from servers.manager import ServerManager
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
        manager = ServerManager() if cfg.env.start_servers else None

        def _f() -> gym.Env:
            return create_socket_env(cfg.env, server_manager=manager)

        _f.manager = manager
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
    if hasattr(env_fn, "manager") and env_fn.manager is not None:
        env_fn.manager.stop()


if __name__ == "__main__":
    main()
