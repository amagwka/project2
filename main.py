import argparse

from config import get_config
from train import (
    toggle_pause,
    create_server_manager,
    create_environment,
    setup_models,
    run_sb3_training,
    train_loop,
)
from utils.hotkeys import start_listener


def main() -> None:
    parser = argparse.ArgumentParser(description="SocketApp PPO trainer")
    parser.add_argument(
        "--sb3",
        action="store_true",
        default=False,
        help="use stable-baselines3 PPO",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="training timesteps for sb3",
    )
    args = parser.parse_args()

    cfg = get_config()
    manager = create_server_manager(cfg)

    if args.sb3:
        run_sb3_training(cfg, timesteps=args.timesteps, manager=manager)
        if manager is not None:
            manager.stop()
        return

    env = create_environment(cfg, manager)
    actor, critic, opt_actor, opt_critic, buffer = setup_models(cfg)
    start_listener(toggle_pause)
    train_loop(cfg, env, actor, critic, opt_actor, opt_critic, buffer)


if __name__ == "__main__":
    main()
