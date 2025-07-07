import argparse

from config import get_config
from train import run_training, toggle_pause
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
    if not args.sb3:
        start_listener(toggle_pause)
    run_training(cfg, use_sb3=args.sb3, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
