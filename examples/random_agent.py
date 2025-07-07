from envs.socket_env import create_socket_env
from config import get_config

def main():
    cfg = get_config()
    env = create_socket_env(cfg.env)
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"[Step {step:02d}] action={action} extrinsic={info['extrinsic']:.3f} "
            f"intrinsic={info['intrinsic']:.3f} total={reward:.3f}"
        )
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
