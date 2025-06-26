from envs.socket_env import SocketAppEnv

def main():
    env = SocketAppEnv(max_steps=1000, device="cpu", start_servers=True, combined_server=True)
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
