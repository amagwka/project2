from envs.socket_env import SocketAppEnv
from config import get_config

def main():
    cfg = get_config()
    env = SocketAppEnv(
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
        world_model_time=cfg.env.world_model.time_interval,
    )
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
