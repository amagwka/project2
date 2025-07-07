import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env
from time import sleep
from threading import Thread
from pynput import keyboard
from utils import logger
from config import get_config

from envs.socket_env import SocketAppEnv
from models.nn import Actor, Q_Critic
from utils.rollout import RolloutBufferNoDone, compute_gae
from models.ppo import ppo_update


cfg = get_config()

DEVICE = cfg.training.device
STATE_DIM = cfg.training.state_dim
ACTION_DIM = cfg.training.action_dim
SEQ_LEN = cfg.training.seq_len

paused = True


def hotkey_listener() -> None:
    def on_press(key):
        global paused
        if key == keyboard.Key.f9:
            paused = not paused
            sleep(0.1)
            print(f"[Pause toggled] Paused = {paused}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def main() -> None:
    global paused
    parser = argparse.ArgumentParser(description="SocketApp PPO trainer")
    parser.add_argument("--sb3", action="store_true",default=False,
                        help="use stable-baselines3 PPO")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="training timesteps for sb3")
    args = parser.parse_args()

    if args.sb3:
        env_fn = lambda: SocketAppEnv(
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
        vec_env = make_vec_env(env_fn, n_envs=1)
        model = SB3PPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.training.rollout_len,
            batch_size=cfg.training.rollout_len,
            learning_rate=cfg.training.learning_rate,
            verbose=1,
        )
        model.learn(total_timesteps=args.timesteps)
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print("Learned action:", action)
        vec_env.close()
        return

    Thread(target=hotkey_listener, daemon=True).start()

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
    )
    obs, _ = env.reset()

    actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)
    critic = Q_Critic(shared_lstm=actor.lstm, action_dim=ACTION_DIM).to(DEVICE)
    optim_actor = optim.Adam(actor.parameters(), lr=cfg.training.learning_rate)
    optim_critic = optim.Adam(critic.parameters(), lr=cfg.training.learning_rate)
    buffer = RolloutBufferNoDone(
        cfg.training.rollout_len,
        STATE_DIM,
        ACTION_DIM,
        "cpu",
    )

    step_count = 0
    print("[Started in PAUSED mode] Press F9 to toggle")

    while True:
        if paused:
            continue

        state_tensor = torch.from_numpy(obs).float()
        emb_seq = buffer.get_latest_state_seq(state_tensor).to(DEVICE)

        logits = actor(emb_seq)
        dist = td.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        logp = dist.log_prob(action)
        act_onehot = F.one_hot(action, ACTION_DIM).float()

        obs, reward, terminated, truncated, _ = env.step(action.item())
        act_onehot = act_onehot.unsqueeze(0)
        logger.log_scalar("Reward/Total", reward, step_count)
        action_probs = dist.probs.squeeze(0).detach().cpu().numpy()
        logger.log_histogram("Action/Probs", action_probs, step_count)
        logger.log_action_bins("Action/Bins", action_probs, step_count)

        value = critic(emb_seq, act_onehot.to(DEVICE)).squeeze().detach()
        logp_detached = logp.detach()
        buffer.add(state_tensor, act_onehot.cpu(), reward, value.cpu(), logp_detached.cpu())
        if terminated or truncated:
            obs, _ = env.reset()

        step_count += 1
        logger.log_scalar("Reward/Total", reward, step_count)

        if buffer.ready() and step_count % cfg.training.update_every == 0:
            s, a, r, v, lp = buffer.get()
            returns, adv = compute_gae(r, v)
            metrics = ppo_update(actor, critic, optim_actor, optim_critic,
                                 s, a, lp, returns, adv)
            logger.log_dict({
                "Loss/Actor": metrics.get("actor_loss", 0.0),
                "Loss/Critic": metrics.get("critic_loss", 0.0),
                "KLDiv": metrics.get("kl_div", 0.0),
            }, step_count)
            print(f"[PPO Update] Step {step_count}")


if __name__ == "__main__":
    main()
