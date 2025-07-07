import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env

from config import Config
from envs.socket_env import create_socket_env
from models.nn import Actor, Q_Critic
from models.ppo import ppo_update
from utils.rollout import RolloutBufferNoDone, compute_gae
from utils import logger


paused = True


def toggle_pause() -> None:
    """Toggle the global paused flag and print its new state."""
    global paused
    paused = not paused
    print(f"[Pause toggled] Paused = {paused}")


def run_training(cfg: Config, use_sb3: bool = False, timesteps: int = 5000) -> None:
    """Execute PPO training."""
    global paused

    if use_sb3:
        env_fn = lambda: create_socket_env(cfg.env)
        vec_env = make_vec_env(env_fn, n_envs=1)
        model = SB3PPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.training.rollout_len,
            batch_size=cfg.training.rollout_len,
            learning_rate=cfg.training.learning_rate,
            verbose=1,
        )
        model.learn(total_timesteps=timesteps)
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print("Learned action:", action)
        vec_env.close()
        return

    device = cfg.training.device
    state_dim = cfg.training.state_dim
    action_dim = cfg.training.action_dim

    env = create_socket_env(cfg.env)
    obs, _ = env.reset()

    actor = Actor(state_dim=state_dim, action_dim=action_dim).to(device)
    critic = Q_Critic(shared_lstm=actor.lstm, action_dim=action_dim).to(device)
    optim_actor = optim.Adam(actor.parameters(), lr=cfg.training.learning_rate)
    optim_critic = optim.Adam(critic.parameters(), lr=cfg.training.learning_rate)
    buffer = RolloutBufferNoDone(
        cfg.training.rollout_len,
        state_dim,
        action_dim,
        "cpu",
    )

    step_count = 0
    print("[Started in PAUSED mode] Press F9 to toggle")

    while True:
        if paused:
            time.sleep(0.05)
            continue

        state_tensor = torch.from_numpy(obs).float()
        emb_seq = buffer.get_latest_state_seq(state_tensor).to(device)

        logits = actor(emb_seq)
        dist = td.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        logp = dist.log_prob(action)
        act_onehot = F.one_hot(action, action_dim).float()

        obs, reward, terminated, truncated, _ = env.step(action.item())
        act_onehot = act_onehot.unsqueeze(0)
        logger.log_scalar("Reward/Total", reward, step_count)
        action_probs = dist.probs.squeeze(0).detach().cpu().numpy()
        logger.log_histogram("Action/Probs", action_probs, step_count)
        logger.log_action_bins("Action/Bins", action_probs, step_count)

        value = critic(emb_seq, act_onehot.to(device)).squeeze().detach()
        logp_detached = logp.detach()
        buffer.add(state_tensor, act_onehot.cpu(), reward, value.cpu(), logp_detached.cpu())
        if terminated or truncated:
            obs, _ = env.reset()

        step_count += 1
        logger.log_scalar("Reward/Total", reward, step_count)

        if buffer.ready() and step_count % cfg.training.update_every == 0:
            s, a, r, v, lp = buffer.get()
            returns, adv = compute_gae(r, v)
            metrics = ppo_update(actor, critic, optim_actor, optim_critic, s, a, lp, returns, adv)
            logger.log_dict(
                {
                    "Loss/Actor": metrics.get("actor_loss", 0.0),
                    "Loss/Critic": metrics.get("critic_loss", 0.0),
                    "KLDiv": metrics.get("kl_div", 0.0),
                },
                step_count,
            )
            print(f"[PPO Update] Step {step_count}")

