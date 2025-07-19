import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env

from config import Config
from envs.nats_env import create_nats_env, NatsAppEnv
from servers.manager import ServerManager
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


def create_server_manager(cfg: Config) -> Optional[ServerManager]:
    """Return a ``ServerManager`` if the config requests one."""
    return ServerManager() if cfg.env.start_servers else None


def create_environment(cfg: Config, manager: Optional[ServerManager]) -> NatsAppEnv:
    """Instantiate ``NatsAppEnv`` using ``manager``."""
    return create_nats_env(cfg.env, server_manager=manager)


def create_vec_env(cfg: Config, manager: Optional[ServerManager]):
    """Create a vectorized environment for Stable Baselines."""
    env_fn = lambda: create_nats_env(cfg.env, server_manager=manager)
    return make_vec_env(env_fn, n_envs=1)


def setup_sb3_model(cfg: Config, vec_env) -> SB3PPO:
    """Initialize the Stable Baselines3 PPO model."""
    return SB3PPO(
        "MlpPolicy",
        vec_env,
        n_steps=cfg.training.rollout_len,
        batch_size=cfg.training.rollout_len,
        learning_rate=cfg.training.learning_rate,
        verbose=1,
    )


def setup_models(cfg: Config) -> Tuple[Actor, Q_Critic, optim.Optimizer, optim.Optimizer, RolloutBufferNoDone]:
    """Create actor/critic networks, optimizers and rollout buffer."""
    device = cfg.training.device
    state_dim = cfg.training.state_dim
    action_dim = cfg.training.action_dim

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
    return actor, critic, optim_actor, optim_critic, buffer


def run_sb3_training(cfg: Config, timesteps: int = 5000, manager: Optional[ServerManager] = None) -> None:
    """Run PPO training using Stable Baselines3."""
    vec_env = create_vec_env(cfg, manager)
    model = setup_sb3_model(cfg, vec_env)
    model.learn(total_timesteps=timesteps)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print("Learned action:", action)
    vec_env.close()


def train_loop(
    cfg: Config,
    env: NatsAppEnv,
    actor: Actor,
    critic: Q_Critic,
    optim_actor: optim.Optimizer,
    optim_critic: optim.Optimizer,
    buffer: RolloutBufferNoDone,
) -> None:
    """Main training loop for the custom PPO implementation."""
    global paused

    device = cfg.training.device
    action_dim = cfg.training.action_dim
    obs, _ = env.reset()

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


def run_training(cfg: Config, use_sb3: bool = False, timesteps: int = 5000) -> None:
    """Execute PPO training using either SB3 or the custom implementation."""
    manager = create_server_manager(cfg)

    if use_sb3:
        run_sb3_training(cfg, timesteps=timesteps, manager=manager)
        if manager is not None:
            manager.stop()
        return

    env = create_environment(cfg, manager)
    actor, critic, optim_actor, optim_critic, buffer = setup_models(cfg)
    train_loop(cfg, env, actor, critic, optim_actor, optim_critic, buffer)

