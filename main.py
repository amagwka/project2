import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from time import sleep
from threading import Thread
from pynput import keyboard
from utils import logger

from envs.socket_env import SocketAppEnv
from models.nn import Actor, Q_Critic
from utils.rollout import RolloutBufferNoDone, compute_gae
from models.ppo import ppo_update


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE_DIM, ACTION_DIM, SEQ_LEN = 384, 7, 70

paused = False


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
    Thread(target=hotkey_listener, daemon=True).start()

    env = SocketAppEnv(device=DEVICE, combined_server=True,
                       start_servers=True,use_world_model=True, enable_logging=True)
    obs, _ = env.reset()

    actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)
    critic = Q_Critic(shared_lstm=actor.lstm, action_dim=ACTION_DIM).to(DEVICE)
    optim_actor = optim.Adam(actor.parameters(), lr=3e-4)
    optim_critic = optim.Adam(critic.parameters(), lr=3e-4)
    buffer = RolloutBufferNoDone(2048, STATE_DIM, ACTION_DIM, "cpu")

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
        logger.log_histogram("Action/Probs", dist.probs.squeeze(0).detach().cpu().numpy(), step_count)

        value = critic(emb_seq, act_onehot.to(DEVICE)).squeeze().detach()
        logp_detached = logp.detach()
        buffer.add(state_tensor, act_onehot.cpu(), reward, value.cpu(), logp_detached.cpu())
        if terminated or truncated:
            obs, _ = env.reset()

        step_count += 1
        logger.log_scalar("Reward/Total", reward, step_count)

        if buffer.ready() and step_count % 256 == 0:
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
