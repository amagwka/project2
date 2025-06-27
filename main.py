import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from time import sleep
from threading import Thread
from pynput import keyboard
from torch.utils.tensorboard import SummaryWriter

from envs.socket_env import SocketAppEnv
from models.nn import Actor, Q_Critic
from utils.rollout import RolloutBufferNoDone, compute_gae
from models.ppo import ppo_update


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE_DIM, ACTION_DIM, SEQ_LEN = 384, 7, 70

# Global pause flag toggled via hotkey
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

    writer = SummaryWriter(log_dir="runs/ppo_run")
    env = SocketAppEnv(device=DEVICE, combined_server=True, start_servers=False)
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
        writer.add_scalar("Reward/Total", reward, step_count)

        value = critic(emb_seq, act_onehot.to(DEVICE)).squeeze()
        buffer.add(state_tensor, act_onehot.cpu(), reward, value.cpu(), logp.cpu())
        if terminated or truncated:
            obs, _ = env.reset()

        step_count += 1
        writer.add_scalar("Reward/Total", reward, step_count)

        if buffer.ready() and step_count % 256 == 0:
            s, a, r, v, lp = buffer.get()
            returns, adv = compute_gae(r, v)
            ppo_update(actor, critic, optim_actor, optim_critic, s, a, lp, returns, adv)
            print(f"[PPO Update] Step {step_count}")


if __name__ == "__main__":
    main()
