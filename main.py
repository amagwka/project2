import socket
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from time import sleep
from threading import Thread
from pynput import keyboard
from torch.utils.tensorboard import SummaryWriter

from models.nn import Actor, Q_Critic
from utils.rollout import RolloutBufferNoDone, compute_gae
from models.ppo import ppo_update
from utils.observations import LocalObs
from utils.intrinsic import E3BIntrinsicReward
from utils.shrink import LogStencilMemory


def send_action_to_server(action_idx: int, host: str = "127.0.0.1", port: int = 5005) -> None:
    """Send a single action index to the UDP server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        message = str(action_idx).encode()
        sock.sendto(message, (host, port))
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        sock.close()


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
    obs = LocalObs(source=1, mode="dino", model_name="facebook/dinov2-with-registers-small", device=DEVICE)
    e3b = E3BIntrinsicReward(latent_dim=STATE_DIM, decay=1, ridge=0.1, device=DEVICE)
    memo = LogStencilMemory(max_len=1000, steps=SEQ_LEN, feature_dim=STATE_DIM)

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

        emb_np = obs.get_embedding()
        memo.add(emb_np[None, :])

        seq = memo.shrinked()
        emb_seq = torch.from_numpy(seq).float().to(DEVICE).unsqueeze(0)

        ir = e3b.compute(emb_np)

        logits = actor(emb_seq)
        dist = td.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        logp = dist.log_prob(action)
        act_onehot = F.one_hot(action, ACTION_DIM).float()

        send_action_to_server(action.item())
        sleep(0.05)
        act_onehot = act_onehot.unsqueeze(0)
        extrinsic = 0.0
        total_r = extrinsic + ir
        writer.add_scalar("Reward/Total", total_r, step_count)

        value = critic(emb_seq, act_onehot).squeeze()
        buffer.add(emb_seq[0].cpu(), act_onehot.cpu(), total_r, value.cpu(), logp.cpu())

        step_count += 1
        writer.add_scalar("Reward/Total", total_r, step_count)

        if buffer.ready() and step_count % 256 == 0:
            s, a, r, v, lp = buffer.get()
            returns, adv = compute_gae(r, v)
            ppo_update(actor, critic, optim_actor, optim_critic, s, a, lp, returns, adv)
            print(f"[PPO Update] Step {step_count}")


if __name__ == "__main__":
    main()
