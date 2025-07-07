import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.distributions as td
import torch.optim as optim

from envs.bandit_env import MultiArmedBanditEnv
from models.ppo import ppo_update
from utils.rollout import RolloutBufferNoDone, compute_gae


def test_bandit_env_ppo():
    torch.manual_seed(0)
    env = MultiArmedBanditEnv([0.1, 0.9])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(state_dim, action_dim)

        def forward(self, seq):
            return self.fc(seq[:, 0])

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(state_dim + action_dim, 1)

        def forward(self, seq, act):
            x = torch.cat([seq[:, 0], act], dim=-1)
            return self.fc(x).squeeze(-1)

    actor = Actor()
    critic = Critic()
    opt_a = optim.Adam(actor.parameters(), lr=0.0005)
    opt_c = optim.Adam(critic.parameters(), lr=0.0005)
    buf = RolloutBufferNoDone(32, state_dim, action_dim, device="cpu")

    obs, _ = env.reset()
    for _ in range(2000):
        s = torch.tensor(obs, dtype=torch.float32)
        seq = buf.get_latest_state_seq(s)
        logits = actor(seq)
        dist = td.Categorical(logits=logits.squeeze(0))
        act = dist.sample()
        logp = dist.log_prob(act)
        onehot = nn.functional.one_hot(act, action_dim).float()
        obs, rew, _, _, _ = env.step(int(act))
        val = critic(seq, onehot.unsqueeze(0)).detach()
        buf.add(s, onehot, rew, val, logp.detach())

        if buf.ready():
            s_b, a_b, r_b, v_b, lp_b = buf.get()
            returns, adv = compute_gae(r_b, v_b)
            ppo_update(
                actor,
                critic,
                opt_a,
                opt_c,
                s_b,
                a_b,
                lp_b,
                returns,
                adv,
                num_epochs=2,
                batch_size=16,
            )

    with torch.no_grad():
        seq = torch.zeros(1, buf.seq_len, state_dim)
        logits = actor(seq)
        print("Learned probs:", logits.softmax(-1))
        best = logits.argmax().item()
    assert best == 1
