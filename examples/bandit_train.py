import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.bandit_env import MultiArmedBanditEnv
from models.ppo import ppo_update
from utils.rollout import RolloutBufferNoDone, compute_gae
from utils import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit PPO trainer")
    parser.add_argument("--sb3", action="store_true",
                        help="use stable-baselines3 PPO")
    parser.add_argument("--timesteps", type=int, default=20000,
                        help="training timesteps for sb3")
    args = parser.parse_args()

    env = MultiArmedBanditEnv([0.43, 0.57])

    if args.sb3:
        vec_env = make_vec_env(lambda: MultiArmedBanditEnv([0.43, 0.57]), n_envs=1)
        model = SB3PPO("MlpPolicy", vec_env, n_steps=256, batch_size=256,
                       learning_rate=1e-4, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print("Learned action:", action)
        vec_env.close()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    buffer = RolloutBufferNoDone(512, state_dim, action_dim, device)

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(state_dim, action_dim)

        def forward(self, seq):
            x = seq[:, 0]
            return self.fc(x)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(state_dim + action_dim, 1)

        def forward(self, seq, act):
            x = torch.cat([seq[:, 0], act], dim=-1)
            return self.fc(x).squeeze(-1)

    actor = Actor().to(device)
    critic = Critic().to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=0.0001)
    opt_critic = optim.Adam(critic.parameters(), lr=0.0001)

    obs, _ = env.reset()
    step_count = 0
    for step in range(20000):
        s = torch.tensor(obs, dtype=torch.float32, device=device)
        seq = buffer.get_latest_state_seq(s)
        logits = actor(seq)
        dist = td.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        logp = dist.log_prob(action)
        onehot = nn.functional.one_hot(action, action_dim).float()

        obs, reward, _, _, _ = env.step(int(action))
        logger.log_scalar("Reward/Total", reward, step_count)
        action_probs = dist.probs.detach().cpu().numpy()
        logger.log_histogram("Action/Probs", action_probs, step_count)
        logger.log_action_bins("Action/Bins", action_probs, step_count)
        value = critic(seq, onehot.unsqueeze(0)).detach()
        buffer.add(s, onehot, reward, value, logp.detach())

        if buffer.ready():
            s_batch, a_batch, r_batch, v_batch, lp_batch = buffer.get()
            returns, adv = compute_gae(r_batch, v_batch)
            metrics = ppo_update(actor, critic, opt_actor, opt_critic,
                                 s_batch, a_batch, lp_batch, returns, adv)
            logger.log_dict({
                "Loss/Actor": metrics.get("actor_loss", 0.0),
                "Loss/Critic": metrics.get("critic_loss", 0.0),
                "KLDiv": metrics.get("kl_div", 0.0),
            }, step_count)
            #print(f"[PPO Update] Step {step_count}")

        step_count += 1
        logger.log_scalar("Reward/Total", reward, step_count)

    with torch.no_grad():
        seq = torch.zeros(1, buffer.seq_len, state_dim, device=device)
        logits = actor(seq)
        print("Learned probs:", logits.softmax(-1))


if __name__ == "__main__":
    main()
