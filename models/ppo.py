import torch
import torch.nn.functional as F
import torch.distributions as td

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_EPS      = 0.2

def ppo_update(actor, critic, optim_actor, optim_critic,
               buf_states, buf_actions, buf_logp_old, buf_returns, buf_adv):
    """Perform a PPO update and return training metrics."""
    batch = len(buf_states)
    idx = torch.randperm(batch)

    metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "kl_div": 0.0,
        "num_batches": 0,
    }

    for _ in range(1):  # epochs
        for start in range(0, batch, 256):
            sl = idx[start:start+256]

            # Directly use the state sequences from the buffer
            state_seq = buf_states[sl].to(DEVICE)  # shape: [batch_size, seq_len, state_dim]
            actions    = buf_actions[sl].to(DEVICE)
            logp_old   = buf_logp_old[sl].to(DEVICE)
            returns    = buf_returns[sl].to(DEVICE)
            adv        = buf_adv[sl].to(DEVICE)

            # Actor forward pass
            logits = actor(state_seq)  # [batch_size, action_dim]
            dist = td.Categorical(logits=logits)
            logp = dist.log_prob(actions.argmax(-1))  # [batch_size]

            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            approx_kl = (logp_old - logp).mean()

            # Critic forward pass
            q_pred = critic(state_seq, actions)  # critic expects same seq
            critic_loss = F.mse_loss(q_pred, returns)

            loss = actor_loss + 0.5 * critic_loss

            optim_actor.zero_grad()
            optim_critic.zero_grad()
            loss.backward()
            optim_actor.step()
            optim_critic.step()

            metrics["actor_loss"] += actor_loss.item()
            metrics["critic_loss"] += critic_loss.item()
            metrics["kl_div"] += approx_kl.item()
            metrics["num_batches"] += 1

    n = max(1, metrics.pop("num_batches"))
    for k in metrics:
        metrics[k] /= n
    return metrics
