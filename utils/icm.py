from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .intrinsic import BaseIntrinsicReward


class ICMIntrinsicReward(BaseIntrinsicReward):
    """Intrinsic Curiosity Module computing prediction error."""

    def __init__(self, obs_dim: int = 384, action_dim: int = 7, hidden_dim: int = 256, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        ).to(device)
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)
        self.reset()

    def reset(self) -> None:
        self.prev_obs: np.ndarray | None = None
        self.predicted_action: int | None = None

    def compute_pair(self, prev_obs: np.ndarray, obs: np.ndarray, action: int) -> float:
        o1 = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.tensor(action, dtype=torch.long, device=self.device)
        a_one = F.one_hot(a, num_classes=self.action_dim).float()
        pred_obs = self.forward_model(torch.cat([o1, a_one], dim=0))
        forward_err = F.mse_loss(pred_obs, o2, reduction="mean")
        logits = self.inverse_model(torch.cat([o1, o2], dim=0))
        self.predicted_action = int(torch.argmax(logits).item())
        return forward_err.item()

    def compute(self, observation: np.ndarray, env) -> float:
        action = getattr(env, "last_action", None)
        if self.prev_obs is None or action is None:
            self.prev_obs = np.asarray(observation)
            return 0.0
        reward = self.compute_pair(self.prev_obs, observation, int(action))
        self.prev_obs = np.asarray(observation)
        return reward

