from __future__ import annotations
import numpy as np
from utils.icm import ICMIntrinsicReward
from .base import BaseUDPIntrinsicServer, IntrinsicInput
from .registry import register


@register("icm")
class ICMServer(BaseUDPIntrinsicServer):
    """UDP server providing ICM intrinsic rewards."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5009, *, latent_dim: int = 384, action_dim: int = 7, device: str = "cpu"):
        self.module = ICMIntrinsicReward(obs_dim=latent_dim, action_dim=action_dim, device=device)
        self.action_dim = action_dim
        self._prev_obs: np.ndarray | None = None
        self._prev_action: int | None = None
        super().__init__(host, port, latent_dim=latent_dim, device=device)

    def reset(self) -> None:
        self.module.reset()
        self._prev_obs = None
        self._prev_action = None

    def handle(self, data: bytes, addr):
        if data == b"RESET":
            self.reset()
            return b"OK"
        if data == b"PING":
            return b"PONG"
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size != self.latent_dim + 1:
            return b"ERR"
        action = int(arr[0])
        obs = arr[1:]
        inp = IntrinsicInput(observation=obs, action=action)
        val = float(self.compute(inp))
        return f"{val:.6f}"

    def compute(self, inp: IntrinsicInput) -> float:
        obs = np.asarray(inp.observation)
        action = int(inp.action if inp.action is not None else 0)
        if self._prev_obs is None:
            self._prev_obs = obs
            self._prev_action = action
            return 0.0
        reward = self.module.compute_pair(self._prev_obs, obs, action)
        self._prev_obs = obs
        self._prev_action = action
        return float(reward)

