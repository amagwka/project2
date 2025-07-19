from __future__ import annotations

import numpy as np
from utils.intrinsic import E3BIntrinsicReward
from .nats_base import NatsServer


class E3MRewardServer(NatsServer):
    """NATS server computing E3M intrinsic reward."""

    def __init__(self, subject: str = "rewards.e3m", *, latent_dim: int = 384,
                 device: str = "cpu", url: str = "nats://127.0.0.1:4222"):
        super().__init__(subject, url)
        self.reward = E3BIntrinsicReward(latent_dim=latent_dim, device=device)

    async def handle(self, data: bytes) -> bytes | None:
        if data == b"RESET":
            self.reward.reset()
            return b"OK"
        arr = np.frombuffer(data, dtype=np.float32)
        val = float(self.reward.compute(arr, None))
        return f"{val:.6f}".encode()


start_e3m_reward_server = E3MRewardServer
