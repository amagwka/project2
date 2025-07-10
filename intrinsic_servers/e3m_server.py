from __future__ import annotations
import numpy as np
from utils.intrinsic import E3BIntrinsicReward
from .base import BaseUDPIntrinsicServer, IntrinsicInput
from .registry import register


@register("e3b")
class E3MServer(BaseUDPIntrinsicServer):
    """Intrinsic server wrapping :class:`E3BIntrinsicReward`."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5008, *, latent_dim: int = 384, device: str = "cpu"):
        self.module = E3BIntrinsicReward(latent_dim=latent_dim, device=device)
        super().__init__(host, port, latent_dim=latent_dim, device=device)

    def reset(self) -> None:
        self.module.reset()

    def compute(self, inp: IntrinsicInput) -> float:
        return float(self.module.compute(np.asarray(inp.observation), None))
