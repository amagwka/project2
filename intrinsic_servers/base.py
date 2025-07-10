from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from servers.base import UdpServer


@dataclass
class IntrinsicInput:
    observation: np.ndarray
    action: Optional[int] = None
    step_id: Optional[int] = None
    env_time: Optional[float] = None
    done: Optional[bool] = False


class BaseUDPIntrinsicServer(UdpServer):
    """Base class for UDP intrinsic reward servers."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0, *, latent_dim: int = 384, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = device
        super().__init__(host, port)

    # ------------------------------------------------------------------
    # Methods to override
    # ------------------------------------------------------------------
    def compute(self, inp: IntrinsicInput) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # UDP handler
    # ------------------------------------------------------------------
    def handle(self, data: bytes, addr):
        if data == b"RESET":
            self.reset()
            return b"OK"
        if data == b"PING":
            return b"PONG"
        obs = np.frombuffer(data, dtype=np.float32)
        inp = IntrinsicInput(observation=obs)
        val = float(self.compute(inp))
        return f"{val:.6f}"
