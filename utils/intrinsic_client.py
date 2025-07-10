import socket
from typing import Tuple
import numpy as np


class IntrinsicClient:
    """UDP client for an intrinsic reward server."""

    def __init__(self, addr: Tuple[str, int], timeout: float = 0.2):
        self.addr = addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def __enter__(self) -> "IntrinsicClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def compute(self, obs: np.ndarray) -> float:
        self.sock.sendto(obs.astype(np.float32).tobytes(), self.addr)
        data, _ = self.sock.recvfrom(64)
        return float(data.decode().strip())

    def send_reset(self) -> None:
        try:
            self.sock.sendto(b"RESET", self.addr)
            self.sock.recvfrom(32)
        except Exception:
            pass

    def close(self) -> None:
        self.sock.close()
