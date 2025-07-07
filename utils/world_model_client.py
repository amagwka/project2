import socket
from typing import Tuple
import numpy as np

class WorldModelClient:
    """UDP client that queries a world model server for predictions."""
    def __init__(self, addr: Tuple[str, int], timeout: float = 0.2):
        self.addr = addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def __enter__(self) -> "WorldModelClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def predict(self, obs_sequence: np.ndarray) -> np.ndarray:
        """Send an observation sequence and return the predicted next embedding."""
        self.sock.sendto(obs_sequence.tobytes(), self.addr)
        data, _ = self.sock.recvfrom(65535)
        return np.frombuffer(data, dtype=np.float32)

    def close(self) -> None:
        self.sock.close()

