import numpy as np
from utils.intrinsic_registry import get_reward
from .base import UdpServer


class IntrinsicServer(UdpServer):
    """UDP server wrapping an intrinsic reward module."""

    def __init__(self, reward_name: str, host: str = "0.0.0.0", port: int = 5008,
                 latent_dim: int = 384, device: str = "cpu"):
        self.reward_name = reward_name
        cls = get_reward(reward_name)
        try:
            self.reward = cls(latent_dim=latent_dim, device=device)
        except TypeError:
            self.reward = cls()
        super().__init__(host, port)

    def handle(self, data: bytes, addr):
        if data == b"RESET":
            self.reward.reset()
            return b"OK"
        if data == b"PING":
            return b"PONG"
        obs = np.frombuffer(data, dtype=np.float32)
        val = self.reward.compute(obs, None)
        return f"{float(val):.6f}"


def start_udp_intrinsic_server(reward_name: str, host: str = "0.0.0.0",
                               port: int = 5008, latent_dim: int = 384,
                               device: str = "cpu") -> None:
    """Convenience helper for ``IntrinsicServer``."""
    server = IntrinsicServer(reward_name, host=host, port=port,
                             latent_dim=latent_dim, device=device)
    server.serve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the UDP intrinsic reward server.")
    parser.add_argument("--name", default="E3BIntrinsicReward",
                        help="Registered reward class name")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind")
    parser.add_argument("--port", type=int, default=5008, help="UDP port to listen on")
    parser.add_argument("--latent-dim", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

    start_udp_intrinsic_server(
        reward_name=args.name,
        host=args.host,
        port=args.port,
        latent_dim=args.latent_dim,
        device=args.device,
    )
