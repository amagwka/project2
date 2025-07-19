from __future__ import annotations
import importlib
import numpy as np
from utils.intrinsic_registry import get_reward
from servers.nats_base import NatsServer


class IntrinsicServer(NatsServer):
    """NATS server wrapping an intrinsic reward module."""

    def __init__(self, reward_name: str, subject: str = "intrinsic", *, latent_dim: int = 384, device: str = "cpu", url: str = "nats://127.0.0.1:4222"):
        self.reward_name = reward_name
        cls = None
        try:
            cls = get_reward(reward_name)
        except KeyError:
            if "." in reward_name:
                mod_name, cls_name = reward_name.rsplit(".", 1)
                module = importlib.import_module(mod_name)
                cls = getattr(module, cls_name)
            else:
                raise
        try:
            self.reward = cls(latent_dim=latent_dim, device=device)
        except TypeError:
            self.reward = cls()
        super().__init__(subject, url)

    async def handle(self, data: bytes) -> bytes | None:
        if data == b"RESET":
            if hasattr(self.reward, "reset"):
                self.reward.reset()
            return b"OK"
        val = self.reward.compute(np.frombuffer(data, dtype=np.float32), None)
        return f"{float(val):.6f}".encode()

NatsIntrinsicServer = IntrinsicServer


def start_nats_intrinsic_server(reward_name: str, subject: str = "intrinsic", *, latent_dim: int = 384, device: str = "cpu", url: str = "nats://127.0.0.1:4222") -> None:
    """Convenience helper for ``NatsIntrinsicServer``."""

    server = NatsIntrinsicServer(
        reward_name,
        subject=subject,
        latent_dim=latent_dim,
        device=device,
        url=url,
    )
    server.serve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the NATS intrinsic reward server.")
    parser.add_argument("--name", default="E3BIntrinsicReward", help="Registered reward class name or module path")
    parser.add_argument("--subject", default="intrinsic", help="NATS subject")
    parser.add_argument("--url", default="nats://127.0.0.1:4222", help="NATS server URL")
    parser.add_argument("--latent-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

    start_nats_intrinsic_server(
        reward_name=args.name,
        subject=args.subject,
        latent_dim=args.latent_dim,
        device=args.device,
        url=args.url,
    )
