from __future__ import annotations

from servers.intrinsic_server import IntrinsicServer


class ICMNatsServer(IntrinsicServer):
    """Convenience NATS server for ``ICMIntrinsicReward``."""

    def __init__(self, subject: str = "intrinsic", *, latent_dim: int = 384,
                 action_dim: int = 7, device: str = "cpu", url: str = "nats://127.0.0.1:4222"):
        name = "ICMIntrinsicReward"
        super().__init__(name, subject=subject, latent_dim=latent_dim,
                         device=device, url=url)
        self.action_dim = action_dim


def start_icm_server(subject: str = "intrinsic", *, latent_dim: int = 384,
                     action_dim: int = 7, device: str = "cpu",
                     url: str = "nats://127.0.0.1:4222") -> None:
    server = ICMNatsServer(subject=subject, latent_dim=latent_dim,
                           action_dim=action_dim, device=device, url=url)
    server.serve()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the ICM intrinsic reward server")
    parser.add_argument("--subject", default="intrinsic", help="NATS subject")
    parser.add_argument("--url", default="nats://127.0.0.1:4222", help="NATS server URL")
    parser.add_argument("--latent-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()
    start_icm_server(subject=args.subject, latent_dim=args.latent_dim,
                     action_dim=args.action_dim, device=args.device, url=args.url)
