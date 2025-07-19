from __future__ import annotations

from servers.intrinsic_server import IntrinsicServer


class E3BNatsServer(IntrinsicServer):
    """Convenience NATS server for ``E3BIntrinsicReward``."""

    def __init__(self, subject: str = "intrinsic", *, latent_dim: int = 384,
                 device: str = "cpu", url: str = "nats://127.0.0.1:4222"):
        super().__init__("E3BIntrinsicReward", subject=subject,
                         latent_dim=latent_dim, device=device, url=url)


def start_e3b_server(subject: str = "intrinsic", *, latent_dim: int = 384,
                     device: str = "cpu", url: str = "nats://127.0.0.1:4222") -> None:
    server = E3BNatsServer(subject=subject, latent_dim=latent_dim,
                           device=device, url=url)
    server.serve()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the E3B intrinsic reward server")
    parser.add_argument("--subject", default="intrinsic", help="NATS subject")
    parser.add_argument("--url", default="nats://127.0.0.1:4222", help="NATS server URL")
    parser.add_argument("--latent-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()
    start_e3b_server(subject=args.subject, latent_dim=args.latent_dim,
                     device=args.device, url=args.url)
