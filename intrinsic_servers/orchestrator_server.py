from __future__ import annotations
import importlib
import asyncio
from typing import Optional, Callable, Sequence
from nats.aio.client import Client as NATS
import numpy as np
from utils.intrinsic_registry import get_reward
from utils.intrinsic import BaseIntrinsicReward


class NatsServer:
    """Simple NATS request/reply server."""

    def __init__(self, subject: str, url: str = "nats://127.0.0.1:4222"):
        self.subject = subject
        self.url = url
        self.loop = asyncio.new_event_loop()
        self.nc = NATS()
        self._shutdown = asyncio.Event()
        print(f"[NatsServer] Listening on {subject} at {url}")

    async def handle(self, data: bytes) -> Optional[bytes]:
        raise NotImplementedError

    async def _cb(self, msg):
        reply = await self.handle(msg.data)
        if isinstance(reply, str):
            reply = reply.encode()
        if reply is not None and msg.reply:
            await self.nc.publish(msg.reply, reply)

    async def _serve(self, cleanup: Optional[Callable[[], None]] = None):
        await self.nc.connect(servers=[self.url])
        await self.nc.subscribe(self.subject, cb=self._cb)
        await self._shutdown.wait()
        await self.nc.drain()
        if cleanup:
            cleanup()

    def serve(self, cleanup: Optional[Callable[[], None]] = None) -> None:
        try:
            self.loop.run_until_complete(self._serve(cleanup))
        except KeyboardInterrupt:
            pass
        finally:
            if not self.loop.is_closed():
                self.loop.close()

    def shutdown(self) -> None:
        self.loop.call_soon_threadsafe(self._shutdown.set)


class NatsIntrinsicOrchestrator(NatsServer):
    """NATS server combining multiple intrinsic rewards."""

    def __init__(self, reward_names: Sequence[str], subject: str = "intrinsic", *,
                 latent_dim: int = 384, device: str = "cpu",
                 url: str = "nats://127.0.0.1:4222"):
        super().__init__(subject, url)
        self.modules: list[BaseIntrinsicReward] = []
        for name in reward_names:
            cls = None
            try:
                cls = get_reward(name)
            except KeyError:
                if "." in name:
                    mod_name, cls_name = name.rsplit(".", 1)
                    module = importlib.import_module(mod_name)
                    cls = getattr(module, cls_name)
                else:
                    raise
            try:
                inst = cls(latent_dim=latent_dim, device=device)
            except TypeError:
                inst = cls()
            self.modules.append(inst)

    async def handle(self, data: bytes) -> bytes | None:
        if data == b"RESET":
            for m in self.modules:
                if hasattr(m, "reset"):
                    m.reset()
            return b"OK"
        arr = np.frombuffer(data, dtype=np.float32)
        total = 0.0
        for m in self.modules:
            total += float(m.compute(arr, None))
        return f"{total:.6f}".encode()


NatsIntrinsicOrchestratorServer = NatsIntrinsicOrchestrator


def start_nats_orchestrator_server(reward_names: Sequence[str], subject: str = "intrinsic", *,
                                   latent_dim: int = 384, device: str = "cpu",
                                   url: str = "nats://127.0.0.1:4222") -> None:
    server = NatsIntrinsicOrchestratorServer(
        reward_names,
        subject=subject,
        latent_dim=latent_dim,
        device=device,
        url=url,
    )
    server.serve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the NATS intrinsic orchestrator server")
    parser.add_argument("names", nargs="+", help="Intrinsic reward class names or module paths")
    parser.add_argument("--subject", default="intrinsic", help="NATS subject")
    parser.add_argument("--url", default="nats://127.0.0.1:4222", help="NATS server URL")
    parser.add_argument("--latent-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

    start_nats_orchestrator_server(
        args.names,
        subject=args.subject,
        latent_dim=args.latent_dim,
        device=args.device,
        url=args.url,
    )
