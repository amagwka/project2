from __future__ import annotations

from .nats_base import NatsServer
from .tracker import RewardTracker


class DummyExternalTracker(RewardTracker):
    """Minimal in-game reward tracker placeholder."""

    def compute_reward(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class ExternalRewardServer(NatsServer):
    """Simple external reward server."""

    def __init__(self, tracker: RewardTracker | None = None,
                 subject: str = "rewards.in_game", url: str = "nats://127.0.0.1:4222"):
        super().__init__(subject, url)
        self.tracker = tracker or DummyExternalTracker()

    async def handle(self, data: bytes) -> bytes | None:
        cmd = data.decode().strip().upper()
        if cmd == "GET":
            r = self.tracker.compute_reward()
            return f"{r:.6f}".encode()
        if cmd == "RESET":
            self.tracker.reset()
            return b"OK"
        return b"ERR"


start_external_reward_server = ExternalRewardServer
