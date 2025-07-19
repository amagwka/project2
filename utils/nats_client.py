import asyncio
import numpy as np
from nats.aio.client import Client as NATS


class NatsClient:
    """Simple synchronous wrapper around a NATS client."""

    def __init__(self, url: str = "nats://127.0.0.1:4222", timeout: float = 0.5):
        self.url = url
        self.timeout = timeout
        self.nc = NATS()
        self.loop = asyncio.new_event_loop()
        self.connected = False

    def _connect(self) -> None:
        if not self.connected:
            self.loop.run_until_complete(self.nc.connect(servers=[self.url]))
            self.connected = True

    def request(self, subject: str, data: bytes = b"") -> bytes:
        self._connect()
        msg = self.loop.run_until_complete(
            self.nc.request(subject, data, timeout=self.timeout)
        )
        return msg.data

    def close(self) -> None:
        if self.connected:
            self.loop.run_until_complete(self.nc.drain())
            self.loop.close()
            self.connected = False


class NatsActionClient(NatsClient):
    """Client for actions and rewards over NATS."""

    def send_action(self, action_idx: int) -> None:
        self.request("actions", str(action_idx).encode())

    def get_reward(self) -> float:
        data = self.request("actions", b"GET")
        try:
            return float(data.decode())
        except Exception:
            return 0.0

    def send_reset(self) -> None:
        self.request("actions", b"RESET")


class NatsWorldModelClient(NatsClient):
    """Client for world model predictions over NATS."""

    def predict(self, obs_sequence):
        data = obs_sequence.astype("float32").tobytes()
        reply = self.request("world_model", data)
        return np.frombuffer(reply, dtype=np.float32)


class NatsIntrinsicClient(NatsClient):
    """Client for intrinsic reward over NATS."""

    def compute(self, obs, action=None) -> float:
        arr = obs.astype("float32")
        if action is not None:
            arr = np.concatenate((np.array([action], dtype=np.float32), arr))
        data = self.request("intrinsic", arr.tobytes())
        try:
            return float(data.decode())
        except Exception:
            return 0.0

    def send_reset(self) -> None:
        self.request("intrinsic", b"RESET")
