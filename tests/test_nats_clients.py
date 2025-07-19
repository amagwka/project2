import sys
import subprocess
import asyncio
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.nats_client import NatsActionClient, NatsWorldModelClient
from servers.nats_base import NatsServer
from tests.utils import get_free_port


class DummyRewardServer(NatsServer):
    def __init__(self, url: str):
        super().__init__("actions", url)
        self.actions = []

    async def handle(self, data: bytes) -> bytes | None:
        msg = data.decode().strip().upper()
        if msg == "GET":
            return b"1.5"
        if msg == "RESET":
            return b"OK"
        self.actions.append(msg)
        return b"DONE"


def test_nats_client_send_action_and_get_reward():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server = DummyRewardServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsActionClient(url)
        client.send_action(3)
        reward = client.get_reward()
        client.send_reset()
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert server.actions == ["3"]
    assert abs(reward - 1.5) < 1e-6


class DummyModelServer(NatsServer):
    def __init__(self, url: str):
        super().__init__("world_model", url)

    async def handle(self, data: bytes) -> bytes | None:
        arr = np.frombuffer(data, dtype=np.float32)
        return (arr + 1).astype(np.float32).tobytes()


def test_world_model_client_predict():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server = DummyModelServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsWorldModelClient(url)
        obs = np.zeros((2, 3), dtype=np.float32)
        pred = client.predict(obs)
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert np.allclose(pred, obs.flatten() + 1)
