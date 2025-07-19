import sys
import subprocess
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
        m = data.decode().strip().upper()
        if m == "GET":
            return b"2.0"
        if m == "RESET":
            return b"OK"
        self.actions.append(m)
        return b"DONE"


def test_nats_client_basic_roundtrip():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server = DummyRewardServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsActionClient(url)
        client.send_action(5)
        r = client.get_reward()
        client.send_reset()
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert server.actions == ["5"]
    assert abs(r - 2.0) < 1e-6


def test_world_model_client_predict_roundtrip():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    class EchoServer(NatsServer):
        def __init__(self, url: str):
            super().__init__("world_model", url)

        async def handle(self, data: bytes) -> bytes | None:
            arr = np.frombuffer(data, dtype=np.float32)
            return (arr + 2).astype(np.float32).tobytes()

    server = EchoServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsWorldModelClient(url)
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pred = client.predict(obs)
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert np.allclose(pred, obs + 2)
