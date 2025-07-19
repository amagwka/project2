import sys
import subprocess
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.nats_client import NatsActionClient, NatsWorldModelClient
from servers.nats_base import NatsServer, PatternNatsServer
from tests.utils import get_free_port


class DummyRewardServer(NatsServer):
    def __init__(self, url: str):
        super().__init__("rewards.in_game", url)

    async def handle(self, data: bytes) -> bytes | None:
        m = data.decode().strip().upper()
        if m == "GET":
            return b"2.0"
        if m == "RESET":
            return b"OK"
        return b"ERR"


class DummyActionServer(PatternNatsServer):
    def __init__(self, url: str):
        super().__init__("actions.>", url)
        self.actions = []

    async def handle(self, subject: str, data: bytes) -> bytes | None:
        self.actions.append(data.decode())
        return b"OK"


def test_nats_client_basic_roundtrip():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reward_server = DummyRewardServer(url)
    action_server = DummyActionServer(url)
    rt = threading.Thread(target=reward_server.serve, daemon=True)
    at = threading.Thread(target=action_server.serve, daemon=True)
    rt.start()
    at.start()

    try:
        client = NatsActionClient(url)
        client.send_action(5)
        r = client.get_reward()
        client.send_reset()
        client.close()
    finally:
        reward_server.shutdown()
        action_server.shutdown()
        rt.join(timeout=1)
        at.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert action_server.actions != []
    assert abs(r - 2.0) < 1e-6


def test_world_model_client_predict_roundtrip():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    class EchoServer(NatsServer):
        def __init__(self, url: str):
            super().__init__("rewards.world_model", url)

        async def handle(self, data: bytes) -> bytes | None:
            return b"1.2"

    server = EchoServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsWorldModelClient(url)
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r = client.compute(obs, obs)
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert abs(r - 1.2) < 1e-6
