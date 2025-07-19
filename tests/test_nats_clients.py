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


class DummyActionServer(PatternNatsServer):
    def __init__(self, url: str):
        super().__init__("actions.>", url)
        self.actions = []

    async def handle(self, subject: str, data: bytes) -> bytes | None:
        self.actions.append(data.decode())
        return b"OK"


class DummyRewardServer(NatsServer):
    def __init__(self, url: str):
        super().__init__("rewards.in_game", url)

    async def handle(self, data: bytes) -> bytes | None:
        msg = data.decode().strip().upper()
        if msg == "GET":
            return b"1.5"
        if msg == "RESET":
            return b"OK"
        return b"ERR"


def test_nats_client_send_action_and_get_reward():
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
        client = NatsActionClient(url, queue=1)
        client.send_action(3)
        reward = client.get_reward()
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
    assert abs(reward - 1.5) < 1e-6


class DummyModelServer(NatsServer):
    def __init__(self, url: str):
        super().__init__("rewards.world_model", url)

    async def handle(self, data: bytes) -> bytes | None:
        return b"0.5"


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
        r = client.compute(obs, np.ones(3, dtype=np.float32))
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

    assert abs(r - 0.5) < 1e-6
