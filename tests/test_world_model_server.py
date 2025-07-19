import subprocess
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from tests.utils import get_free_port
from utils.nats_client import NatsWorldModelClient
from servers.world_model_reward_server import WorldModelRewardServer


class EchoWorldModelServer(WorldModelRewardServer):
    def __init__(self, url: str):
        super().__init__(model_path="", subject="rewards.world_model", obs_dim=3, seq_len=2, device="cpu", url=url)

    async def handle(self, data: bytes) -> bytes | None:
        return b"0.7"


def test_world_model_reward():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server = EchoWorldModelServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsWorldModelClient(url)
        obs = np.zeros((2, 3), dtype=np.float32)
        r = client.compute(obs, np.ones(3, dtype=np.float32))
        assert abs(r - 0.7) < 1e-6
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)
