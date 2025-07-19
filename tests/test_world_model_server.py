import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from tests.utils import get_free_port
from utils.nats_client import NatsWorldModelClient
from servers.world_model_server import NatsWorldModelServer


class EchoWorldModelServer(NatsWorldModelServer):
    def __init__(self, url: str):
        super().__init__(model_path="", subject="world_model", obs_dim=4, seq_len=2, device="cpu", model_type="mlp", url=url)

    async def handle(self, data: bytes) -> bytes | None:
        arr = np.frombuffer(data, dtype=np.float32)
        return (arr + 1).astype(np.float32).tobytes()


def test_world_model_server_predict():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server = EchoWorldModelServer(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    try:
        client = NatsWorldModelClient(url)
        arr = np.zeros((2, 4), dtype=np.float32)
        pred = client.predict(arr)
        assert pred.size == 8
        client.close()
    finally:
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)

