import subprocess
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from servers.e3m_reward_server import E3MRewardServer
from utils.nats_client import NatsIntrinsicClient
from tests.utils import get_free_port


def test_intrinsic_server_compute_and_reset():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    server = E3MRewardServer(subject='rewards.e3m', latent_dim=4, device='cpu', url=url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    client = NatsIntrinsicClient(url)
    try:
        arr = np.zeros(4, dtype=np.float32)
        val = client.compute(arr)
        assert val == 1.0
        client.send_reset()
    finally:
        client.close()
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)


