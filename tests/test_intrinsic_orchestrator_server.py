import subprocess
import time
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from intrinsic_servers.orchestrator_server import NatsIntrinsicOrchestratorServer
from utils.nats_client import NatsIntrinsicClient
from tests.utils import get_free_port


def test_orchestrator_server_combines_rewards():
    port = get_free_port()
    url = f"nats://127.0.0.1:{port}"
    proc = subprocess.Popen(["nats-server", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.2)
    server = NatsIntrinsicOrchestratorServer([
        'examples.custom_curiosity.ConstantCuriosity',
        'examples.custom_curiosity.ConstantCuriosity',
    ], subject='intrinsic', latent_dim=4, device='cpu', url=url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    client = NatsIntrinsicClient(url)
    try:
        arr = np.zeros(4, dtype=np.float32)
        val = client.compute(arr)
        assert val == 2.0
        client.send_reset()
    finally:
        client.close()
        server.shutdown()
        t.join(timeout=1)
        proc.terminate()
        proc.wait(timeout=5)
