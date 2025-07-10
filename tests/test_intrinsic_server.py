import threading
import socket
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from servers.intrinsic_server import IntrinsicServer
from tests.utils import get_free_udp_port


def test_intrinsic_server_compute_and_reset():
    port = get_free_udp_port()
    server = IntrinsicServer(
        'examples.custom_curiosity.ConstantCuriosity',
        host='127.0.0.1',
        port=port,
        latent_dim=4,
        device='cpu'
    )
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    addr = ('127.0.0.1', port)
    try:
        arr = np.zeros(4, dtype=np.float32)
        sock.sendto(arr.tobytes(), addr)
        data, _ = sock.recvfrom(64)
        val = float(data.decode())
        assert val == 1.0
        sock.sendto(b'RESET', addr)
        data, _ = sock.recvfrom(64)
        assert data == b'OK'
    finally:
        server.shutdown()
        t.join(timeout=1)
        sock.close()


