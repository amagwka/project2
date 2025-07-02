import subprocess
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def get_free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "servers.world_model_server",
        "--model-path",
        "",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--obs-dim",
        "4",
        "--seq-len",
        "2",
        "--device",
        "cpu",
        "--model-type",
        "mlp",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_world_model_server_ping_and_predict():
    port = get_free_udp_port()
    proc = start_server(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    addr = ("127.0.0.1", port)

    try:
        # Wait for server to start
        for _ in range(20):
            try:
                sock.sendto(b"PING", addr)
                data, _ = sock.recvfrom(16)
                if data == b"PONG":
                    break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail("no PONG reply")

        arr = np.zeros((2, 4), dtype=np.float32)
        sock.sendto(arr.tobytes(), addr)
        data, _ = sock.recvfrom(64)
        pred = np.frombuffer(data, dtype=np.float32)
        assert pred.size == 4
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        sock.close()
