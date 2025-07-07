import sys
import socket
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.udp_client import UdpClient
from utils.world_model_client import WorldModelClient


def _get_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_udp_client_basic_roundtrip():
    action_port = _get_port()
    reward_port = _get_port()

    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_sock.bind(("127.0.0.1", action_port))
    action_sock.settimeout(0.1)

    reward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reward_sock.bind(("127.0.0.1", reward_port))
    reward_sock.settimeout(0.1)

    received = []
    stop = threading.Event()

    def action_server():
        while not stop.is_set():
            try:
                data, addr = action_sock.recvfrom(64)
            except socket.timeout:
                continue
            received.append(data.decode().strip())

    def reward_server():
        while not stop.is_set():
            try:
                data, addr = reward_sock.recvfrom(64)
            except socket.timeout:
                continue
            msg = data.decode().strip().upper()
            if msg == "GET":
                reward_sock.sendto(b"2.0", addr)
            elif msg == "RESET":
                reward_sock.sendto(b"OK", addr)

    t1 = threading.Thread(target=action_server, daemon=True)
    t2 = threading.Thread(target=reward_server, daemon=True)
    t1.start()
    t2.start()

    with UdpClient(("127.0.0.1", action_port), ("127.0.0.1", reward_port)) as client:
        client.send_action(5)
        r = client.get_reward()
        client.send_reset()

    stop.set()
    t1.join(timeout=1)
    t2.join(timeout=1)
    action_sock.close()
    reward_sock.close()

    assert received == ["5"]
    assert abs(r - 2.0) < 1e-6


def test_world_model_client_predict_roundtrip():
    port = _get_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.1)
    stop = threading.Event()

    def server():
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            arr = np.frombuffer(data, dtype=np.float32)
            sock.sendto((arr + 2).astype(np.float32).tobytes(), addr)

    t = threading.Thread(target=server, daemon=True)
    t.start()

    with WorldModelClient(("127.0.0.1", port)) as client:
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pred = client.predict(obs)

    stop.set()
    t.join(timeout=1)
    sock.close()

    assert np.allclose(pred, obs + 2)
