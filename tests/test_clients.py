import sys
import socket
import threading
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.udp_client import UdpClient
from utils.world_model_client import WorldModelClient
from tests.utils import get_free_udp_port


def test_udp_client_send_action_and_get_reward():
    action_port = get_free_udp_port()
    reward_port = get_free_udp_port()

    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_sock.bind(("127.0.0.1", action_port))
    action_sock.settimeout(0.1)

    reward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reward_sock.bind(("127.0.0.1", reward_port))
    reward_sock.settimeout(0.1)

    actions = []
    stop_event = threading.Event()

    def action_server():
        while not stop_event.is_set():
            try:
                data, addr = action_sock.recvfrom(64)
            except socket.timeout:
                continue
            actions.append(data.decode().strip())

    def reward_server():
        while not stop_event.is_set():
            try:
                data, addr = reward_sock.recvfrom(64)
            except socket.timeout:
                continue
            msg = data.decode().strip().upper()
            if msg == "GET":
                reward_sock.sendto(b"1.5", addr)
            elif msg == "RESET":
                reward_sock.sendto(b"OK", addr)

    t1 = threading.Thread(target=action_server, daemon=True)
    t2 = threading.Thread(target=reward_server, daemon=True)
    t1.start()
    t2.start()

    client = UdpClient(("127.0.0.1", action_port), ("127.0.0.1", reward_port))
    client.send_action(3)
    reward = client.get_reward()
    client.send_reset()

    stop_event.set()
    t1.join(timeout=1)
    t2.join(timeout=1)
    client.close()
    action_sock.close()
    reward_sock.close()

    assert actions == ["3"]
    assert abs(reward - 1.5) < 1e-6


def test_world_model_client_predict():
    port = get_free_udp_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.1)

    stop_event = threading.Event()

    def model_server():
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            arr = np.frombuffer(data, dtype=np.float32)
            reply = (arr + 1).astype(np.float32)
            sock.sendto(reply.tobytes(), addr)

    t = threading.Thread(target=model_server, daemon=True)
    t.start()

    client = WorldModelClient(("127.0.0.1", port))
    obs = np.zeros((2, 3), dtype=np.float32)
    pred = client.predict(obs)

    stop_event.set()
    t.join(timeout=1)
    client.close()
    sock.close()

    assert np.allclose(pred, obs.flatten() + 1)
