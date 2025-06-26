import threading
import socket
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

try:
    from envs.socket_env import SocketAppEnv
except ModuleNotFoundError:
    SocketAppEnv = None


def dummy_reward_server(host="127.0.0.1", port=9006, reward=1.0, stop_event=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.1)
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                continue
            msg = data.decode().strip().upper()
            if msg == "GET":
                sock.sendto(str(reward).encode(), addr)
            elif msg == "RESET":
                sock.sendto(b"OK", addr)
            else:
                sock.sendto(b"ERR", addr)
    finally:
        sock.close()


def test_socket_env_basic():
    if SocketAppEnv is None:
        pytest.skip("gymnasium not installed")
    stop = threading.Event()
    server_thread = threading.Thread(target=dummy_reward_server, kwargs={"stop_event": stop})
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.1)

    env = SocketAppEnv(max_steps=1, device="cpu", action_host="127.0.0.1", action_port=9005,
                       reward_host="127.0.0.1", reward_port=9006,
                       embedding_model=None, combined_server=False, enable_logging=False)
    obs, _ = env.reset()
    assert hasattr(obs, "shape")
    assert len(obs) == env.state_dim

    obs, reward, terminated, truncated, _ = env.step(0)
    assert isinstance(reward, float)
    assert reward > 0
    assert not terminated
    assert truncated

    env.close()
    stop.set()
    server_thread.join(timeout=1)
