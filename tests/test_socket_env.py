import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

try:
    from envs.socket_env import SocketAppEnv
except ModuleNotFoundError:
    SocketAppEnv = None



class DummyUdpClient:
    def __init__(self, reward: float = 1.0):
        self.reward = reward
        self.actions = []

    def send_action(self, action_idx: int) -> None:
        self.actions.append(action_idx)

    def send_reset(self) -> None:
        pass

    def get_reward(self) -> float:
        return self.reward

    def close(self) -> None:
        pass


class DummyObsEncoder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_embedding(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)

    def close(self):
        pass


def test_socket_env_basic():
    if SocketAppEnv is None:
        pytest.skip("gymnasium not installed")
    udp = DummyUdpClient(reward=1.0)
    obs_enc = DummyObsEncoder()
    env = SocketAppEnv(
        max_steps=1,
        device="cpu",
        embedding_model=None,
        combined_server=False,
        enable_logging=False,
        start_servers=False,
        use_world_model=False,
        udp_client=udp,
        obs_encoder=obs_enc,
    )
    obs, _ = env.reset()
    assert hasattr(obs, "shape")
    assert len(obs) == env.state_dim

    obs, reward, terminated, truncated, _ = env.step(0)
    assert isinstance(reward, float)
    assert reward > 0
    assert not terminated
    assert truncated

    env.close()

