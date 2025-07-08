import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

try:
    from envs.socket_env import SocketAppEnv
    from utils.intrinsic import BaseIntrinsicReward
except ModuleNotFoundError:
    SocketAppEnv = None
    class BaseIntrinsicReward:
        def reset(self):
            pass
        def compute(self, obs, env):
            return 0.0

from utils.intrinsic_registry import register_reward, get_reward, CompositeIntrinsicReward


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


def test_register_and_get_reward():
    class MyReward(BaseIntrinsicReward):
        def reset(self):
            pass
        def compute(self, obs, env):
            return 2.5

    register_reward("MyReward", MyReward)
    assert get_reward("MyReward") is MyReward


def test_env_builds_reward_from_registry():
    if SocketAppEnv is None:
        pytest.skip("gymnasium not installed")

    class RegReward(BaseIntrinsicReward):
        def reset(self):
            pass
        def compute(self, obs, env):
            return 3.0

    register_reward("RegReward", RegReward)

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
        server_manager=None,
        config=None,
        intrinsic_reward=None,
        intrinsic_name="RegReward",
    )
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert info["intrinsic"] == 3.0
    env.close()


def test_composite_intrinsic_reward():
    class R1(BaseIntrinsicReward):
        def reset(self):
            pass
        def compute(self, obs, env):
            return 1.0

    class R2(BaseIntrinsicReward):
        def reset(self):
            pass
        def compute(self, obs, env):
            return 2.0

    register_reward("R1", R1)
    register_reward("R2", R2)

    comp = CompositeIntrinsicReward(["R1", "R2"], latent_dim=1, device="cpu")
    comp.reset()
    val = comp.compute(np.zeros(1, dtype=np.float32), None)
    assert val == 3.0
