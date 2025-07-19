import sys
import numpy as np
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

try:
    from envs.nats_env import NatsAppEnv
    from servers.manager import ServerManager
    from utils.intrinsic import BaseIntrinsicReward
except ModuleNotFoundError:
    NatsAppEnv = None
    ServerManager = None
    class BaseIntrinsicReward:
        def reset(self):
            pass
        def compute(self, obs, env):
            return 0.0



class DummyNatsClient:
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


class DummyIntrinsic(BaseIntrinsicReward):
    def __init__(self):
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def compute(self, obs, env) -> float:
        return 42.0


def test_nats_env_basic():
    if NatsAppEnv is None:
        pytest.skip("gymnasium not installed")
    udp = DummyNatsClient(reward=1.0)
    obs_enc = DummyObsEncoder()
    env = NatsAppEnv(
        max_steps=1,
        device="cpu",
        embedding_model=None,
        enable_logging=False,
        start_servers=False,
        use_world_model=False,
        action_client=udp,
        obs_encoder=obs_enc,
        server_manager=None,
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


def test_nats_env_custom_intrinsic():
    if NatsAppEnv is None:
        pytest.skip("gymnasium not installed")
    udp = DummyNatsClient(reward=1.0)
    obs_enc = DummyObsEncoder()
    intrinsic = DummyIntrinsic()
    env = NatsAppEnv(
        max_steps=1,
        device="cpu",
        embedding_model=None,
        enable_logging=False,
        start_servers=False,
        use_world_model=False,
        action_client=udp,
        obs_encoder=obs_enc,
        server_manager=None,
        intrinsic_reward=intrinsic,
    )
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert info["intrinsic"] == 42.0
    assert intrinsic.reset_called
    env.close()


def test_nats_env_config_intrinsic():
    if NatsAppEnv is None:
        pytest.skip("gymnasium not installed")
    udp = DummyNatsClient(reward=1.0)
    obs_enc = DummyObsEncoder()
    from config import get_config
    cfg = get_config()
    cfg.env.intrinsic_names = ["examples.custom_curiosity.ConstantCuriosity"]
    env = NatsAppEnv(
        max_steps=1,
        device="cpu",
        embedding_model=None,
        enable_logging=False,
        start_servers=False,
        use_world_model=False,
        action_client=udp,
        obs_encoder=obs_enc,
        server_manager=None,
        config=cfg.env,
    )
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert info["intrinsic"] == 1.0
    env.close()


def test_no_server_spawn_when_disabled(monkeypatch):
    if NatsAppEnv is None:
        pytest.skip("gymnasium not installed")

    popen_calls = []

    def fake_popen(*args, **kwargs):
        popen_calls.append(args)
        class Dummy:
            def terminate(self):
                pass
            def wait(self, timeout=None):
                pass
        return Dummy()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    udp = DummyNatsClient()
    obs_enc = DummyObsEncoder()
    mgr = ServerManager()
    env = NatsAppEnv(
        max_steps=1,
        device="cpu",
        embedding_model=None,
        enable_logging=False,
        start_servers=False,
        use_world_model=False,
        action_client=udp,
        obs_encoder=obs_enc,
        server_manager=mgr,
    )

    assert popen_calls == []
    env.close()
