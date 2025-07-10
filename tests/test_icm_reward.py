import numpy as np

try:
    from utils.icm import ICMIntrinsicReward
except Exception:  # pragma: no cover - torch missing
    ICMIntrinsicReward = None


class DummyEnv:
    def __init__(self):
        self.last_action = None


def test_icm_basic_prediction():
    if ICMIntrinsicReward is None:
        return
    env = DummyEnv()
    icm = ICMIntrinsicReward(obs_dim=4, action_dim=3, device="cpu")
    env.last_action = 0
    r1 = icm.compute(np.zeros(4, dtype=np.float32), env)
    assert r1 == 0.0
    env.last_action = 1
    r2 = icm.compute(np.ones(4, dtype=np.float32), env)
    assert isinstance(r2, float)
    assert icm.predicted_action is not None

