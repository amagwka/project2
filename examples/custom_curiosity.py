"""Minimal custom curiosity module returning a constant bonus."""

from utils.intrinsic import BaseIntrinsicReward

class ConstantCuriosity(BaseIntrinsicReward):
    """Simple curiosity module returning a constant bonus."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def reset(self) -> None:
        pass

    def compute(self, observation, env) -> float:
        return float(self.value)
