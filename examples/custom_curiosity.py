from utils.curiosity_base import CuriosityReward

class ConstantCuriosity(CuriosityReward):
    """Simple curiosity module returning a constant bonus."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def reset(self) -> None:
        pass

    def compute(self, obs, env):
        return float(self.value)
