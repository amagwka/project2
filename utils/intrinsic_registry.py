from typing import Dict, Type, List

from .intrinsic import BaseIntrinsicReward, E3BIntrinsicReward
from .icm import ICMIntrinsicReward

# Registry mapping names to intrinsic reward classes
INTRINSIC_REWARDS: Dict[str, Type[BaseIntrinsicReward]] = {}


def register_reward(name: str, cls: Type[BaseIntrinsicReward]) -> None:
    """Register an intrinsic reward class under ``name``."""
    INTRINSIC_REWARDS[name] = cls


# Backwards compatibility
register_intrinsic = register_reward


def get_reward(name: str) -> Type[BaseIntrinsicReward]:
    """Retrieve a registered intrinsic reward class by ``name``."""
    if name not in INTRINSIC_REWARDS:
        raise KeyError(f"Unknown intrinsic reward: {name}")
    return INTRINSIC_REWARDS[name]


# Backwards compatibility
get_intrinsic = get_reward


class CompositeIntrinsicReward(BaseIntrinsicReward):
    """Combine multiple intrinsic rewards by summing their outputs."""

    def __init__(self, reward_names: List[str], latent_dim: int = 384, device: str = "cpu"):
        self.modules: List[BaseIntrinsicReward] = []
        for n in reward_names:
            cls = get_reward(n)
            try:
                inst = cls(latent_dim=latent_dim, device=device)
            except TypeError:
                inst = cls()
            self.modules.append(inst)

    def reset(self) -> None:
        for m in self.modules:
            m.reset()

    def compute(self, observation, env) -> float:
        return float(sum(m.compute(observation, env) for m in self.modules))


# Built-in registrations
register_intrinsic("E3BIntrinsicReward", E3BIntrinsicReward)
register_intrinsic("CompositeIntrinsicReward", CompositeIntrinsicReward)
register_intrinsic("ICMIntrinsicReward", ICMIntrinsicReward)

# Optionally register example curiosity modules
try:
    from examples.custom_curiosity import ConstantCuriosity

    register_intrinsic("ConstantCuriosity", ConstantCuriosity)
except Exception:
    pass
