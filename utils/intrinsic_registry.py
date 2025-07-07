from typing import Dict, Type

from .intrinsic import BaseIntrinsicReward, E3BIntrinsicReward

INTRINSIC_REWARDS: Dict[str, Type[BaseIntrinsicReward]] = {}


def register_intrinsic(name: str, cls: Type[BaseIntrinsicReward]) -> None:
    """Register an intrinsic reward class by name."""
    INTRINSIC_REWARDS[name] = cls


def get_intrinsic(name: str) -> Type[BaseIntrinsicReward]:
    """Retrieve a registered intrinsic reward class."""
    if name not in INTRINSIC_REWARDS:
        raise KeyError(f"Unknown intrinsic reward: {name}")
    return INTRINSIC_REWARDS[name]


# Built-in registrations
register_intrinsic("E3BIntrinsicReward", E3BIntrinsicReward)

# Optionally register example curiosity modules
try:
    from examples.custom_curiosity import ConstantCuriosity

    register_intrinsic("ConstantCuriosity", ConstantCuriosity)
except Exception:
    pass
