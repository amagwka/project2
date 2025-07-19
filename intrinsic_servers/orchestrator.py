from __future__ import annotations
from typing import Iterable, Tuple

from utils.intrinsic import BaseIntrinsicReward


class IntrinsicRewardOrchestrator:
    """Combine multiple intrinsic reward modules with optional weights."""

    def __init__(self, modules: Iterable[Tuple[BaseIntrinsicReward, float]]):
        self.modules = list(modules)

    def compute_all(self, observation, env=None) -> float:
        total = 0.0
        for module, weight in self.modules:
            total += module.compute(observation, env) * weight
        return float(total)

    def reset(self) -> None:
        for module, _ in self.modules:
            if hasattr(module, "reset"):
                module.reset()
