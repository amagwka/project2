from __future__ import annotations
from typing import Iterable, Tuple

from .base import BaseUDPIntrinsicServer, IntrinsicInput


class IntrinsicRewardOrchestrator:
    """Combine multiple intrinsic servers with optional weights."""

    def __init__(self, modules: Iterable[Tuple[BaseUDPIntrinsicServer, float]]):
        self.modules = list(modules)

    def compute_all(self, inp: IntrinsicInput) -> float:
        total = 0.0
        for module, weight in self.modules:
            total += module.compute(inp) * weight
        return float(total)
