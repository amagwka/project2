from typing import Protocol
import numpy as np
import gymnasium as gym


class IntrinsicReward(Protocol):
    """Protocol for intrinsic reward modules."""

    def reset(self) -> None:
        """Reset the internal state if any."""

    def compute(self, observation: np.ndarray, env: gym.Env) -> float:
        """Return intrinsic reward for an observation in the given environment."""


class CuriosityReward:
    """Interface for intrinsic reward modules."""

    def reset(self) -> None:  # pragma: no cover - interface
        """Reset the internal state if any."""
        raise NotImplementedError

    def compute(self, obs, env):  # pragma: no cover - interface
        """Return intrinsic reward for observation in the given environment."""
        raise NotImplementedError
