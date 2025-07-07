from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

class IntrinsicReward(ABC):
    """Abstract base class for intrinsic reward modules."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state if any."""

    @abstractmethod
    def compute(self, observation: np.ndarray, env: gym.Env) -> float:
        """Return intrinsic reward for an observation in the given environment."""
