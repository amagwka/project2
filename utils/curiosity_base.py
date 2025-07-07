class CuriosityReward:
    """Interface for intrinsic reward modules."""

    def reset(self) -> None:  # pragma: no cover - interface
        """Reset the internal state if any."""
        raise NotImplementedError

    def compute(self, obs, env):  # pragma: no cover - interface
        """Return intrinsic reward for observation in the given environment."""
        raise NotImplementedError
