from abc import ABC, abstractmethod


class RewardTracker(ABC):
    """Base interface for extrinsic reward trackers."""

    @abstractmethod
    def compute_reward(self) -> float:
        """Return the accumulated extrinsic reward since the last call."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state used for computing rewards."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Clean up resources before shutting down."""
        raise NotImplementedError
