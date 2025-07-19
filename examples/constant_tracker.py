"""Minimal RewardTracker returning a constant value."""

from servers.tracker import RewardTracker


class ConstantRewardTracker(RewardTracker):
    """Tracker that always returns the same reward."""

    def __init__(self, value: float = 1.0):
        self.value = float(value)

    def compute_reward(self) -> float:
        return self.value

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


if __name__ == "__main__":
    from servers.reward_server import start_nats_reward_server

    tracker = ConstantRewardTracker(0.5)
    start_nats_reward_server(tracker)
