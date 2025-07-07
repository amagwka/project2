"""Server utilities and reward tracker registry."""

from .base import UdpServer
from .reward_server import ExternalRewardTracker

try:
    from examples.constant_tracker import ConstantRewardTracker
except Exception:  # pragma: no cover - optional example module
    ConstantRewardTracker = None


REWARD_TRACKERS = {"external": ExternalRewardTracker}
if ConstantRewardTracker is not None:
    REWARD_TRACKERS["constant"] = ConstantRewardTracker

__all__ = ["UdpServer", "REWARD_TRACKERS", "ExternalRewardTracker"]

