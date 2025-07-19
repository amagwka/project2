"""Server utilities and reward tracker registry."""

from .base import UdpServer
from .nats_base import NatsServer
from .reward_server import (
    ExternalRewardTracker,
    start_nats_reward_server,
)
from .intrinsic_server import (
    IntrinsicServer,
    start_udp_intrinsic_server,
    start_nats_intrinsic_server,
)
from .action_server import (
    ActionRewardServer,
    start_combined_udp_server,
    start_nats_combined_server,
)
from .world_model_server import start_nats_world_model_server

try:
    from examples.constant_tracker import ConstantRewardTracker
except Exception:  # pragma: no cover - optional example module
    ConstantRewardTracker = None


REWARD_TRACKERS = {"external": ExternalRewardTracker}
if ConstantRewardTracker is not None:
    REWARD_TRACKERS["constant"] = ConstantRewardTracker

__all__ = [
    "UdpServer",
    "NatsServer",
    "REWARD_TRACKERS",
    "ExternalRewardTracker",
    "IntrinsicServer",
    "start_udp_intrinsic_server",
    "start_nats_intrinsic_server",
    "start_nats_reward_server",
    "ActionRewardServer",
    "start_combined_udp_server",
    "start_nats_combined_server",
    "start_nats_world_model_server",
]

