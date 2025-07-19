"""Environment package exports.

This module lazily imports optional environments so that the package can be used
without requiring all optional dependencies (e.g. ``cv2``).  Tests that only
need the bandit environment should still run even if other heavy dependencies
are missing.
"""

try:
    from .nats_env import NatsAppEnv  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    NatsAppEnv = None  # type: ignore

from .bandit_env import MultiArmedBanditEnv

__all__ = [
    "NatsAppEnv",
    "MultiArmedBanditEnv",
]
