"""Utility helpers for logging and other features."""

try:
    from . import logger  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    logger = None

