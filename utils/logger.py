from typing import Any, Dict, Iterable

import numpy as np
from torch.utils.tensorboard import SummaryWriter

_writer = SummaryWriter()


def log_scalar(tag: str, value: float, step: int) -> None:
    """Log a scalar value to TensorBoard."""
    _writer.add_scalar(tag, float(value), step)


def log_histogram(tag: str, values: Iterable[float], step: int) -> None:
    """Log a histogram of values to TensorBoard."""
    arr = np.asarray(list(values), dtype=np.float32)
    _writer.add_histogram(tag, arr, step)


def log_action_bins(tag: str, values: Iterable[float], step: int) -> None:
    """Log per-action probabilities as individual scalar bins."""
    arr = np.asarray(list(values), dtype=np.float32)
    for idx, val in enumerate(arr):
        _writer.add_scalar(f"{tag}/{idx}", float(val), step)


def log_dict(metrics: Dict[str, Any], step: int) -> None:
    """Log multiple scalar metrics from a dictionary."""
    for k, v in metrics.items():
        try:
            val = float(v)
        except Exception:
            continue
        _writer.add_scalar(k, val, step)


def get_writer() -> SummaryWriter:
    """Expose the underlying ``SummaryWriter`` for advanced usage."""
    return _writer
