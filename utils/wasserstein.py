import numpy as np


def wasserstein_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Approximate 1D Wasserstein distance between two vectors."""
    u = np.asarray(u, dtype=np.float32).ravel()
    v = np.asarray(v, dtype=np.float32).ravel()
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    u_cdf = np.cumsum(u_sorted)
    v_cdf = np.cumsum(v_sorted)
    return float(np.mean(np.abs(u_cdf - v_cdf)))
