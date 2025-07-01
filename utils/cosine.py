import numpy as np


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Return 1 - cosine similarity between two vectors."""
    u = np.asarray(u, dtype=np.float32).ravel()
    v = np.asarray(v, dtype=np.float32).ravel()
    if u.size == 0 or v.size == 0:
        return 0.0
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    cos_sim = float(np.dot(u, v) / (norm_u * norm_v))
    return 1.0 - cos_sim
