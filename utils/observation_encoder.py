from typing import Optional
import numpy as np

from utils.observations import LocalObs

class ObservationEncoder:
    """Wrapper for ``LocalObs`` providing DINO embeddings."""
    def __init__(self,
                 source: int = 1,
                 model_name: Optional[str] = "facebook/dinov2-with-registers-small",
                 device: str = "cpu",
                 embedding_dim: int = 384):
        self._obs = LocalObs(source=source,
                             mode="dino",
                             model_name=model_name,
                             device=device,
                             embedding_dim=embedding_dim)

    def get_embedding(self) -> np.ndarray:
        return self._obs.get_embedding()

    def close(self) -> None:
        self._obs.close()

