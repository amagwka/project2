import numpy as np
import torch

from .nats_base import NatsServer
from models.world_model import LSTMWorldModel


class WorldModelRewardServer(NatsServer):
    """Compute intrinsic reward from world model prediction error."""

    def __init__(self, model_path: str, *, obs_dim: int = 384, seq_len: int = 30,
                 subject: str = "rewards.world_model", device: str = "cpu",
                 url: str = "nats://127.0.0.1:4222"):
        super().__init__(subject, url)
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.device = device
        self.model = LSTMWorldModel(obs_dim=obs_dim).to(device)
        if model_path:
            try:
                state = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state)
            except Exception:
                pass
        self.model.eval()

    async def handle(self, data: bytes) -> bytes | None:
        if data == b"RESET":
            return b"OK"
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size != (self.seq_len + 1) * self.obs_dim:
            return b"ERR"
        obs_seq = torch.from_numpy(arr[: self.seq_len * self.obs_dim]
                                    .reshape(self.seq_len, self.obs_dim))
        next_obs = torch.from_numpy(arr[self.seq_len * self.obs_dim :])
        obs_seq = obs_seq.unsqueeze(0).to(self.device)
        next_obs = next_obs.to(self.device)
        with torch.no_grad():
            pred = self.model(obs_seq).squeeze(0)
        mse = torch.mean((pred - next_obs) ** 2).item()
        return f"{mse:.6f}".encode()


start_world_model_reward_server = WorldModelRewardServer
