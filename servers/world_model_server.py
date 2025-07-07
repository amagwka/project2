import numpy as np
import torch
from models.world_model import LSTMWorldModel
from lab.rnn_baseline import RNNPredictor
from lab.mlp_world_model import MLPWorldModel
from servers.base import UdpServer

MODEL_TYPES = {
    "lstm": LSTMWorldModel,
    "gru": RNNPredictor,
    "mlp": MLPWorldModel,
}


DEFAULT_MODEL_PATH = "lab/scripts/rnn_lstm.pt"


class WorldModelServer(UdpServer):
    """UDP server serving predictions from a trained world model."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        host: str = "0.0.0.0",
        port: int = 5007,
        obs_dim: int = 384,
        seq_len: int = 30,
        device: str = "cuda",
        model_type: str = "lstm",
    ):
        self.model_type = model_type.lower()
        model_cls = MODEL_TYPES.get(self.model_type, LSTMWorldModel)
        if self.model_type == "gru":
            self.model = model_cls(input_dim=obs_dim, rnn_type="GRU").to(device)
        elif self.model_type == "mlp":
            self.model = model_cls(input_dim=obs_dim, num_layers=3).to(device)
        else:
            self.model = model_cls(obs_dim=obs_dim).to(device)
        if model_path:
            try:
                state = torch.load(model_path, map_location=device)
                try:
                    self.model.load_state_dict(state)
                except RuntimeError:
                    renamed = {}
                    for k, v in state.items():
                        if k.startswith("lstm."):
                            renamed["rnn." + k[5:]] = v
                        else:
                            renamed[k] = v
                    self.model.load_state_dict(renamed)

                print(f"[WorldModelServer] Loaded model from {model_path}")
            except Exception as e:
                print(f"[WorldModelServer] Failed to load {model_path}: {e}")
        self.model.eval()

        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.device = device
        super().__init__(host, port)

    def handle(self, data: bytes, addr):
        if data == b"PING":
            return b"PONG"

        arr = np.frombuffer(data, dtype=np.float32).copy()
        if arr.size != self.seq_len * self.obs_dim:
            return b"ERR"
        obs_seq = torch.from_numpy(arr.reshape(self.seq_len, self.obs_dim)).to(self.device)
        with torch.no_grad():
            if self.model_type == "mlp":
                inp = obs_seq[-1].unsqueeze(0)
            else:
                inp = obs_seq.unsqueeze(0)
            pred = self.model(inp).squeeze(0).cpu().numpy().astype(np.float32)
        pred /= 10.0
        return pred.tobytes()


def start_udp_world_model_server(
    model_path: str = DEFAULT_MODEL_PATH,
    host: str = "0.0.0.0",
    port: int = 5007,
    obs_dim: int = 384,
    seq_len: int = 30,
    device: str = "cuda",
    model_type: str = "lstm",
):
    """Helper function launching ``WorldModelServer``."""

    server = WorldModelServer(
        model_path=model_path,
        host=host,
        port=port,
        obs_dim=obs_dim,
        seq_len=seq_len,
        device=device,
        model_type=model_type,
    )
    server.serve()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start the UDP world model server.')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Path to the world model checkpoint')
    parser.add_argument('--host', default='0.0.0.0', help='Interface to bind')
    parser.add_argument('--port', type=int, default=5007, help='UDP port to listen on')
    parser.add_argument('--obs-dim', type=int, default=384, help='Dimension of observation embeddings')
    parser.add_argument('--seq-len', type=int, default=30, help='Length of the observation sequence')
    parser.add_argument('--device', default='cuda', help='Torch device')
    parser.add_argument('--model-type', default='lstm', choices=['lstm', 'gru', 'mlp'],
                        help='Type of world model to load')
    args = parser.parse_args()

    start_udp_world_model_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        obs_dim=args.obs_dim,
        seq_len=args.seq_len,
        device=args.device,
        model_type=args.model_type,
    )
