import socket
import numpy as np
import torch
from models.world_model import LSTMWorldModel
from lab.rnn_baseline import RNNPredictor
from lab.mlp_world_model import MLPWorldModel


DEFAULT_MODEL_PATH = "lab/scripts/rnn_lstm.pt"


def _infer_rnn_config(state: dict[str, torch.Tensor], gates: int) -> tuple[int, int]:
    """Infer ``hidden_dim`` and ``num_layers`` from an RNN state dict."""
    weight_keys = [k for k in state if k.endswith("weight_ih_l0")]
    if not weight_keys:
        return 512, 2
    key = weight_keys[0]
    hidden_dim = state[key].shape[0] // gates
    layers = 0
    while f"rnn.weight_ih_l{layers}" in state or f"lstm.weight_ih_l{layers}" in state:
        layers += 1
    return hidden_dim, max(1, layers)


def _infer_mlp_config(state: dict[str, torch.Tensor], obs_dim: int) -> tuple[int, int]:
    """Infer ``hidden_dim`` and ``num_layers`` from an MLP state dict."""
    linears = [k for k in state if k.startswith("net") and k.endswith(".weight")]
    linears.sort()
    num_layers = len(linears)
    if not linears:
        return obs_dim, 1
    first = state[linears[0]]
    hidden_dim = first.shape[0]
    return hidden_dim, max(1, num_layers)


from typing import Optional


def start_udp_world_model_server(model_path: str = DEFAULT_MODEL_PATH, host: str = '0.0.0.0',
                                 port: int = 5007, obs_dim: int = 384,
                                 seq_len: int = 30, device: str = 'cuda',
                                 model_type: Optional[str] = 'auto'):
    """Start a UDP server that predicts the next observation embedding."""
    state = None
    if model_path:
        try:
            state = torch.load(model_path, map_location=device)
        except Exception as e:
            print(f"[WorldModelServer] Failed to load {model_path}: {e}")
            state = None

    if model_type is None or model_type == 'auto':
        if state is not None:
            if any(k.startswith('net.') for k in state):
                model_type = 'mlp'
            else:
                key = next((k for k in state if 'weight_ih_l0' in k), None)
                if key and state[key].shape[0] % 4 == 0:
                    model_type = 'lstm'
                elif key and state[key].shape[0] % 3 == 0:
                    model_type = 'gru'
                else:
                    model_type = 'lstm'
        else:
            model_type = 'lstm'
    model_type = model_type.lower()

    # Infer hidden dim and layer count when a checkpoint was loaded
    hidden_dim = 512
    num_layers = 2 if model_type == 'mlp' else 3
    if state is not None:
        if model_type == 'mlp':
            hidden_dim, num_layers = _infer_mlp_config(state, obs_dim)
        elif model_type == 'gru':
            hidden_dim, num_layers = _infer_rnn_config(state, 3)
        else:
            hidden_dim, num_layers = _infer_rnn_config(state, 4)

    if model_type == 'mlp':
        model = MLPWorldModel(input_dim=obs_dim, hidden_dim=hidden_dim,
                              num_layers=num_layers).to(device)
    elif model_type == 'gru':
        model = RNNPredictor(input_dim=obs_dim, hidden_dim=hidden_dim,
                             num_layers=num_layers, rnn_type='GRU').to(device)
    else:
        model = LSTMWorldModel(obs_dim=obs_dim, hidden_dim=hidden_dim,
                               num_layers=num_layers).to(device)

    if state is not None:
        try:
            model.load_state_dict(state)
            print(f"[WorldModelServer] Loaded model from {model_path}")
        except RuntimeError:
            renamed = {}
            for k, v in state.items():
                if k.startswith("lstm."):
                    renamed["rnn." + k[5:]] = v
                else:
                    renamed[k] = v
            model.load_state_dict(renamed)
            print(f"[WorldModelServer] Loaded model from {model_path} (renamed)")
    model.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow quick rebinding in case a previous instance recently exited
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"[WorldModelServer] Failed to bind udp://{host}:{port}: {e}")
        print("[WorldModelServer] Falling back to an ephemeral port.")
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    print(f"[WorldModelServer] Listening on udp://{host}:{port}")

    try:
        while True:
            data, addr = sock.recvfrom(65535)
            if not data:
                continue
            if data == b'PING':
                sock.sendto(b'PONG', addr)
                continue

            arr = np.frombuffer(data, dtype=np.float32)
            if arr.size != seq_len * obs_dim:
                sock.sendto(b'ERR', addr)
                continue
            obs_seq = torch.from_numpy(arr.reshape(1, seq_len, obs_dim)).to(device)
            with torch.no_grad():
                pred = model(obs_seq).cpu().numpy().astype(np.float32)
            sock.sendto(pred.tobytes(), addr)
    except KeyboardInterrupt:
        print('\n[WorldModelServer] Shutdown.')
    finally:
        sock.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start the UDP world model server.')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Path to the world model checkpoint')
    parser.add_argument('--host', default='0.0.0.0', help='Interface to bind')
    parser.add_argument('--port', type=int, default=5007, help='UDP port to listen on')
    parser.add_argument('--obs-dim', type=int, default=384, help='Dimension of observation embeddings')
    parser.add_argument('--seq-len', type=int, default=30, help='Length of the observation sequence')
    parser.add_argument('--device', default='cuda', help='Torch device')
    parser.add_argument('--model-type', default='auto', choices=['lstm', 'gru', 'mlp', 'auto'],
                        help='Type of world model to load or "auto" to infer from the checkpoint')
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
