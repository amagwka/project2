import socket
import numpy as np
import torch
from models.world_model import LSTMWorldModel


DEFAULT_MODEL_PATH = "lab/scripts/rnn_lstm.pt"


def start_udp_world_model_server(model_path: str = DEFAULT_MODEL_PATH, host: str = '0.0.0.0',
                                 port: int = 5007, obs_dim: int = 384,
                                 seq_len: int = 30, device: str = 'cpu'):
    """Start a UDP server that predicts the next observation embedding."""
    model = LSTMWorldModel(obs_dim=obs_dim).to(device)
    if model_path:
        try:
            state = torch.load(model_path, map_location=device)
            try:
                model.load_state_dict(state)
            except RuntimeError:
                # Older checkpoints may use ``lstm`` as the module name.
                # Rename keys on-the-fly to match the current implementation.
                renamed = {}
                for k, v in state.items():
                    if k.startswith("lstm."):
                        renamed["rnn." + k[5:]] = v
                    else:
                        renamed[k] = v
                model.load_state_dict(renamed)

            print(f"[WorldModelServer] Loaded model from {model_path}")
        except Exception as e:
            print(f"[WorldModelServer] Failed to load {model_path}: {e}")
    model.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
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
    start_udp_world_model_server()
