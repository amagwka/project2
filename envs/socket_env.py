import gymnasium as gym
import numpy as np
import socket
import torch
from time import sleep

from utils.observations import LocalObs
from utils.intrinsic import E3BIntrinsicReward

class SocketAppEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 max_steps=1000,
                 device="cuda",
                 action_dim=7,
                 state_dim=384,
                 action_host="127.0.0.1", action_port=5005,
                 reward_host="127.0.0.1", reward_port=5006,
                 embedding_model="facebook/dinov2-with-registers-small",
                 combined_server=False):

        super().__init__()
        self.max_steps = max_steps
        self.device = device
        self.step_count = 0
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # When combined_server=True, both actions and reward queries go
        # through the same UDP socket/port.  This matches the behaviour of
        # ``start_combined_udp_server`` in ``action_module.py``.
        self.combined_server = combined_server

        self.action_addr = (action_host, action_port)
        self.reward_addr = (reward_host, reward_port)

        self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if combined_server:
            self.reward_socket = self.action_socket
            self.reward_addr = self.action_addr
        else:
            self.reward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reward_socket.settimeout(0.2)

        self.obs_encoder = LocalObs(source=1, mode="dino", model_name=embedding_model, device=device)
        self.intrinsic = E3BIntrinsicReward(latent_dim=state_dim, decay=1.0, ridge=0.1, device=device)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self._send_reset()
        self.intrinsic.reset()
        emb_np = self.obs_encoder.get_embedding()
        return emb_np.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        self._send_action(action)
        sleep(0.07)

        obs_np = self.obs_encoder.get_embedding()
        extrinsic = self._get_reward()
        intrinsic = self.intrinsic.compute(obs_np)
        reward = extrinsic + intrinsic

        terminated = False
        truncated = self.step_count >= self.max_steps
        return obs_np.astype(np.float32), reward, terminated, truncated, {}

    def _send_action(self, action_idx):
        msg = str(action_idx).encode()
        self.action_socket.sendto(msg, self.action_addr)
        if self.combined_server:
            # Combined server replies to every command. Read and discard
            # the acknowledgement to keep the socket state clean for the
            # next reward query.
            try:
                self.reward_socket.recvfrom(32)
            except Exception:
                pass

    def _get_reward(self):
        try:
            self.reward_socket.sendto(b"GET", self.reward_addr)
            data, _ = self.reward_socket.recvfrom(32)
            return float(data.decode().strip())
        except Exception:
            return 0.0

    def _send_reset(self):
        try:
            self.reward_socket.sendto(b"RESET", self.reward_addr)
            self.reward_socket.recvfrom(32)
        except Exception:
            pass

    def render(self):
        pass  # No GUI

    def close(self):
        self.action_socket.close()
        self.reward_socket.close()
