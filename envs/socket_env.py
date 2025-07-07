import gymnasium as gym
import numpy as np
import socket
import torch
import subprocess
import sys
from time import sleep, perf_counter
from threading import Thread, Event
from servers.constants import (
    ARROW_DELAY,
    WAIT_DELAY,
    NON_ARROW_DELAY,
    ARROW_IDX,
    WAIT_IDX,
)
from typing import Optional
from config import EnvConfig

from utils.observations import LocalObs
from utils.intrinsic import E3BIntrinsicReward
from utils.cosine import cosine_distance

class SocketAppEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=1000,
        device="cuda",
        action_dim=7,
        state_dim=384,
        action_host="127.0.0.1",
        action_port=5005,
        reward_host="127.0.0.1",
        reward_port=5006,
        embedding_model="facebook/dinov2-with-registers-small",
        combined_server=True,
        start_servers=True,
        enable_logging=True,
        use_world_model=True,
        world_model_host="127.0.0.1",
        world_model_port=5007,
        world_model_path="lab/scripts/mlp_world_model.pt",
        world_model_type="mlp",
        world_model_interval=5,
        world_model_time=1.0,  # unused
        config: Optional[EnvConfig] = None,
    ):

        if config is not None:
            max_steps = config.max_steps
            device = config.device
            action_dim = config.action_dim
            state_dim = config.state_dim
            action_host = config.action_host
            action_port = config.action_port
            reward_host = config.reward_host
            reward_port = config.reward_port
            embedding_model = config.embedding_model
            combined_server = config.combined_server
            start_servers = config.start_servers
            enable_logging = config.enable_logging
            use_world_model = config.use_world_model
            world_model_host = config.world_model.host
            world_model_port = config.world_model.port
            world_model_path = config.world_model.model_path
            world_model_type = config.world_model.model_type
            world_model_interval = config.world_model.interval_steps
            world_model_time = config.world_model.time_interval  # unused

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
        self.start_servers = start_servers
        self.enable_logging = enable_logging
        self.use_world_model = use_world_model
        self.wm_addr = (world_model_host, world_model_port)
        self.world_model_path = world_model_path
        self.world_model_type = world_model_type
        self.wm_interval_steps = int(max(1, world_model_interval))
        # ``world_model_time`` was previously used to throttle requests based on
        # wall clock.  The reward from the world model is now triggered purely
        # on step intervals to avoid irregular spacing in the logged values.
        self._server_processes = []
        self._logger = None
        self._last_action_time = perf_counter()

        self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if combined_server:
            self.reward_socket = self.action_socket
            self.reward_addr = self.action_addr
        else:
            self.reward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reward_socket.settimeout(0.2)

        if self.use_world_model:
            self.wm_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.wm_socket.settimeout(0.2)
        else:
            self.wm_socket = None
        self.obs_history = []

        self.obs_encoder = LocalObs(source=1, mode="dino", model_name=embedding_model, device=device, embedding_dim=state_dim)
        self.intrinsic = E3BIntrinsicReward(latent_dim=state_dim, decay=1.0, ridge=0.1, device=device)

        if self.enable_logging:
            from utils import logger
            self._logger = logger

        if self.start_servers:
            self._launch_servers()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self._send_reset()
        self.intrinsic.reset()
        self.obs_history.clear()
        emb_np = self.obs_encoder.get_embedding()
        return emb_np.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        now = perf_counter()
        elapsed = now - self._last_action_time
        delay = 0.0
        if action in ARROW_IDX:
            delay = ARROW_DELAY
        elif action == WAIT_IDX:
            delay = WAIT_DELAY
        else:
            delay = NON_ARROW_DELAY
        self._send_action(action)
        if delay > 0 and elapsed < delay:
            sleep(delay - elapsed)
        self._last_action_time = perf_counter()

        obs_np = self.obs_encoder.get_embedding()
        extrinsic = self._get_reward()
        intrinsic = self.intrinsic.compute(obs_np)
        model_bonus = 0.0
        self.obs_history.append(obs_np.copy())
        if len(self.obs_history) > 30:
            self.obs_history.pop(0)
        if (
            self.use_world_model and
            len(self.obs_history) == 30 and
            self.step_count % self.wm_interval_steps == 0
        ):
            context = np.stack(self.obs_history, axis=0).astype(np.float32) * 10.0
            try:
                self.wm_socket.settimeout(0.2)
                self.wm_socket.sendto(context.tobytes(), self.wm_addr)
                pred_bytes, _ = self.wm_socket.recvfrom(65535)
                # test here for timeout error
                pred = np.frombuffer(pred_bytes, dtype=np.float32)
                if pred.size == self.state_dim:
                    dist = cosine_distance(pred, obs_np)
                    model_bonus = -dist*10
            except socket.timeout:
                model_bonus = 99.99
            except Exception:
                model_bonus = 99.99

        reward = extrinsic + intrinsic + model_bonus

        if self._logger is not None:
            self._logger.log_scalar("Reward/Extrinsic", extrinsic, self.step_count)
            self._logger.log_scalar("Reward/Intrinsic", intrinsic, self.step_count)
            self._logger.log_scalar("Reward/ModelBonus", model_bonus, self.step_count)
            self._logger.log_scalar("Reward/Total", reward, self.step_count)

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {
            "extrinsic": float(extrinsic),
            "intrinsic": float(intrinsic),
            "model_bonus": float(model_bonus),
        }
        return obs_np.astype(np.float32), reward, terminated, truncated, info

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
        if self.wm_socket is not None:
            self.wm_socket.close()
        for proc in self._server_processes:
            proc.terminate()
            proc.wait(timeout=1)

    def _launch_servers(self):
        """Launch reward/action servers as subprocesses if ports are free."""
        def port_in_use(addr):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.bind(addr)
            except OSError:
                s.close()
                return True
            s.close()
            return False

        if self.combined_server:
            if not port_in_use(self.action_addr):
                cmd = [sys.executable, '-m', 'servers.action_server']
                p = subprocess.Popen(cmd)
                self._server_processes.append(p)
        else:
            if not port_in_use(self.reward_addr):
                cmd = [sys.executable, '-m', 'servers.reward_server']
                p = subprocess.Popen(cmd)
                self._server_processes.append(p)

        if self.use_world_model and not port_in_use(self.wm_addr):
            cmd = [
                sys.executable, '-m', 'servers.world_model_server',
                '--model-path', self.world_model_path,
                '--model-type', self.world_model_type,
                '--host', self.wm_addr[0],
                '--port', str(self.wm_addr[1])
            ]
            p = subprocess.Popen(cmd)
            self._server_processes.append(p)
