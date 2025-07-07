import gymnasium as gym
import numpy as np
import socket
import torch
import subprocess
import sys
import importlib
from time import sleep, perf_counter
from threading import Thread, Event
from servers.constants import (
    ARROW_DELAY,
    WAIT_DELAY,
    NON_ARROW_DELAY,
    ARROW_IDX,
    WAIT_IDX,
)
from typing import Optional, Callable
from config import EnvConfig

from utils.observation_encoder import ObservationEncoder
from utils.curiosity_base import CuriosityReward, IntrinsicReward
from utils.intrinsic import E3BIntrinsicReward, BaseIntrinsicReward
from utils.cosine import cosine_distance
from utils.udp_client import UdpClient
from utils.world_model_client import WorldModelClient
from dataclasses import asdict

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
        config: Optional[EnvConfig] = None,
        udp_client: Optional[UdpClient] = None,
        world_model_client: Optional[WorldModelClient] = None,
        obs_encoder: Optional[ObservationEncoder] = None,
        intrinsic_reward: Optional[BaseIntrinsicReward] = None,
        server_launcher: Optional[Callable[["SocketAppEnv"], None]] = None,
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
            ir_config = getattr(config, "intrinsic_reward", None)
            intrinsic_cls_path = getattr(config, "intrinsic_cls", None)
        else:
            ir_config = None
            intrinsic_cls_path = None

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
        self._server_processes = []
        self._logger = None
        self._last_action_time = perf_counter()

        self.udp_client = udp_client or UdpClient(self.action_addr, self.reward_addr, combined_server)
        if self.use_world_model:
            self.wm_client = world_model_client or WorldModelClient(self.wm_addr)
        else:
            self.wm_client = None
        self.obs_history = []

        self.obs_encoder = obs_encoder or ObservationEncoder(
            source=1,
            model_name=embedding_model,
            device=device,
            embedding_dim=state_dim,
        )
        if intrinsic_reward is not None:
            self.intrinsic = intrinsic_reward
        else:
            if intrinsic_cls_path is not None:
                mod_name, cls_name = intrinsic_cls_path.rsplit(".", 1)
                module = importlib.import_module(mod_name)
                cls = getattr(module, cls_name)
                try:
                    self.intrinsic = cls(latent_dim=state_dim, device=device)
                except TypeError:
                    self.intrinsic = cls()
            elif ir_config is not None:
                module = importlib.import_module(ir_config.module_path)
                cls = getattr(module, ir_config.class_name)
                try:
                    self.intrinsic = cls(latent_dim=state_dim, device=device)
                except TypeError:
                    self.intrinsic = cls()
            else:
                self.intrinsic = E3BIntrinsicReward(
                    latent_dim=state_dim,
                    decay=1.0,
                    ridge=0.1,
                    device=device,
                )

        if self.enable_logging:
            from utils import logger
            self._logger = logger

        self._server_launcher = server_launcher

        if self.start_servers:
            if self._server_launcher is not None:
                self._server_launcher(self)
            else:
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
        intrinsic = self.intrinsic.compute(obs_np, self)
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
                pred = self.wm_client.predict(context)
                if pred.size == self.state_dim:
                    dist = cosine_distance(pred, obs_np)
                    model_bonus = -dist * 10
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
        self.udp_client.send_action(action_idx)

    def _get_reward(self):
        return self.udp_client.get_reward()

    def _send_reset(self):
        self.udp_client.send_reset()

    def render(self):
        pass  # No GUI

    def close(self):
        self.udp_client.close()
        if self.wm_client is not None:
            self.wm_client.close()
        if hasattr(self.obs_encoder, "close"):
            self.obs_encoder.close()
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


def create_socket_env(cfg: EnvConfig) -> SocketAppEnv:
    """Instantiate ``SocketAppEnv`` from an ``EnvConfig``."""
    env_kwargs = asdict(cfg)
    wm = env_kwargs.pop("world_model")
    env_kwargs.update({
        "world_model_host": wm["host"],
        "world_model_port": wm["port"],
        "world_model_path": wm["model_path"],
        "world_model_type": wm["model_type"],
        "world_model_interval": wm["interval_steps"],
    })
    env_kwargs.pop("intrinsic_reward", None)
    env_kwargs.pop("intrinsic_cls", None)
    env_kwargs["config"] = cfg
    return SocketAppEnv(**env_kwargs)
